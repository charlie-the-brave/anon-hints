from typing import Tuple
from os.path import join
import gym
import numpy as np
import torch
from torch import nn, optim
from time import sleep

from agent.memory import Memory, Dataset


class PPO:
    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        lr: float = 1e-3,
        lr_a: float = 3e-4,
        lr_c: float = 1e-3,
        batch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon: int = 1024,
        epochs_per_step: int = 5,
        num_steps: int = 1000,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        no_gui: bool = True,
        enable_reward_norm: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        input_noise_level: float = 0.

    ) -> None:
        self.device = device
        self.env = env
        self.net = net.to(device)

        self.lr = lr
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.gamma = gamma
        self.horizon = horizon
        self.epochs_per_step = epochs_per_step
        self.num_steps = num_steps
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.no_gui = no_gui
        self.enable_reward_norm = enable_reward_norm
        self.input_noise_level = input_noise_level

        self.optim = [optim.Adam(self.net.actor_fc.parameters(), lr=self.lr_a)]
        self.optim.append(optim.Adam(self.net.critic.parameters(), lr=self.lr_c))
        # (!) WARN: update list of filtered out parameters -- e.g., actor, critic, and generator
        non_gac_params = [ p[1] for p in self.net.named_parameters() if not self.is_GAC_parameter(p[0]) ]
        if len(non_gac_params) > 0: self.optim.append(optim.Adam(non_gac_params, lr=self.lr))

        self.state = self._to_tensor(env.reset())
        self.alpha = 1.0


    def train_batch(
      self,
      states: torch.Tensor,
      old_actions: torch.Tensor,
      old_log_probs: torch.Tensor,
      rewards: torch.Tensor,
      advantages: torch.Tensor,
      old_values: torch.Tensor,
      unused: torch.Tensor
    ):
      assert not self.net.is_exploring()
      for opt in self.optim:
        opt.zero_grad()

      outputs = self.net(states)  # note: identical to states that generated old_values
      values = outputs[0].squeeze(1)

      policy = self.net.pi()
      entropy = policy.entropy().mean()
      log_probs = policy.log_prob(old_actions)  # watchout for large abs values here

      # aggregate probs within batch
      if len(log_probs.shape) > 1:
        log_probs = log_probs.sum(dim=1)
        old_log_probs = old_log_probs.sum(dim=1)

      ratio = (log_probs - old_log_probs).exp()  # same as policy / policy_old
      policy_loss_raw = ratio * advantages
      policy_loss_clip = (
        ratio.clamp(min=1 - self.clip, max=1 + self.clip) * advantages
      )
      policy_loss = -torch.min(policy_loss_raw, policy_loss_clip).mean()

      with torch.no_grad():
          value_target = advantages + old_values  # V_t = (Q_t - V_t) + V_t (discounted future return)

      value_loss = nn.MSELoss()(values, value_target)  # TD^2 = V_t - V_t'

      entropy_loss = -entropy

      loss = (
         policy_loss
        + self.value_coef * value_loss
        + self.entropy_coef * entropy_loss
       )

      loss.backward()

      for opt in self.optim:
        opt.step()

      return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()


    def collect_trajectory(self, num_steps: int, delay_ms: int = 0) -> Dataset:
      states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
      self.state = self._to_tensor(self.env.reset()) # TODO 11/16: does this fs up?

      for t in range(num_steps):
          # Run one step of the environment based on the current policy
          state = self.state
          outputs = self.net(state)
          value = outputs[0].squeeze(0)

          # Take action based on current observation
          policy = self.net.pi()
          action = policy.sample()
          log_prob = policy.log_prob(action)
          # NOTE: actions from non-categorical will keep batch dim;otherwise, will be 1D
          if self.net.action_type != 'discrete': 
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

          next_state, reward, done, _ = self.env.step(action.cpu().numpy())

          if done:
              next_state = self.env.reset()

          # Store the transition for current state
          states.append([self.state, self._to_tensor(self.env.compute_cues())])
          actions.append(action)
          rewards.append(reward)
          log_probs.append(log_prob)
          values.append(value)
          dones.append(done)

          # Update state
          self.state = self._to_tensor(next_state)

          if not self.no_gui:
              self.env.render()

          if delay_ms > 0:
              sleep(delay_ms / 1000)

      # Get value of last state (used in GAE)
      outputs = self.net(self.state)
      final_value = outputs[0].squeeze(0)

      # Compute generalized advantage estimates
      advantages = self._compute_gae(rewards, values, dones, final_value)

      # Convert to tensors
      actions = torch.stack(actions)
      log_probs = torch.stack(log_probs)
      advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
      rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
      values = torch.cat(values)
      dones = torch.tensor(dones, dtype=torch.int32, device=self.device)

      return Memory(states, actions, log_probs, rewards, advantages, values, dones)

    def is_GAC_parameter(self, param: str) -> bool:
        return 'actor' in param or 'critic' in param or 'generator' in param

    def adjust_learning_params(self, **kwargs):
      assert 'step' in kwargs.keys(), 'ppo requires step parameter to set learning rate'
      self._set_step_params(kwargs['step'])

    def save(self, filepath: str):
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath: str, strict_match: bool=True):
        self.net.load_state_dict(torch.load(filepath, map_location=self.device), strict=strict_match)

    def predict(
        self, state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._to_tensor(state)
        outputs = self.net(state)
        value, act = outputs[0], outputs[1:]
        return value, act

    def sample_action(self, state):
        self.net(state)
        action = self.net.pi().sample()
        return action.detach().squeeze(0).cpu().numpy()

    def preprocess_state(self, state):
        return self._to_tensor(state)

    def _normalise_reward(self, reward_batch):
        reward_batch = np.array(reward_batch)
        mean, std = reward_batch.mean(), reward_batch.std()
        return reward_batch - mean / max(std, 1)

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = [0] * len(rewards)

        if self.enable_reward_norm:
            rewards = self._normalise_reward(rewards)

        last_advantage = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (1 - dones[i]) * self.gamma * last_value - values[i]
            advantages[i] = (
                delta + (1 - dones[i]) * self.gamma * self.gae_lambda * last_advantage
            )

            last_value = values[i]
            last_advantage = advantages[i]

        return advantages

    def _to_tensor(self, x):
        # optionally apply noise
        if self.input_noise_level > 0.:
          x = self.env.apply_gaussian_noise(x, self.input_noise_level)
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _set_step_params(self, step):
        # interpolate self.alpha between 1.0 and 0.0
        self.alpha = 1.0 - step / self.num_steps
        assert 0 <= self.alpha <= 1

        # set custom learning rates
        for param_group_a in self.optim[0].param_groups:
            param_group_a["lr"] = self.lr_a * self.alpha
        for param_group_c in self.optim[1].param_groups:
            param_group_c["lr"] = self.lr_c * self.alpha
        # set default learning rates for other components (e.g., encoder)
        for opt in self.optim[2:]:
          for param_group in opt.param_groups:
              param_group["lr"] = self.lr * self.alpha

