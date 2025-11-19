import gym
import torch
from torch import nn, optim
from time import sleep
from agent.ppo import PPO

from agent.memory import Memory


class ConditionalPPO(PPO):
    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        lr: float = 1e-3,
        lr_a: float = 3e-4,
        lr_c: float = 1e-3,
        lr_g: float = 1e-3,
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
        use_gt_cues: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        input_noise_level: float = 0.

    ) -> None:
        super().__init__(
          env, net, lr, lr_a, lr_c, batch_size, gamma, gae_lambda, horizon,
          epochs_per_step, num_steps, clip, value_coef, entropy_coef,
          no_gui, enable_reward_norm, device, input_noise_level=0.
        )
        self.z_noise_level = input_noise_level
        self.use_gt_cues = use_gt_cues
        self.icue = self.env.slice_cues(self.net.z_type)
        if self.net.finetuning():
            self.lr_g = lr_g
            self.optim.append(optim.Adam(self.net.generator.parameters(), lr=lr_g))


    def train_batch(
        self,
        states: torch.Tensor,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
        cues: torch.Tensor,
    ):
      assert not self.net.is_exploring()
      for opt in self.optim:
          opt.zero_grad()

      if self.use_gt_cues:
        gt_cues = cues[:, self.icue[0]:self.icue[1]]
        values, _, _, cue_pred = self.net(states, gt_cues)
      else:
        values, _, _, cue_pred = self.net(states)

      values = values.squeeze(1)

      policy = self.net.pi()
      entropy = policy.entropy().mean()
      log_probs = policy.log_prob(old_actions) # watchout for large abs values here

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

      value_loss = nn.MSELoss()(values, value_target) # TD^2 = V_t - V_t'

      entropy_loss = -entropy

      loss = (
          policy_loss
          + self.value_coef * value_loss
          + self.entropy_coef * entropy_loss
      )

      loss.backward()

      if self.net.finetuning():
        gen_loss = self.net.generator.compute_loss(cues, cue_pred)
        gen_loss.backward()
        L = gen_loss.item()
      else:
        L = loss.item()

      for opt in self.optim:
          opt.step()

      return L, policy_loss.item(), value_loss.item(), entropy_loss.item()


    def collect_trajectory(self, num_steps: int, delay_ms: int = 0) -> Memory:
      states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

      icue = self.env.slice_cues(self.net.z_type)
      for t in range(num_steps):
          # Run one step of the environment based on the current policy
          state = self.state

          if self.z_noise_level > 0:
            assert self.input_noise_level == 0, "image inputs should not be noisy"
            noised_cues = self.env.apply_gaussian_noise(self.env.compute_cues(), self.z_noise_level)
            cues = torch.tensor(noised_cues, dtype=torch.float32, device=self.device).unsqueeze(0)
          else:
            cues = self._to_tensor(self.env.compute_cues())

          if self.use_gt_cues:
            gt_cues = cues[:, icue[0]:icue[1]]
            outputs = self.net(state, gt_cues)
          else:
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

          next_state = self._to_tensor(next_state)

          # Store the transition for current state
          states.append([self.state, cues])
          actions.append(action)
          rewards.append(reward)
          log_probs.append(log_prob)
          values.append(value)
          dones.append(done)

          self.state = next_state

          if not self.no_gui:
              self.env.render()

          if delay_ms > 0:
              sleep(delay_ms / 1000)

      # Get value of last state (used in GAE)
      cues = self._to_tensor(self.env.compute_cues())
      gt_cues = cues[:, icue[0]:icue[1]]
      outputs = self.net(self.state, gt_cues)
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
