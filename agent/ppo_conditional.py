import gym
import torch
from torch import nn, optim
from time import sleep
from agent.ppo import PPO

from agent.memory import Memory
from agent.hint_generator import HintGenerator


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
        input_noise_level: float = 0.,
        z_noise_level: float = 0.,
        generator_checkpoint_path: str = None,
        should_finetune: bool = False,
        env_args: dict = None

    ) -> None:
        super().__init__(
          env, net, lr, lr_a, lr_c, batch_size, gamma, gae_lambda, horizon,
          epochs_per_step, num_steps, clip, value_coef, entropy_coef,
          no_gui, enable_reward_norm, device, input_noise_level=0.
        )

        self.icue = self.env.slice_cues(self.net.z_type)
        self.z_noise_level = z_noise_level
        self.use_gt_cues = use_gt_cues

        self.lr_g = lr_g
        self.generator = HintGenerator(net.device, env_args['cxt_size'])
        if not use_gt_cues:
          self.generator.initialise(env_args, scale='small')
          self.generator.set_finetune(should_finetune)
          # optionally load generator from checkpoint
          if generator_checkpoint_path is not None:
            self.generator.load(generator_checkpoint_path)
            self.generator.eval() # (!) disable default training mode
          # optionally finetune generator
          if self.generator.finetuning():
              self.optim.append(optim.Adam([p for p in self.generator.parameters()], lr=lr_g))


    def _to_tensor(self, x):
        # optionally apply noise to input
        if self.input_noise_level > 0.:
          x = self.env.apply_gaussian_noise(x, self.input_noise_level)
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)


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
      # validate frozen generator weights
      gen_wts = torch.sum(torch.tensor([p.sum() for p in self.generator.parameters()])).item()

      assert not self.net.is_exploring()
      for opt in self.optim:
          opt.zero_grad()

      if self.use_gt_cues:
        gt_cues = cues[:, self.icue[0]:self.icue[1]]
        values, _, _, z = self.net(states, gt_cues)
        z_error = nn.MSELoss()(gt_cues, z).mean().item()
      else:
        x, cxt = states[0].squeeze(1), states[1].squeeze(1)
        self.generator.fill_context(cxt)
        predicted_cues, _ = self.generator.predict()
        values, _, _, z = self.net(x, predicted_cues.detach())
        z_error = nn.MSELoss()(cues[:, self.icue[0]:self.icue[1]], z).mean().item()

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

      # (optional) finetune with independent loss
      if self.generator.finetuning():
        self.generator.fill_context(states[1].squeeze(1))
        cue_pred, confidence = self.generator.predict()
        gen_loss = self.generator.compute_loss(cues[:, self.icue[0]:self.icue[1]], cue_pred.detach(), None, None) # for now only regression samples
        (loss + gen_loss).backward()
        L, _ = gen_loss.item(), loss.item()
      else:
        loss.backward()
        L = loss.item()

      for opt in self.optim:
          opt.step()

      if not self.generator.finetuning():
        assert torch.sum(torch.tensor([p.sum() for p in self.generator.parameters()])).item() - gen_wts == 0, "generator weights should not change"

      return L, policy_loss.item(), value_loss.item(), entropy_loss.item(), z_error


    def collect_trajectory(self, num_steps: int, delay_ms: int = 0) -> Memory:
      states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

      icue = self.env.slice_cues(self.net.z_type)
      for t in range(num_steps):
          # Run one step of the environment based on the current policy
          state = self.state

          if self.z_noise_level > 0: # noise cues independently from observations
            assert self.input_noise_level == 0, "image inputs should not be noisy"
            noised_cues = self.env.apply_gaussian_noise(self.env.compute_cues(), self.z_noise_level)
            cues = torch.tensor(noised_cues, dtype=torch.float32, device=self.device).unsqueeze(0)
          else:
            cues = self._to_tensor(self.env.compute_cues())

          if self.use_gt_cues:
            gt_cues = cues[:, icue[0]:icue[1]]
            outputs = self.net(state, gt_cues)
          else:
            predicted_cues, _ = self.generator(state)
            outputs = self.net(state, predicted_cues.detach())

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
          if self.use_gt_cues:
            states.append([self.state, cues])
          else:
            states.append([(self.state, torch.cat(list(self.generator.context), dim=1)), cues])
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
      outputs = self.net(self.state, gt_cues) if self.use_gt_cues else self.net(self.state, self.generator(state)[0].detach())
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


    def _set_step_params(self, step):
        super()._set_step_params(step)
        # set generator learning rate
        if not self.use_gt_cues and self.generator.finetuning():
          for param_group_g in self.optim[3].param_groups:
              param_group_g["lr"] = self.lr_g * self.alpha
