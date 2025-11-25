from typing import Mapping, Any

import torch
from torch import nn
from omegaconf import OmegaConf
from os.path import join
#from agent.dqn import CarRacingDQNAgent
from agent.conditional_policy import FiLMPolicy, ImgConcatCondPolicy, AddCondPolicy, ConcatCondPolicy, StateCondPolicy
from agent.helpers import policy_discrete, policy_gaussian
from agent.helpers import make_encoder, make_critic, make_actor


def update_and_save_config(cfg, checkpoint_key, extra_keys=[]):
    # update cfg with agent parameters from checkpoint
    # (algo params in ppo kwards should not change)
    # note: allow env mismatch for cross-env training
    # by changing env name in checkpoint config
    checkpoint_overrides_cfg = {
        k : v for (k,v) in OmegaConf.load(join(cfg[checkpoint_key], "config.yaml")).items()
        if 'z_' in k or k in extra_keys
    }
    cfg.update(checkpoint_overrides_cfg)
    # synthesize internal z_type 
    if cfg['z_info'] == 'none':
        cfg['policy_type'] = 'default' # `:) in case of sweep with conditional policy type
        cfg['z_type'] = 'none'
    else:
        cfg['z_type'] = str.lower(cfg['z_source'] + '_' + cfg['z_method'] + '_' + cfg['z_info'])
    OmegaConf.save(cfg, open(join(cfg.out_dir, "config.yaml"), 'w'))
    return cfg


def load_policy(policy, filepath):
    policy.load_state_dict(torch.load(filepath), strict=True)


def save_policy(policy, filepath):
    torch.save(policy.state_dict(), filepath)


def make_policy(cfg, env_args):
    dv = torch.device(cfg['device'])

    if cfg['checkpoint_path'] is not None:
      update_and_save_config(cfg, "checkpoint_path", ['generator_checkpoint_path', 'policy_type', 'agent_type', 'state_type', 'action_type', 'env_name'])

    if cfg['policy_type'] == 'default': # however, conditional can have z_info none (e.g., sweeps)
        assert cfg['z_info'] == 'none', f"default policy cannot have z_info {cfg['z_info']}"

    if cfg['policy_type'] == 'default' or cfg['z_info'] == 'none':
        return PolicyNet(env_args, cfg['state_type'], cfg['action_type'], dv, -1)
    elif cfg['policy_type'] == 'conditional':
        env_args['visual_embedding_size'] = 4096  # legacy: pulled this param out to config
        return make_conditional_policy(cfg, env_args)
    elif cfg['policy_type'] == 'random':
        return RandomPolicy(env_args['action_space'], cfg['action_type'], dv)
    else:
        raise Exception(f"unsupported policy_type {cfg['policy_type']}")


def make_conditional_policy(cfg, env_args):
    dv = torch.device(cfg['device'])

    if 'predicted' in cfg['z_source']:
      assert cfg['generator_checkpoint_path'] is not None, f"must provide generator checkpoint to train with z_source, {cfg['z_source']}"
      # TODO: update from dir in root result folder not from hydra job subdir
      #update_and_save_config(cfg, "generator_checkpoint_path", ['generator'])
      env_args['generator_checkpoint_path'] = cfg['generator_checkpoint_path']
      env_args['cxt_size'] = cfg['generator']['cxt_size']

    z_type = cfg['z_method'] # note: use updated config (from checkpoint if applicable)
    if 'fc' in z_type:
        if cfg['state_type'] == 'vector':
            raise Exception(f"can't instantiate {cfg['z_type']} for {cfg['state_type']}")
        policy = FiLMPolicy(env_args, cfg['state_type'], cfg['action_type'], dv)
    elif 'mc' in z_type:
        if cfg['state_type'] == 'vector':
            raise Exception(f"can't instantiate {z_type} for {cfg['state_type']}")
        policy = ImgConcatCondPolicy(env_args, cfg['state_type'], cfg['action_type'], dv)
    elif 'ac' in z_type:
        policy = AddCondPolicy(env_args, cfg['state_type'], cfg['action_type'], dv)
    elif 'lc' in z_type:
        policy = ConcatCondPolicy(env_args, cfg['state_type'], cfg['action_type'], dv)
    elif 'sc' in z_type:
        policy = StateCondPolicy(env_args, cfg['state_type'], cfg['action_type'], dv)
    else:
      raise Exception(f"unsupported conditional z_method {z_type}")

    policy.initialise(env_args)
    return policy


class ACNet(nn.Module):
    def __init__(self, state_dim, action_dim, state_type, action_type) -> None:
        super().__init__()

        self.pre_process = lambda x: x
        self.n_actions = action_dim[0]
        self.action_type = action_type
        self.state_type = state_type
        self.explore_threshold = 0.0
        self.pi = lambda: policy_discrete(self) if self.action_type == 'discrete' else policy_gaussian(self)
        self.encoder = None
        self.actor_fc = None
        self.critic = None

        #print("Action Dimension:", self.n_actions)

    def is_exploring(self):
        return self.explore_threshold > 0.0

    def forward(self, x):
        raise NotImplementedError()


class PolicyNet(ACNet):
    def __init__(self, env_args, state_type, action_type, device, encoder_size) -> None:
        state_dim = env_args['observation_space'].shape
        action_dim = env_args['action_space'].shape
        super().__init__(state_dim, action_dim, state_type, action_type)
        # todo 3/27: train decoder separately for debugging

        self.encoder, self.pre_process, z_out_size = make_encoder(state_type, state_dim, device, encoder_size)
        self.actor_fc, self.logits = make_actor(action_type, self.n_actions, z_out_size, device)
        self.critic = make_critic(state_type, z_out_size, device)

        # for continuous actions
        self.sig_eps = 1e-2  # lb for predicted std (should train without this?)
        self.sig_max = 1e4  # ub for predicted std

    def forward(self, x):
        ##TODO: undo 11/16
        #j = x[:, 6:8]
        #j[:, 1] = x[:, 6] + x[:, 7] # double: first and second relative to first
        #x = j
        # note: input shape should be b x h x w x c
        x = self.pre_process(x)

        x = self.encoder(x)

        # Estimate value of the state
        value = self.critic(x)

        # Estimate the parameters of a Categorical distribution over actions
        self.logits = self.actor_fc(x)

        return value, self.logits, x


class RandomPolicy:
    def __init__(self, action_space, action_type, device):
        assert action_type == 'discrete' or action_type == 'continuous'
        self.action_type = action_type
        self.z_type = 'none'
        self.device = device
        self.m_actions = action_space.shape[0]
        self.pre_process = lambda x : torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        # basic module function that may be called
        self.requires_grad = False
        self.eval = lambda: self
        self.train = lambda: self
        self.training = False
        self.parameters = lambda: []


    def to(self, device):
        self.device = device
        return self


    def __call__(self, x, z=None):
        return self.forward(x)


    def forward(self, x):
        x = self.pre_process(x)
        # note: by now, input shape should be b x h x w x c
        if self.action_type == 'discrete': # simulate uniform categorical
            return torch.randint(0, self.m_actions, (1, x.shape[0]), device=self.device)
        elif self.action_type == 'continuous': # simulate uniform continuous in [-1,1]
            return torch.tanh(20*torch.rand(x.shape[0], self.m_actions, device=self.device)-10) # steep saturation
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")