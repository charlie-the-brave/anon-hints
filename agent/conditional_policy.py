import torch
import numpy as np
from torch import nn
from agent.helpers import policy_discrete, policy_gaussian, make_encoder, make_critic, make_actor, make_backbone
from agent.hint_generator import HintGenerator


class ConditionalPolicy(nn.Module):
  def __init__(self, env_args, state_type, action_type, device) -> None:
    super().__init__()

    self.device = device
    self.z_type = env_args['z_type']
    self.state_dim = env_args['observation_space'].shape
    self.action_dim = env_args['action_space'].shape
    self.z_out_size = env_args['visual_embedding_size']
    self.z_in_size = env_args['z_in_size']
    self.n_actions = self.action_dim[0]
    self.action_type = action_type
    self.state_type = state_type
    self.pi = lambda: policy_discrete(self) if self.action_type == 'discrete' else policy_gaussian(self)
    self.explore_threshold = 0.0
    self.should_finetune = False

    # for continuous actions
    self.sig_eps = 1e-2  # lb for predicted std (should train without this?)
    self.sig_max = 1e4  # ub for predicted std

    #print("Action Dimension:", self.n_actions)

    
  def initialise(self, env_args):
    should_suppress_input = '_sc' in self.z_type
    should_embed_gt = ('_fc' in self.z_type or  # for both gt and predicted
                       '_lc' in self.z_type or  # do not set for mc or fc
                       '_ac' in self.z_type)
    should_concat_to_embed = '_lc' in self.z_type
    should_concat_to_input = '_mc' in self.z_type

    # optional model components - if not training, make sure there are no unused components
    self.encoder = None
    self.pre_process = None
    self.generator = None
    self.dim_converter = None
    self.gt_encoder = None

    # additional channel with conditioning var
    if should_concat_to_input:
      self.state_dim = list(self.state_dim)
      self.state_dim[-1] += self.z_in_size

    if not should_suppress_input:
      self.encoder, self.pre_process, _out_size = make_encoder(self.state_type, self.state_dim, self.device)
      # dim_converter converts encoder output to desired z out size
      if self.state_type == 'vector':
        self.z_out_size = _out_size
      elif _out_size != self.z_out_size:
        self.dim_converter = nn.Sequential(*[nn.Linear(_out_size, self.z_out_size), nn.ReLU()]).to(self.device)

    out_size = self.z_out_size
    if should_embed_gt: # todo: lstm?
      self.gt_encoder = nn.Sequential(*[
        nn.Linear(self.z_in_size, 64),
        nn.ReLU(),
        nn.Linear(64, out_size)
      ]).to(self.device)

    # only for concatenating conditioning variable to obs embedding
    if should_concat_to_embed:
      self.z_out_size += out_size

    # actor/critic for data collection
    self.actor_fc, self.logits = make_actor(self.action_type, self.n_actions, self.z_out_size, self.device)
    self.critic = make_critic(self.state_type, self.z_out_size, self.device)


    # TODO: explore ViTs with seq of frames?
    if 'predicted' in env_args['z_type']:
      self.generator = HintGenerator(self.device, env_args['cxt_size'])
      self.generator.initialise(env_args, scale='small')

  def set_finetune(self, should_finetune: bool):
      self.should_finetune = self.generator is not None and should_finetune

  def finetuning(self):
      return self.should_finetune

  def is_exploring(self):
      return self.explore_threshold > 0.0

  def condition(self, x, z):
    """ computes action using input data and conditioning info z (predicted/ground truth) """
    raise NotImplementedError()

  def forward(self, x, z=None):
    if z is None: # predict z from context (1 x z_in_size)
      z, confidence = self.generator(x)
      z = z.detach() # stop gradient to frozen generator
    # apply conditioning to input (1 x z_out_size)
    y = self.condition(x, z)
    # estimate state-value (1x1)
    value = self.critic(y)
    # generate action logits (1 x m)
    self.logits = self.actor_fc(y)

    return value, self.logits, y, z


class StateCondPolicy(ConditionalPolicy):
  def __init__(self, env_args, state_type, action_type, device) -> None:
    super().__init__(env_args, state_type, action_type, device)
    self.max_context = 1
    self.z_out_size = self.z_in_size * 4  # todo: very arbitrary...
    self.visual_embedding_size = 0

  def condition(self, _, z):
    if 'noise' in self.z_type:
      return self.gt_encoder(torch.rand_like(z))
    else:
      assert z is not None
      return self.gt_encoder(z)


class FiLMPolicy(ConditionalPolicy):
  """
  a conditional policy that takes as input images and a conditioning vector
  implementation of feature-wise linear modulation conditioning scheme
  idea is to apply affine transform to at the end of residual blocks in CNN
    residual blocks are a seq of conv layers interleaved with batch norm and skip connections
    the transform is applied right before the last activation (usually ReLU)
  each parameter of the transform are given by linear transform of conditional emb
  e.g., A = nn.linear(cv), b = nn.linear(cv) ---> x = A * x + b

  https://arxiv.org/abs/1709.07871
  original implementation for visual qa
  https://github.com/ethanjperez/film/blob/master/vr/models/filmed_net.py
  a simplified implementation for language conditioned policies
  https://github.com/ShengranHu/Thought-Cloning/blob/05c53e940f53f31236a293932766a1b6629cbaed/babyai/model.py#L24
  """
  def __init__(self, env_args, state_type, action_type, device, max_context=1) -> None:
    super().__init__(env_args, state_type, action_type, device)
    state_dim = env_args['observation_space'].shape
    self.max_context = max_context
    self.visual_embedding_size = 4096
    # override base class initialisation of these components
    self.pre_process = lambda x: x.permute((0, 3, 1, 2))
    self.encoder = nn.Sequential(  # expecting b x c x h x w
      nn.Conv2d(state_dim[2], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
    ).to(device)
    x = torch.zeros(1, *state_dim, device=device)
    x = self.pre_process(x)
    x = self.encoder(x)
    #x = nn.Flatten(x)
    self.z_out_size = int(np.prod(x.size())) # must use np to avoid creating tensor

    self.A = nn.Linear(self.z_in_size, self.z_out_size).to(device)
    self.b = nn.Linear(self.z_in_size, self.z_out_size).to(device)

  def condition(self, x, z):
    # note: input shape should be b x h x w x c
    y = self.pre_process(x)
    y = self.encoder(y)
    y = y.reshape(y.shape[0], -1)
    y = self.A(z) * y
    y = self.b(z) + y
    y = nn.functional.relu(y) # todo: batch norm?
    return y


class ConcatCondPolicy(ConditionalPolicy):
  def __init__(self, env_args, state_type, action_type, device, max_context=1) -> None:
    super().__init__(env_args, state_type, action_type, device)
    self.max_context = max_context
    self.visual_embedding_size = env_args["visual_embedding_size"]

  def condition(self, x, z):
    # note: input shape should be b x h x w x c
    y = self.pre_process(x)
    y = self.encoder(y)
    if self.dim_converter is not None:
      y = self.dim_converter(y)

    # todo 3/3: efficiently incorporate context
    if 'noise' in self.z_type:
        _z = self.gt_encoder(torch.rand_like(z))
    else:
        assert z is not None
        _z = self.gt_encoder(z)

    return torch.cat([y, _z], dim=1)


class AddCondPolicy(ConditionalPolicy):
  def __init__(self, env_args, state_type, action_type, device) -> None:
    super().__init__(env_args, state_type, action_type, device)
    self.max_context = 1
    # no input preprocessing needed

  def condition(self, x, z):
    # note: input shape should be b x h x w x c
    y = self.pre_process(x)
    y = self.encoder(y)
    if self.dim_converter is not None:
      y = self.dim_converter(y) # match dimensions of z embedding

    if 'noise' in self.z_type:
        _z = self.gt_encoder(torch.rand_like(z))
    else:
        assert z is not None
        _z = self.gt_encoder(z)

    return y + _z


class ImgConcatCondPolicy(ConditionalPolicy):
  def __init__(self, env_args, state_type, action_type, device) -> None:
    super().__init__(env_args, state_type, action_type, device)
    self.max_context = 1
    self.visual_embedding_size = env_args["visual_embedding_size"]

  def condition(self, x, z):
    # note: input shape should be b x h x w x c
    mask_shape = list(x.shape[:-1])
    mask_shape.append(1)  # tile only one channel
    if 'noise' in self.z_type:
      mask_shape[-1] = self.z_in_size
      z_mask = torch.rand(mask_shape).permute((0,3,1,2)).to(x.device)
    else:
      # match dims of obs, b x c x w x h
      # works for batched items; checkout z_mask.T[i,:,:,:] vs s[i,:,...]
      assert z is not None
      z_mask = z.repeat(*mask_shape[1:],1).permute((2,3,0,1)).to(x.device)  # works for batch item
      #z_mask = torch.tile(z, mask_shape).to(x.device) # works for single item

    x = self.pre_process(x)
    y = torch.cat([x, z_mask], dim=1)
    if self.dim_converter is None:
      return self.encoder(y)
    else:
      y = self.encoder(y)
      return self.dim_converter(y)
