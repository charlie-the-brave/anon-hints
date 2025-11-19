import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical, Normal, Beta


DEFAULT_Z_SIZE = 128 # image-based e.g., car racing, atari
MODEL_SCALES = dict(small=1, medium=8, large=16)

class ResNetBlock2(nn.Module):
  def __init__(self, input_size, output_size, stride):
    super().__init__()
    self.block = nn.Sequential(  # expecting b x c x h x w
      nn.Conv2d(input_size, output_size, kernel_size=3, stride=stride, padding=1),
      nn.BatchNorm2d(output_size),
      nn.ReLU(),
      nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(output_size),
      # add non-linearity in fwd call (cleaner)
    )

    if input_size != output_size or stride != 1: # todo: probs not... use conv in resnet?
      #from torchvision import transforms as trans
      #self.subsample = trans.Resize(int(output_size), interpolation=trans.BICUBIC)
      self.subsample = nn.Sequential(  # expecting b x c x h x w
        nn.Conv2d(input_size, output_size, kernel_size=1, stride=stride),
        nn.BatchNorm2d(output_size)
      )
    else:
      self.subsample = None

    self.relu = nn.ReLU()
    self.in_size = input_size
    self.out_size = output_size

  def forward(self, x):
    # todo: test whether subsample gives matching output as first layer in block
    res = self.subsample(x) if self.subsample else x
    y = self.block(x)
    y += res
    y = self.relu(y)
    return y


class ResNet34(nn.Module):
  # referenced from https://github.com/ShengranHu/Thought-Cloning/blob/main/babyai/model.py#L24
  def __init__(self, state_dim, inner_blocks, block_sizes, stride, out_size):
    super().__init__()
    self.pre_process = lambda x: (x - 0.5) # for dataloader, center on [-0.5,0.5]
    hidden_size = 32

    self.blocks = list()
    self.input_layer = nn.Sequential(  # expecting b x c x h x w
      nn.Conv2d(state_dim[2], hidden_size, kernel_size=8, stride=4, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    self.avgpool = nn.AvgPool2d(8, stride=1) # a direct function of conv2d strides in inner block
    self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    # assemble remaining block sequence (32 total layers)
    self._add_inner_block(ResNetBlock2, hidden_size, block_sizes[0], inner_blocks[0], stride[0])
    self._add_inner_block(ResNetBlock2, block_sizes[0], block_sizes[1], inner_blocks[1], stride[1])
    self._add_inner_block(ResNetBlock2, block_sizes[1], block_sizes[2], inner_blocks[2], stride[2])
    self._add_inner_block(ResNetBlock2, block_sizes[2], block_sizes[3], inner_blocks[3], stride[3])
    self.blocks = nn.Sequential(*self.blocks)
    self.fc_out = nn.Linear(block_sizes[-1], out_size)


  def _add_inner_block(self, block_type, in_size, out_size, k_blocks, stride=1):
    self.blocks.append(block_type(in_size, out_size, stride))
    for k in range(1, k_blocks):
      self.blocks.append(block_type(out_size, out_size, stride=1))

  def forward(self, x):
    x = self.pre_process(x)
    y = self.input_layer(x)
    y = self.maxpool(y)
    y = self.blocks(y)
    y = self.avgpool(y) # TODO: currently hardcoded for imagenet 244x244->7x7 or gym 64x64->2x2
    y = nn.Flatten()(y)
    y = self.fc_out(y)
    return y


def make_encoder(state_type, state_dim, device, output_size=-1):
  if state_type == 'observation':
    # note: expecting normalised image, so center pels
    pre_process = lambda x: (2 * x - 1).permute((0, 3, 1, 2))

    encoder = nn.Sequential(  # expecting b x c x h x w
      nn.Conv2d(state_dim[2], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.Flatten(),
    )

    x = torch.zeros(1, *state_dim)
    x = pre_process(x)
    x = encoder(x)
    z_out_size = int(np.prod(x.size())) # must use np to avoid creating tensor
  elif state_type == 'vector':
    pre_process = lambda x: x
    #z_hidden_size = 512
    z_hidden_size = 128
    z_out_size = DEFAULT_Z_SIZE
    encoder = nn.Sequential(
      nn.Linear(state_dim[0], 2 * z_hidden_size),
      nn.ReLU(),
      nn.Linear(2 * z_hidden_size, z_hidden_size),
      nn.ReLU(),
      nn.Linear(z_hidden_size, z_out_size),
    )
  else:
    raise Exception(f'unknown state type "{state_type}"')

  return encoder.to(device), pre_process, z_out_size

def make_critic(state_type, z_out_size, device):
    # Estimates the value of the state
    c_hidden_size = [256, 128]
    return nn.Sequential(
      nn.Linear(z_out_size, c_hidden_size[0]),
      nn.LayerNorm(c_hidden_size[0]),
      nn.Tanh(),
      # previous layers for stabilisation (https://arxiv.org/abs/1607.06450)
      nn.Linear(c_hidden_size[0], c_hidden_size[0]),
      nn.ReLU(),
      nn.Linear(c_hidden_size[0], c_hidden_size[1]),
      nn.ReLU(),
      nn.Linear(c_hidden_size[1], 1),
    ).to(device)


def make_actor(action_type, n_actions, z_out_size, device):
  # Compute logits or params of distribution over actions
  if action_type == 'discrete':
    # logits are category probs
    a_hidden_size = 2 * [128]  # attempt for classic control
    actor = nn.Sequential(
      nn.Linear(z_out_size, a_hidden_size[0]),
      nn.ReLU(),
      nn.Linear(a_hidden_size[0], n_actions),
      nn.Softmax()
    )
    logits = torch.zeros((1, n_actions), device=device, requires_grad=True)
  elif action_type == 'continuous':
    # when using tanh, logits are mean and log(std) in [-1,1], std = exp(logits[n_actions:])
    # listing 11, https://arxiv.org/pdf/2105.07998.pdf
    # https://github.com/NadeemWard/pytorch_simple_policy_gradients/blob/36ffcf649c5e9ef3b22cc7b3a5ca5bc2818a842c/actor-critic/policy.py#L60
    a_hidden_size = 2 * [256]
    # a_hidden_size = [256, 128] car racing
    actor = nn.Sequential(
      nn.Linear(z_out_size, a_hidden_size[0]),
      nn.LayerNorm(a_hidden_size[0]),
      nn.Tanh(),
      # previous layers for stabilisation (https://arxiv.org/abs/1607.06450)
      nn.Linear(a_hidden_size[0], a_hidden_size[0]),
      nn.ReLU(),
      nn.Linear(a_hidden_size[0], 2 * n_actions),
      nn.Sigmoid()  # agents learn faster with sigmoid than with rescaled tanh outputs
    )
    logits = torch.zeros((1, 2 * n_actions), device=device, requires_grad=True)
  else:
    raise Exception(f'unknown state type "{action_type}"')

  return actor.to(device), logits


def make_backbone(env_args, hidden_size, context_size, device, scale='small'):
  assert scale.lower() in MODEL_SCALES.keys(), f'unknown scale "{scale}"'
  scale = scale.lower()

  in_dims = list(env_args['observation_space'].shape)
  in_dims[-1] = context_size # context contains greyscale 1-channel images
  z_type = env_args['z_type']
  z_in_size = env_args['z_in_size']
  z_class_size = env_args['z_class_size']
  z_reg_size = z_in_size - z_class_size
  hidden_size = hidden_size * MODEL_SCALES[scale]

  mlp = [nn.Linear(hidden_size, hidden_size), nn.ReLU(),nn.Linear(hidden_size, z_in_size)]
  block_sizes = [64, 128, 256, 512] * MODEL_SCALES[scale] # output sizes for inner blocks 
  inner_blocks = [3, 4, 6, 3] * MODEL_SCALES[scale]       # number of blocks with same block size
  #stride = [1, 2, 2, 1] * MODEL_SCALES[scale]            # size of convolution shift - feature dim
  stride = [1, 1, 1, 1] * MODEL_SCALES[scale]             # size of convolution shift - feature dim
  backbone, heads = None, None # unused model components break training

  if not ('none' in z_type and 'ground_truth' in z_type):
    # select backbone
    if 'resnet' in z_type:
      backbone = ResNet34(in_dims, inner_blocks, block_sizes, stride, hidden_size)
    else:
      raise Exception(f'unknown backbone in z type "{z_type}"')

    # complete model to regress or classify cues
    if z_class_size > 0 and z_reg_size > 0:
      heads = dict(
        classifier=mlp + [nn.Softmax()],  # for categorical output
        regressor=mlp,
      )
      heads['classifier'] = nn.Sequential(*heads['classifier'])
      heads['regressor'] = nn.Sequential(*heads['regressor'])
    elif z_class_size > 0:
      heads = dict(classifier=mlp + [nn.Softmax()])  # for categorical output
      # todo: rm? don't end with activation
      heads['classifier'] = nn.Sequential(*heads['classifier'])
    elif z_reg_size > 0:  
      heads = dict(regressor=mlp)  # todo 7/10: bad idea to bottleneck?
      heads['regressor'] = nn.Sequential(*heads['regressor'])
    else:
      raise Exception(f"unknown z-type {z_type}")

  [item.to(device) for item in [backbone, *heads.values()]]
  return backbone, heads


def policy_beta(ac_net):
  return Beta(ac_net.alpha, ac_net.beta)


def policy_discrete(ac_net):
  """ outputs action for continuous actor-critic """
  if np.random.random() > ac_net.explore_threshold:
    action_probs = ac_net.logits
  else:
    action_probs = 1 / ac_net.n_actions * torch.ones_like(ac_net.logits)
  return Categorical(logits=action_probs)


def policy_gaussian(ac_net):
  """ outputs action for continuous actor-critic """
  if np.random.random() > ac_net.explore_threshold:
    # sigmoid outputs: map mean to [-1,1], std in [0,1]
    mean = 2 * ac_net.logits[:, :ac_net.n_actions] - 1
    sig = torch.clamp(ac_net.logits[:, ac_net.n_actions:], min=ac_net.sig_eps)
  else:
    mean = torch.zeros_like(ac_net.logits[:, :ac_net.n_actions])
    sig = torch.ones_like(ac_net.logits[:, ac_net.n_actions:])

  assert False not in torch.ge(sig, ac_net.sig_eps), f"std is too small! {sig.min()}"
  assert False not in torch.le(sig, ac_net.sig_max), f"std is diverging! {sig.max()}"
  return Normal(mean, sig)

