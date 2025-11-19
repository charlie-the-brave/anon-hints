import torch
from torch.utils.data import Dataset


class Memory(Dataset):
    def __init__(self, states, actions, log_probs, rewards, advantages, values, dones) -> None:
        super().__init__()

        self.actions = actions
        self.log_probs = log_probs
        self.rewards = rewards
        self.advantages = advantages
        self.values = values
        self.cues = torch.cat([x[1] for x in states]) # ensure decimals are type float32
        self.states = torch.cat([x[0] for x in states])
        self.dones = dones

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # NOTE: for compatibility, maintain this order of items (e.g., reward at 3)
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.rewards[idx],
            self.advantages[idx],
            self.values[idx],
            self.cues[idx],
            self.dones[idx]
        )


class MemoryPool(Dataset):
    def __init__(self, maxsize):
        super().__init__()
        self.head = 0 # for adding new samples
        self.size = 0 # for computing sample indices
        self.maxsize = maxsize
        self.empty = True
        self.actions = None
        self.log_probs = None
        self.rewards = None
        self.advantages = None
        self.values = None
        self.cues = None
        self.states = None
        self.dones = None

    def push(self, memory):
      overwrite_size = list(memory.__dict__.values())[0].shape[0]
      residual = min(self.maxsize - self.head, overwrite_size)
      if self.empty:
        self.empty = False
        for k,v in memory.__dict__.items(): # todo: tile instead of leaving zeros
          self.__dict__[k] = torch.zeros((self.maxsize, *v.shape[1:]) if len(v.shape) > 1 else (self.maxsize, 1),
                                         device=v.device, dtype=v.dtype)
          self.__dict__[k][:overwrite_size, :] = v.unsqueeze(1) if len(v.shape) == 1 else v

        self.head = self.size = overwrite_size
      else:
        for k,v in memory.__dict__.items():
          v = v.unsqueeze(1) if len(v.shape) == 1 else v
          self.__dict__[k][self.head:self.head + residual, :] = v[:residual, :]
          self.__dict__[k][:overwrite_size - residual, :] = v[residual:, :] # only writes when residual != overwrite size

        self.size = min(self.size + overwrite_size, self.maxsize)
        self.head = (self.head +  overwrite_size) % self.maxsize

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # NOTE: for compatibility, maintain this order of items (e.g., reward at 3)
        return (
            self.states[idx % self.size],
            self.actions[idx % self.size],
            self.log_probs[idx % self.size],
            self.rewards[idx % self.size],
            self.advantages[idx % self.size],
            self.values[idx % self.size],
            self.cues[idx % self.size],
            self.dones[idx % self.size]
        )