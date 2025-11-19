import pickle
import torch
import numpy as np
import gc as rubbish
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from os.path import join


#
# helpers for tensors
#
def to_tensor(x: np.ndarray, device, typ=torch.float32, batch_dim=True):
    """ converts non-batch array to tensor on device """
    if batch_dim:
      return torch.tensor(x, dtype=typ, device=device).unsqueeze(0)
    else:
      return torch.tensor(x, dtype=typ, device=device)


def MAE(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y)) / len(x)


def free_memory(items):
    #https://discuss.pytorch.org/t/efficient-way-to-run-independent-jobs-on-multiple-gpu-machine/5679/5
    [item.detach().cpu() for item in items if isinstance(item, torch.Tensor)]
    for item in items: del item
    torch.cuda.empty_cache()
    rubbish.collect()


#
# metric loggers and helpers
#
class MetricLogger:
  def __init__(self, cfg, name):
    self.name = name
    self.eval = GenericMetrics('eval', cfg.test_interval)
    if cfg.agent_type == 'ppo':
      self.train = PPOMetrics('train', 1)
    elif cfg.agent_type == 'drq':
      self.train = DrqMetrics('train', 1)
    elif cfg.agent_type == 'test':
      self.train = GenericMetrics('eval', cfg.test_interval)
      self.train.metrics = dict()
    else:
      raise ValueError(f"Unsupported agent type: {cfg.agent_type}")

    self.train.metrics['train/generator_loss'] = list()
    self.eval.metrics['eval/predicted'] = list()
    self.eval.metrics['eval/actual'] = list()

  def epoch(self):
    return len(list(self.train.metrics.values())[0])

  def aggregate(self):
    for k,v in self.train.metrics.items():
      self.train.metrics[k] = np.mean(v, axis=0)
    for k,v in self.eval.metrics.items():
      self.eval.metrics[k] = np.mean(v, axis=0)

  def log(self, key, value, is_train):
    if is_train:
      self.train.metrics[f"train/{key}"].append(value)
    else:
      self.eval.metrics[f"eval/{key}"].append(value)

  def dump(self, outdir, id):
    #assert id < len(self.metrics['train_rewards']), f"{id} must be less than the number of trials"
    for k,v in self.train.metrics.items():
      writeable_key = k.replace('/', '_')
      pickle.dump(v, open(join(outdir, f"t-{id}_{writeable_key}.pkl"), 'wb'))
    for k,v in self.eval.metrics.items():
      writeable_key = k.replace('/', '_')
      pickle.dump(v, open(join(outdir, f"t-{id}_{writeable_key}.pkl"), 'wb'))

    # save as tensoorboard log
    tbl = TBLogger(outdir, id)
    tbl.from_metric_logger(self.train)
    tbl.from_metric_logger(self.eval)
    tbl.close()
    return tbl

  def reset(self):
    # clear all metrics
    for k in self.train.metrics.keys():
      self.train.metrics[k] = list()
    for k in self.eval.metrics.keys():
      self.eval.metrics[k] = list()


class TBLogger:
  def __init__(self, outdir, id):
    self.outdir = outdir
    self.trial_id = id
    self.writer = SummaryWriter(join(outdir, 'tb', str(self.trial_id)))
  
  def from_metric_logger(self, logger):
    for (k,v) in logger.metrics.items():
      v = np.array(v)
      interval = 1 if 'train' in k else logger.interval
      for t, item in enumerate(v):
        if len(v.shape) == 1 or v.shape[1] == 1: # 1D array # TODO: verify not shape[0]
          self.writer.add_scalar(k, item, interval * t)
        else: # 2D array
          [self.writer.add_scalar(f"{k}[{i}]", item[i], t) for i in range(item.shape[0])]

  def close(self):
    self.writer.close()

class GenericMetrics():
  def __init__(self, name, interval):
    self.interval = interval
    self.tb_key_prefix = name 
    self.metrics = dict()
    self.metrics[f'{name}/rewards'] = list()
    self.metrics[f'{name}/accuracy'] = list()


class PPOMetrics(GenericMetrics):
  def __init__(self, name, interval):
    super().__init__(name, interval)
    self.metrics[f'{name}/rewards'] = list()
    self.metrics[f'{name}/pol_loss'] = list()
    self.metrics[f'{name}/val_loss'] = list()
    self.metrics[f'{name}/ent_loss'] = list()
    self.metrics[f'{name}/total_loss'] = list()
    self.metrics[f'{name}/logprobs'] = list()
    self.metrics[f'{name}/values'] = list()


class DrqMetrics(GenericMetrics):
  def __init__(self, name, interval):
    super().__init__(name, interval)
    #self.metrics[f'{name}/values'] = list() 


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


#
# other helpers
#
def print_message(message, mode=None):
  # TODO: fix pprint with these colors; should use \x1b and show without escape
  if mode is None:
    print(message)
  elif mode == 0: # info
    print(bcolors.OKCYAN + message + bcolors.ENDC)
  elif mode == 1: # error
    print(bcolors.FAIL + message + bcolors.ENDC)
  elif mode == 2: # warn
    print(bcolors.WARNING + message + bcolors.ENDC)
  elif mode == 3: # success 
    print(bcolors.OKGREEN + message + bcolors.ENDC)
  else:
    raise NotImplementedError()
