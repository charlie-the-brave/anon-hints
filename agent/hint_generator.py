import hydra
import torch
import numpy as np
from os import makedirs
from os.path import join, exists, dirname
from copy import deepcopy
from collections import deque
from omegaconf import OmegaConf
from itertools import chain
from torchvision import transforms
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, HuberLoss
from torch.utils.data import Dataset, DataLoader
from env import wrappers
from common import video
from agent.helpers import make_backbone
from common.extra import print_message, free_memory, MAE, to_tensor


# TODO: comment this
# Suppress all warnings
import math
import warnings
warnings.filterwarnings("ignore")


HORIZON = 256
TIMEOUT = 512


hint_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
])


class HintGenerator:
    def __init__(self, device: torch.device, context_size):
        self.device = device
        self.backbone = None
        self.heads = None
        self.context = None
        self.context_size = context_size
        self.transforms = hint_transform


    def initialise(self, env_args: dict, scale: int):
        """ create env specific model components """
        # TODO: remove these after exps
        if 'mae' in env_args['z_type']:
            self.loss_type = 'mae'
        elif 'mse' in env_args['z_type']:
            self.loss_type = 'mse'
        elif 'hub' in env_args['z_type']:
            self.loss_type = 'huber'
        else:
            self.loss_type = 'mse'

        self.backbone, self.heads = make_backbone(env_args, device=self.device, hidden_size=32, scale=scale, context_size=self.context_size)
        self.create_context()


    def create_context(self):
        self.context = deque(maxlen=self.context_size)
 

    def fill_context(self, observations: torch.Tensor):
        """ fill context with group of observations (batch x contextsize x dims) """
        self.fresh_context = True
        if len(observations.shape) == 4: # add batch dim
            observations = observations.unsqueeze(0)
        [self.context.append(observations[:, 0]) for _ in range(self.context_size - observations.shape[1])] # prefill if not enough obs
        [self.context.append(observations[:, ic]) for ic in range(observations.shape[1])]

    
    def update_context(self, observation: torch.Tensor):
        """ add observation to context """
        self.fresh_context = True
        self.context.append(observation)


    def compute_loss(self, X_regress: torch.Tensor, Y_regress: torch.Tensor, X_class: torch.Tensor, Y_class: torch.Tensor):
        loss = 0.
        # continuous - mse loss
        if X_regress is not None:
            if self.loss_type == 'mse': # TODO: remove
                loss += MSELoss()(X_regress, Y_regress)  # minimises variance of output
            elif self.loss_type == 'mae':
                loss += L1Loss()(X_regress, Y_regress)   # mean absolute error: avoid penalising outliers (high variance targets)
            elif self.loss_type == 'huber':
                loss += HuberLoss(reduction='mean', delta=0.3)(X_regress, Y_regress)   # interpolate btw L1(smooth) and L2
            else:
                loss = float('inf')
        # discrete - cross-entropy loss
        if X_class is not None:
            loss += CrossEntropyLoss()(X_class, Y_class)
        return loss


    def predict(self):
        assert len(self.context) == self.context_size and self.fresh_context == True, "call update_context before predicting"
        entropy = 0. # TODO: output prediction uncertainty 
                     # consider monte carlo dropout, deep ensembles
                     # https://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf
                     # https://arxiv.org/pdf/1506.02142.pdf
                     # compute confidence interval (mu,sig for gaussian outputs)

        # construct model input by stacking context elements
        obs = deepcopy(list(self.context))
        obs = torch.cat(obs, dim=1) # element dims: b x c x h x w
        self.fresh_context = False
        # predict cues from context
        z = self.backbone(obs.to(self.device))
        y = list()
        for hi in self.heads.values():
            y.append(hi(z))
        # post-process - disjoint outputs to cue 
        y = torch.cat(y, dim=0)
        return y, entropy


    # nn.Module inferface
    def __call__(self, x: torch.FloatTensor):
        """ an inference time forward call to prediction (use fill for batches) """
        x = transforms.Grayscale(num_output_channels=1)(x.permute(0,3,1,2)) # convert to greyscale w/ dims b x c x h x w
        self.update_context(x)
        [self.update_context(x) for _ in range(self.context_size - len(self.context))]
        return self.predict()


    def train(self):
        self.create_context() # refresh context
        [item.train() for item in [self.backbone, *self.heads.values()]]


    def eval(self):
        self.create_context() # refresh context
        [item.eval() for item in [self.backbone, *self.heads.values()]]


    def parameters(self):
        return list(chain(*[item.parameters() for item in [self.backbone, *self.heads.values()]]))


    def load(self, filepath: str):
        self.backbone.load_state_dict(torch.load(filepath), strict=True)
        [v.load_state_dict(torch.load(filepath.replace('.pth', f'_h-{k}.pth')), strict=True) for (k,v) in self.heads.items() if exists(filepath.replace('.pth', f'_h-{k}.pth'))]


    def save(self, filepath: str):
        torch.save(self.backbone.state_dict(), filepath)
        [torch.save(v.state_dict(), filepath.replace('.pth', f'_h-{k}.pth')) for (k,v) in self.heads.items() if v is not None]


class HintDataset(Dataset):
    def __init__(self, observations: list[np.ndarray], cues: list[np.ndarray], actions: list[np.ndarray], context_size: int):
        super().__init__()
        self.transforms = hint_transform
        observations = [self.transforms(obs) for obs in observations]
        # front load obs with dupes for windowing in getitem
        self.observations = torch.cat(context_size * [observations[0]] + observations, dim=0).unsqueeze(1)
        self.cues = torch.cat([torch.FloatTensor(x).unsqueeze(0) for x in cues], dim=0)
        self.actions = torch.cat([torch.FloatTensor(x).unsqueeze(0) for x in actions], dim=0)
        self.context_size = context_size


    def __len__(self):
        return len(self.observations) - self.context_size


    def __getitem__(self, idx):
        # recall that obervations are frontloaded with dupes for windowing
        return self.observations[idx:idx + self.context_size], self.cues[idx], self.actions[idx]


def decay_learning_rate(optim: torch.optim.Optimizer, native_lr: float, step: int):
    """ linearly decays learning rate from config specification """ 
    assert 0 <= step <= TIMEOUT
    decay = 1.0 - step / TIMEOUT

    for param_group in optim.param_groups:
        param_group['lr'] = decay * native_lr


# TODO: remove this function and corresponding cases
def collect_fake_batch(env: wrappers.ControlEnv, policy: torch.nn.Module, horizon: int, z_type: str, context_size: int, seed: int=0):
    obs = env.reset() if seed is None else env.reset(seed=seed)
    observations, cues, actions = list(), list(), [np.zeros(env.action_space.shape[0])]
    for _ in range(horizon + 1):
        act = policy.forward(obs).detach().squeeze(0).cpu().numpy()
        observations.append(obs)
        actions.append(act)
        obs, reward, done = env.step(act)[:3]

    if 'constant' in z_type:
        cues = (horizon + 1) * [2 * np.ones(len(env.compute_cues()))]
    else:
        l = len(env.compute_cues())
        cues = [math.cos(np.pi + 2*np.pi*i/horizon) * np.ones(l) for i in range(horizon + 1)]

    return HintDataset(observations, cues, actions[:-1], context_size)


def collect_batch(env: wrappers.ControlEnv, policy: torch.nn.Module, horizon: int, context_size: int, seed: int=None):
    assert policy.training == False, "policy must be fixed for training"

    obs = env.reset() if seed is None else env.reset(seed=seed)
    cue = np.array(env.compute_cues(), dtype=np.float32)
    m = env.action_space.shape 
    observations, cues, actions = list(), [cue], [np.zeros(m[0] if policy.action_type == 'continuous' else len(m))]
    for _ in range(horizon + 1):
        if policy.z_type == 'none':
            act = policy(to_tensor(obs, policy.device)).detach().squeeze(0).cpu().numpy()
        else:
            icue = env.slice_cues(policy.z_type)
            policy(to_tensor(obs, policy.device), to_tensor(cue, policy.device)[:, icue[0]:icue[1]])
            act = policy.pi().sample().detach().squeeze(0).cpu().numpy()
        observations.append(obs)    # [o0, o1, ..., oN]
        cues.append(cue)            # [c0, c1, ..., cN]
        actions.append(act)         # [0, a1, ..., aN]
        obs, cue = env.step(act)[0], np.array(env.compute_cues(), dtype=np.float32)

    return HintDataset(observations, cues, actions, context_size)


def is_converged(loss: float):
    # TODO: better convergence criterion
    return 0 < abs(loss) < 1e-3 


@hydra.main(config_path='../config', config_name='config.yaml')
def train(cfg):

    from env.envs import make_env
    from agent.base_policy import make_policy
    from common.extra import MetricLogger

    # TODO: fix output dir - currently ignoring hydra job subdir
    out_dir = join(hydra.utils.get_original_cwd(),'results', cfg.out_dir)
    makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, open(join(out_dir, "config.yaml"), 'w'))
    print_message(f"saving to {out_dir}", mode=0)

    env, env_args = make_env(cfg)
    test_env, _ = make_env(cfg)
    #test_env.show_raw_pixels = False # show larger images, but currently not working since frames come from data buffer

    device = torch.device(cfg.device)
    if cfg.checkpoint_path is None:
        policy = make_policy(cfg, env_args)
    else:
        _, chk_env_args = make_env(OmegaConf.load(join(dirname(cfg.checkpoint_path), 'config.yaml')))
        policy = make_policy(OmegaConf.load(join(dirname(cfg.checkpoint_path), 'config.yaml')), chk_env_args)
        policy.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device), strict=True)

    fixed_policy = policy.eval()
    hgen = HintGenerator(device, cfg.generator.cxt_size)
    hgen.initialise(env_args, scale='small')
    #optim = torch.optim.Adam(hgen.parameters(), lr=cfg.generator.lr) # not for this prediction task 
    optim = torch.optim.SGD(hgen.parameters(), lr=cfg.generator.lr, weight_decay=cfg.generator.wd, momentum=cfg.generator.mt)
    
    step = 0
    train_horizon = max(1, int(cfg.generator.pct_split * HORIZON))
    test_horizon = max(1, int((1 - cfg.generator.pct_split) * HORIZON))
    assert env_args['z_class_size'] == 0, "testing non-car envs, which have no discrete cues"
    k_regress = env_args['z_in_size'] - env_args['z_class_size']
    icue = env.slice_cues(env_args['z_type'])

    trial = 0 # later, train for multiple trials
    min_loss = float('inf')
    logger = MetricLogger(cfg, 'hint_generator')
    while step < TIMEOUT:
        hgen.train()
        decay_learning_rate(optim, cfg.generator.lr, step) # for adam

        if 'test' in env_args['z_type']: # TODO: remove this case
            train_data = collect_fake_batch(env, fixed_policy, train_horizon, env_args['z_type'], cfg.generator.cxt_size)
        else:
            train_data = collect_batch(env, fixed_policy, train_horizon, cfg.generator.cxt_size) # do not pass seed!
        loader = DataLoader(train_data, batch_size=cfg.generator.batch_size, shuffle=True)

        # TODO: why is actual height > 2?
        loss = 0. 
        for (O, X, A) in loader:
            X_regress = None; Y_regress = None; X_class = None; Y_class = None
            # predict based on context
            hgen.fill_context(O)
            Y, confidence = hgen.predict()
            X = X[:, icue[0]:icue[1]].to(hgen.device)
            # compute loss
            if k_regress > 0:
                X_regress = X[:, :k_regress]; Y_regress = Y[:, :k_regress]
            if env_args['z_class_size'] > 0:
                X_class = X[:, k_regress:]; Y_class = Y[:, k_regress:]
            L = hgen.compute_loss(X_regress, Y_regress, X_class, Y_class)
            # update parameters
            optim.zero_grad()
            L.backward()
            optim.step()
            free_memory([O, X, A])
            # track avg loss
            loss += L.item()

        loss /= len(loader)
        logger.log('generator_loss', loss, is_train=True)

        if loss < min_loss:
            hgen.save(join(out_dir, f"hgen_best.pth"))
            open(join(out_dir, f"hgen_best.txt"), 'a').write(f"{step}, {loss}\n")
            min_loss = loss
        
        if step % cfg.test_interval == 0:
            hgen.eval()
            with torch.no_grad():
                accuracy = 0.
                frames, gt_frames = list(), list()
                if 'test' in env_args['z_type']: # TODO: remove this case
                    test_data = collect_fake_batch(test_env, fixed_policy, test_horizon, env_args['z_type'], cfg.generator.cxt_size, cfg.seed + step)
                else:
                    test_data = collect_batch(test_env, fixed_policy, test_horizon, cfg.generator.cxt_size, cfg.seed + step)

                for i in range(len(test_data)):
                    obs, cue, _ = test_data[i]
                    hgen.fill_context(obs)
                    prediction, confidence = hgen.predict()
                    prediction = prediction.detach().cpu().numpy().squeeze(0)

                    cue = cue.detach().cpu().numpy()
                    target_cue = cue[icue[0]:icue[1]]
                    # TODO: choose more useful prediction performance measure
                    accuracy += 1 - MAE(prediction[:k_regress], target_cue[:k_regress])
                    #accuracy += (prediction[k_regress:] == cue[cue[icue[0]+k_regress:icue[1]]).mean()
                    logger.log('predicted', prediction, is_train=False) # need 2d for logging
                    logger.log('actual', target_cue, is_train=False)    # need 2d for logging
                    #logger.log('confidence', confidence, is_train=False)

                    if i < min(50, len(test_data)):
                        frame = transforms.Grayscale(num_output_channels=3)(obs[-1].tile(3,1,1)).permute(1,2,0).detach().cpu().numpy()
                        frame = (255 * frame).astype(np.uint8)
                        actual_frame = test_env.draw_cues(env_args['z_type'], cue, deepcopy(frame))
                        cue[icue[0]:icue[1]] = prediction[:]
                        predicted_frame = test_env.draw_cues(env_args['z_type'], cue, deepcopy(frame))
                        frames.append((i, predicted_frame))
                        gt_frames.append((i, actual_frame))

                accuracy /= len(test_data)

                logger.log('accuracy', accuracy, is_train=False)
                print_message(f"step {step} | loss v | {loss: .2f} | accuracy ^ | {accuracy:.2f}", mode=0)
                logger.dump(out_dir, trial)
                video.save_as_gif(frames, join(out_dir, f"traj_{step}.gif"))
                video.save_as_gif(gt_frames, join(out_dir, f"traj_{step}_gt.gif"))

        if is_converged(loss):
            break

        step += 1

    hgen.save(join(out_dir, f"hgen_final.pth"))
    print_message('done :)', mode=3)


if __name__ == '__main__':
    train()