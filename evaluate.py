import sys
import hydra
import torch
import numpy as np
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from os import makedirs
from os.path import join, dirname, isabs

from common import video
from env.envs import make_env
from common.extra import MetricLogger, print_message, to_tensor, MAE
from agent.base_policy import load_policy, make_policy, update_and_save_config

# TODO: comment this
# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


def evaluate_agent(cfg, env, policy, logger, trial):
    device = torch.device(cfg.device)

    policy.eval()
    policy.to(device)

    z_type = cfg['z_type']
    use_gt_cues = 'ground_truth' in z_type

    if cfg['policy_type'] == 'conditional' and not use_gt_cues:
      policy.generator.create_context() # refresh generator context

    # display large frames for first few rollouts
    env.show_raw_pixels = trial > 3 

    rew = 0
    count = 0
    accuracy = 0.0
    done = False
    n_save = 4
    frames = list()
    progress_indicator = list()
    self_state = to_tensor(env.reset(seed=cfg.seed + trial), device)
    cues = to_tensor(env.compute_cues(), device)
    icue = env.slice_cues(z_type)
    while not done or count < cfg.horizon:
        count += 1
        np_cues = cues.detach().squeeze(0).cpu().numpy()
        
        if cfg['policy_type'] == 'conditional' and use_gt_cues:
            gt_cues = cues[:, icue[0]:icue[1]]
            output = policy(self_state, gt_cues)
        else:
            outp = policy(self_state)
            output = [*outp, cues[:, icue[0]:icue[1]]]

        action = policy.pi().sample().detach()
        # NOTE: actions from non-categorical will keep batch dim;otherwise, will be 1D
        if policy.action_type != 'discrete': action = action.squeeze(0)
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        rew += reward
        prediction = output[3].detach().cpu().numpy()
        accuracy += 1 - MAE(prediction, np_cues[icue[0]:icue[1]])
        progress_indicator.append(env.get_progress(np_cues))
        if trial < n_save and count < 200:
            pred_cues = deepcopy(np_cues)
            pred_cues[icue[0]:icue[1]] = prediction.squeeze(0)
            frames.append((count, env.draw_cues(z_type, pred_cues)))

        self_state = to_tensor(next_state, device)
        cues = to_tensor(env.compute_cues(), device)

    logger.log("rewards", rew, is_train=False)
    logger.log("accuracy", accuracy / count, is_train=False)
    #logger.log("progress_mean", np.mean(progress_indicator), is_train=False)
    #logger.log("progress_std", np.std(progress_indicator), is_train=False)
    env.save_progress(trial, cfg.out_dir, progress_indicator, z_type)
    video.save_as_gif(frames, join(cfg.out_dir, f"traj_{trial}.gif"))
    #print_message(f'{trial} | reward: {rew:.2f} | prediction err:{10000.:.2f}', mode=0)

    return rew


def evaluate_block(id, global_cfg, agent_paths):
    avg_returns = list()
    global_checkpoint_path = global_cfg.checkpoint_path
    printed_agent_list = ('\n\t'.join(agent_paths)).replace(global_checkpoint_path + '/','') if len(agent_paths) > 0 else 'none'
    print_message(f"worker {id} evaluating:\n\t{printed_agent_list}\n", mode=0)
    for agent_path in agent_paths:
        cfg = deepcopy(global_cfg)
        logger = MetricLogger(cfg, f'eval metrics (n={cfg.max_trials})')
        agent_subdir = dirname(agent_path).replace(global_checkpoint_path + '/', '')
        assert agent_subdir != agent_path, "checkpoint path shouldn't have glob characters"

        # update eval config with agent parameters from checkpoint
        cfg.checkpoint_path = join(global_checkpoint_path, agent_subdir)
        cfg.out_dir = join(cfg.out_dir, agent_subdir)
        makedirs(cfg.out_dir, exist_ok=True)
        update_and_save_config(cfg, "checkpoint_path", ['generator_checkpoint_path', 'policy_type', 'agent_type', 'state_type', 'action_type', 'env_name'])
        print_message(f"worker {id} saving to: {cfg.out_dir}", mode=0)
        # run evaluation from scratch multiple times 
        avg_return = 0.
        for trial in tqdm(range(cfg.start_trial, cfg.max_trials)):
            env, env_args = make_env(cfg)
            policy = make_policy(cfg, env_args)
            load_policy(policy, agent_path)
            avg_return += evaluate_agent(cfg, env, policy, logger, trial)

        avg_returns.append((
            agent_subdir,
            avg_return / len(list(range(cfg.start_trial, cfg.max_trials)))
            ))
        logger.dump(cfg.out_dir, cfg.max_trials) # store all trials in the same 

    print_message(f'closing worker {id}', mode=3)
    return avg_returns


def report_best(block_results):
    block_res = [worker for block in block_results for worker in block]
    block_res.sort(key=lambda x: x[1]) 
    printed_best_list = '\n\t'.join(f"{i}: {j}" for (i,j) in reversed(block_res))
    print_message(f"top scores:\n\t{printed_best_list}", mode=0)


@hydra.main(config_path="config", config_name="evaluate.yaml")
def evaluate(cfg):
    if cfg.device == "cuda":
        n_gpus = torch.cuda.device_count()
        cfg.device = f"cuda:{str(int(torch.randint(0, n_gpus, (1,))))}"
        print_message(f"running on device {cfg.device}", mode=0)

    if not isabs(cfg.checkpoint_path): 
        print_message("must use absolute checkpoint path", mode=1)
        sys.exit(-1) 

    # grab potentially multiple checkpoints to evaluate
    global_checkpoint_path = cfg.checkpoint_path 
    agent_paths = glob(f"{global_checkpoint_path}/**/{cfg.agent_model}", recursive=True)

    # set up process blocks with fixed config (processes will internally copy cfg)
    block_size = max(1, len(agent_paths) // (cfg.n_cpus - 1))
    agent_blocks = [(wid, cfg, agent_paths[wid*block_size:(wid+1)*block_size]) for wid in range(cfg.n_cpus - 1)]
    if len(agent_paths) % cfg.n_cpus != 0:
        agent_blocks.append((cfg.n_cpus, cfg, agent_paths[(cfg.n_cpus - 1)*block_size:])) # remainders
    
    # run blocks in parallel
    with mp.Pool(processes=cfg.n_cpus) as pool:
        block_results = pool.starmap(evaluate_block, agent_blocks)
        report_best(block_results)

    print_message('done :)', mode=3)

if __name__ == '__main__':
    evaluate()