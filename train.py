import hydra
import traceback
import torch
from os import makedirs 
from os.path import join, dirname
from omegaconf import OmegaConf

from common.extra import MetricLogger, print_message
from agent.base_policy import make_policy
from agent.base_policy import save_policy
from agent.train_ppo import train_ppo
from env.envs import make_env
#from drq import train_drq

# TODO: comment this
# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


def train_agent(cfg, env, policy, logger, trial, best_reward):
    if cfg.agent_type == 'ppo':
        return train_ppo(cfg, env, policy, logger, str(trial), best_reward)
    elif cfg.agent_type == 'test':
        pass
    elif cfg.agent_type == 'drq':
        raise NotImplementedError("DRQ not implemented yet")
    else:
        raise ValueError(f"Unsupported agent type: {cfg.agent_type}")


@hydra.main(config_path="config", config_name="config.yaml")
def train(cfg):
    if cfg.device == "cuda":
        n_gpus = torch.cuda.device_count()
        assert n_gpus > 0, "no gpus available"
        cfg.device = f"cuda:{int(torch.randint(0, n_gpus, (1,)))}"
        print_message(f"training on device {cfg.device}/{n_gpus}", mode=0)

    # verify alignment of checkpoint configs and agent config
    if cfg.generator_checkpoint_path is not None:
        gen_cfg = OmegaConf.load(join(dirname(cfg.generator_checkpoint_path), 'config.yaml'))
        assert gen_cfg.z_info == cfg.z_info, f"z_info mismatch: not {gen_cfg.z_info} != {cfg.z_info}"
        cfg.generator.cxt_size = gen_cfg.generator.cxt_size
        # TODO: enable
        #cfg.generator.should_finetune = gen_cfg.generator.should_finetune

    if cfg.checkpoint_path is not None:
        chk_cfg = OmegaConf.load(join(dirname(cfg.checkpoint_path), 'config.yaml'))
        assert chk_cfg.z_info == cfg.z_info, f"z_info mismatch: {chk_cfg.z_info} != {cfg.z_info}"
    
    # save config alongside results
    outdir = cfg.out_dir
    makedirs(outdir, exist_ok=True)
    OmegaConf.save(cfg, open(join(outdir, "config.yaml"), 'w'))

    # log metrics
    global_logger = MetricLogger(cfg, f'{cfg.agent_type} metrics (n={cfg.max_trials})')

    # TODO: parallel processing, but careful with best model computation, need to share best_reward
    # maybe at end compare best rewards and save corresponding policy
    # generate training results over multiple trials
    best_reward = -float('inf') # globally track best agent
    for trial in range(cfg.start_trial, cfg.max_trials):
        env, env_args = make_env(cfg)
        policy = make_policy(cfg, env_args)

        try:
            best_reward = train_agent(cfg, env, policy, global_logger, trial, best_reward)
            print_message(f'trial {trial} | best reward {best_reward}', mode=0)
        except Exception as e:
            print_message(f"Training failed on {cfg.device}", mode=1)
            traceback.print_exc()
            save_policy(policy, join(outdir, f"stderr_t-{trial}_{global_logger.epoch()}.pth"))
            open(join(outdir, f'stderr-t-{trial}.log'), 'w').write(traceback.format_exc())
            global_logger.dump(outdir, str(trial))
            raise Exception("Training failed")

    print_message('done :)', mode=3)


if __name__ == '__main__':
    train()
