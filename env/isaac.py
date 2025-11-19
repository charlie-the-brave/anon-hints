import math
import sys
import argparse
from omni.isaac.lab.app import AppLauncher

# todo 9/21: start with simple cartpole task running in parallel
help = """
run using isaaclab launcher
.$HOME/Workspace/nvidia/IsaacLab/isaaclab.sh -p ./isaac.py --argX

"""

# invoke isaac omniverse to launch sim
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--env_name", type=str, required=True, help="Name of environments to spawn.")
parser.add_argument("--agent_type", type=str, default='PPO', help="Type of agent to optimise.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch simulation as omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""
"""

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import torch
import gymnasium as gym
import numpy as np
import os
from datetime import datetime
#from utils import video # todo: fix import path

##
# Pre-defined configs
##
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
#from omni.isaac.lab_assets import CARTPOLE_CFG, ANT_CFG, HUMANOID_CFG, SHADOW_HAND_CFG  # isort:skip
#from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
# todo: SPOT_CFG, FRANKA_CFG

seed = 0
num_envs = 20
noise_amp = 0.2
action_amp = 5.0
is_headless = True
max_iterations = 1000000
video_interval = 10000
checkpoint_interval = 10000
video_length = 1000
agent_sigma = 1 # todo: what is this param?

if is_headless:
    args_cli.enable_cameras = True

# load predefined configs?
@hydra_task_config(args_cli.env_name, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # specialise configs
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg["seed"] = seed
    # needed for computing number of minibatches
    agent_cfg["params"]["config"]["minibatch_size"] = 2*agent_cfg["params"]["config"]["horizon_length"]
    agent_cfg["params"]["config"]["max_epochs"] = max_iterations
    agent_cfg["params"]["config"]["save_frequency"] = checkpoint_interval
    agent_cfg["params"]["config"]["save_best_after"] = checkpoint_interval

    # todo: multi-gpu config

    # directory for logging into
    log_root_path = os.path.join("logs", 'rl_games-training', agent_cfg["params"]["config"]["name"])
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.env_name, cfg=env_cfg, render_mode="rgb_array")
    # wrap for video recording
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % video_interval == 0,
        "video_length": video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    
    # todo 9/25: implement similar class to Runner that generates custom algos 
    # cite original python3.10/site-packages/rl_games/torch_runner.py, <LINK>/rl_games/torch_runner.py
    # register ppo.py impl into new custom runner
    # it gets complicated, but example train at ln 1300 in rl_games/common/a2c_common.py
    # env names available at source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/humanoid/__init__.py

    # outputs:
    # tensorboard - logs/rl_games-training/humanoid_direct/2024-09-23_17-57-31/summaries
    # checkpoints - logs/rl_games-training/humanoid_direct/2024-09-23_17-57-31/nn
    # frames - 2024-09-23_17-57-31/videos/train
    # parameters - 2024-09-23_17-57-31/params

    # view metrics:
    # tensorboard --logdir=logs/rl_games-training/humanoid_direct/2024-09-23_17-57-31/summaries
    
    
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # kick off training
    runner.reset()
    runner.run(dict(train=True, play=False, sigma=agent_sigma))

    # close the simulator
    env.close()


if __name__ == '__main__':
    main()

    # close sim app
    simulation_app.close()