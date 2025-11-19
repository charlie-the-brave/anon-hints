from env.car import CarEnv, CarRacing
from env.cartpole import CartEnv
from env.pendulum import PendulumEnv
from env.inverted_pendulum import CartPendulumEnv
from env.acrobot import AcrobotEnv
from env.ant import AntEnv
from env.humanoid import HumanoidEnv
from env.cheetah import CheetahEnv


def make_env(cfg):
    """ a very naive environment factory implementation """
    name = cfg['env_name']
    enable_normalised_actions = cfg['use_normalised_actions']
    import os # NOTE: works around libGL error: failed to load driver: swrast
    os.environ["SDL_VIDEODRIVER"] = "dummy" # hides pygame gui window
    os.environ["DISPLAY"] = ":0"
    os.environ["MUJOCO_GL"] = "egl"

    # synthesize internal z_type 
    if cfg['z_info'] == 'none':
        cfg['policy_type'] = 'default' # `:) in case of sweep with conditional policy type
        cfg['z_type'] = 'none'
    else:
        cfg['z_type'] = str.lower(cfg['z_source'] + '_' + cfg['z_method'] + '_' + cfg['z_info'])

    # pixel-based environments
    if "CarRacing-v1-old" in name:
        env = CarRacing(frame_skip=0, frame_stack=4)
        return env, env.create_policy_args(cfg['z_type'])

    # pixel-optional environments
    has_pixels = cfg['state_type'] == 'observation'
    if "CartPole" in name:
        env = CartEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    elif "Ant" in name: # locomotion
        env = AntEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    elif "Acrobot" in name: # discrete, sparse
        env = AcrobotEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    elif "CarRacing" in name:
        env = CarEnv(name, enable_normalised_actions, frame_skip=0) # sets has_pixels=True
        env.show_raw_pixels = True
    elif "HalfCheetah" in name: # locomotion
        env = CheetahEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    elif "DoublePendulum" in name: # continuous
        env = CartPendulumEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    elif "Pendulum" in name:
        env = PendulumEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    elif "Humanoid" in name: # locomotion
        env = HumanoidEnv(name, enable_normalised_actions, has_pixels, frame_skip=0)
    else:
        raise Exception(f"unsupported env '{cfg['env_name']}'")

    args = env.create_policy_args(cfg['z_type'])
    #TODO: undo 11/16
    #import numpy as np
    #args['observation_space'] = np.ndarray([2])
    args['cxt_size'] = cfg['generator']['cxt_size']
    return env, args
