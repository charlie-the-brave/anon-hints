import gym
#import dm_control as dmc # TODO: next
from gym import spaces
from gym.wrappers.pixel_observation import PixelObservationWrapper
import numpy as np
import torch
import cv2 as cv


# visual policies only: set to false to
# visualise rollouts during eval runs
SHOW_RAW_INPUTS = True


class PixelObserver(PixelObservationWrapper):
    """ provides normalised 64 x 64 pixel observations for policies """
    def __init__(self, gym_env, should_modify=False):
        self.should_modify = should_modify
        super().__init__(gym_env)

        self.RESIZE_SHAPE = (64,64,3)
        self.observation_space['pixels'] = spaces.Box(0, 1.0, self.RESIZE_SHAPE, np.float32)
        self.observation_space = self.observation_space['pixels']
        self.pixels = None

    def _resize(self, obs):
        obs = obs.astype(np.float32) / 255.
        #return cv.resize(obs, (96,96), obs, interpolation=cv.INTER_AREA)
        return cv.resize(obs, self.RESIZE_SHAPE[:2], obs, interpolation=cv.INTER_AREA)
        #return cv.resize(obs, (192,192), obs, interpolation=cv.INTER_AREA)

    def get_pixels(self):
        assert self.pixels is not None, 'Must call reset to get initial obs'
        return self.pixels

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        downsized_obs = self._resize(ret[0]['pixels']) if isinstance(ret, tuple) else self._resize(ret['pixels'])
        if self.should_modify: downsized_obs = 0.5 * (downsized_obs - 1.0)
        self.pixels = downsized_obs
        return (downsized_obs, *ret[1:]) if isinstance(ret, tuple) else downsized_obs

    def step(self, action):
        ret = super().step(action)
        downsized_obs = self._resize(ret[0]['pixels']) if isinstance(ret, tuple) else self._resize(ret['pixels'])
        if self.should_modify: downsized_obs = 0.5 * (downsized_obs - 1.0)
        self.pixels = downsized_obs
        return downsized_obs, *ret[1:]


class ControlEnv(gym.Wrapper):
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        self.env = gym.make(env_name)
        self.env_name = env_name
        super().__init__(self.env)
        if needs_pixels:
            self.env = PixelObserver(self.env)

        self.frame_skip = frame_skip
        self.show_raw_pixels = SHOW_RAW_INPUTS

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes = 0
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.m_actions = self.env.action_space.n if self.is_discrete else self.env.action_space.shape[0]
        self.use_continuous_action_rep = False
        # needed for dagger (stablebaselines) exps for gym <0.26
        self.render_mode = 'rgb_array' if needs_pixels else 'human'
        # never use step normalised for discrete action spaces
        enable_normalised_actions = enable_normalised_actions and not self.is_discrete
        self.step = self._step_normalised if enable_normalised_actions else self._step

    def to_action_space(self, action: np.ndarray):
        """ simply convert tensors in [-1, 1] to np arrays before passing to env apis """
        if self.is_discrete:
            return action.squeeze(0).astype('int')
        else:
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            return action.astype('float')

    def safe_ratio(self, a, b):
        return a / max(1e-1, abs(b)) * np.sign(b)

    def shape_reward(self, reward):
        return np.clip(reward, -1, 1)

    def get_observation(self):
        """ provides visualisation for vector states (as normalised numpy array) """
        if self.show_raw_pixels:
            if isinstance(self.env, PixelObserver):
                return self.env.get_pixels()
            else:
                # envs that already provide pixels as state e.g., car racing
                obs = self.env.render(mode='rgb_array').astype('float') / 255.
                return cv.resize(obs, (96,96), obs, interpolation=cv.INTER_AREA)
        else:
            return self.env.render(mode='rgb_array').astype('float') / 255.

    def get_state(self):
        """ returns internal state, override as needed """
        return self.env.state

    def apply_gaussian_noise(self, thing, noise_scale):
        thing = np.array(thing)
        return thing + np.random.normal(0, noise_scale, thing.shape)

    def apply_dropout_noise(self, thing):
        thing = np.array(thing)
        return thing * np.random.randint(0, 2, thing.shape)

    def get_total_reward(self):
        # unshaped total reward from action repeats
        return self.total_reward

    @property
    def action_space(self):
        if self.is_discrete:
            if self.use_continuous_action_rep:
                space = spaces.Box(np.array([-1 for i in range(self.m_actions)]),
                                   np.array([1 for i in range(self.m_actions)]))
            else:
                space = spaces.Box(np.array([i for i in range(self.m_actions)]),
                                   np.array([i+1 for i in range(self.m_actions)]))
            return space
        else:
            return self.env.action_space

    def reset(self, **kwargs):

        self.t = 0
        self.last_reward_step = 0
        self.n_episodes += 1
        self.total_reward = 0

        return self.env.reset(**kwargs)

    def _step_normalised(self, action):
        """ maps normalised action into env action space before stepping """
        scale = self.env.action_space.high - self.env.action_space.low
        shift = self.env.action_space.low
        return self._step(scale * action + shift)

    def _step(self, action):
        """ rolls sim one timestep forward with (1d vector) action repeat  """
        self.t += 1

        # NOTE: process actions to avoid perplexing type errors
        # e.g., numpy._core.numeric for float32 ndarrays
        action = self.to_action_space(action)

        total_reward = 0
        state, done, info = None, False, None
        for _ in range(self.frame_skip + 1):
            state, reward, done, info = self.env.step(action)
            self.total_reward += reward
            total_reward += reward

            if reward > 0:
                self.last_reward_step = self.t

        if self.t - self.last_reward_step > 30:
            done = True

        reward = total_reward / (self.frame_skip + 1)

        return state, reward, done, info#, torch.tensor(action)


# todo - locomotion tasks
#class DMControlEnv(dmc.Wrapper):
#    pass
