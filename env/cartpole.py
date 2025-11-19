import cv2 as cv
import numpy as np
import pickle
from os.path import join
from env.wrappers import ControlEnv
from torch import FloatTensor
from gym import spaces
#from gym.envs.classic_control import PendulumEnv
from matplotlib import pyplot as plt


class CartEnv(ControlEnv):
    """ A version of pendulum-v1 that returns computes custom state cues """
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        self.is_ood = 'v1r' in env_name or 'v2r' in env_name
        # select goal angle (radians, counter-clockwise)
        self.goal = np.deg2rad(0)
        if self.is_ood: # setup random goal in reset
            env_name = env_name.replace('v1r', 'v1').replace('v2r', 'v1')

        super().__init__(env_name, enable_normalised_actions, needs_pixels, frame_skip)
        self.target_height = np.cos(self.goal)
        self.steps_above_height = 0
        self.timer = 0
        self.m_actions = self.env.action_space.n
        # relax angle at which to fail the episode
        self.theta_threshold_radians = np.deg2rad(30)
        self.x_threshold = 4.0
        self.state_cues = False
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        #self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, **kwargs):
        """ sets and returns initial observation """
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        if self.is_ood:
            epsilon = 0.1 if self.exclude_upright else 0
            self.goal = 2 * np.pi * np.random.uniform(0 + epsilon,1 - epsilon)
            self.target_height = np.cos(self.goal)

        obs = super().reset(**kwargs)
        # overwrite original initial state - reset ranges are too small
        #self.unwrapped.state = self.np_random.uniform(low=-0.75, high=0.75, size=(4,))
        self.unwrapped.state = self.np_random.uniform(low=-0.15, high=0.15, size=(4,))
        self.steps_above_height = 0
        self.timer = 0
        self.th_initial = self.state[0]
        self.frame_initial = self.draw_cues(z_type='none', cues=None)
        return obs

    def _step(self, action):
        self.timer += 1
        action = self.to_action_space(action) # todo: ensure action mapping is correct for drqv2
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        # count as balance for small angle and low ang vel
        _, _, th, dth = self.state

        # convert angles to range [-pi,pi]
        th_target_norm = ((self.goal + np.pi) % (2 * np.pi)) - np.pi
        th_norm = ((th + np.pi) % (2 * np.pi)) - np.pi

        vel_thresh = 4 if self.is_ood else 2
        if abs(th_norm - th_target_norm) < np.deg2rad(15) and abs(dth) < vel_thresh: # angles in world (no need to shift by -pi)
            self.steps_above_height += 1
        else:
            self.steps_above_height = 0

        r = super().step(action)
        # force later termination
        return *r[0:2], r[2] and self.timer >= 40, *r[3:]

    def _step_ood(self, u):
        # count as balance for small angle and low ang vel only at initial
        th_old, dth_old = self.state
        # take step but overwrite reward for non-zero
        ret = super().step(u)

        # convert angles to range [-pi,pi]
        th_target_norm = ((self.goal + np.pi) % (2 * np.pi)) - np.pi
        th_norm = ((th_old + np.pi) % (2 * np.pi)) - np.pi

        vel_thresh = 4 if self.is_ood else 2
        if abs(th_norm - th_target_norm) < np.deg2rad(15) and abs(dth_old) < vel_thresh: # angles in world (no need to shift by -pi)
            self.steps_above_height += 1
        else:
            self.steps_above_height = 0

        # compute cost wrt random goal (ensure cost is float only!)
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        # note that norm(a - b) < norm(a) - norm(b)
        #costs = (abs(th_norm) - abs(th_target_norm)) ** 2 + dth_old**2
        costs = np.log(abs(th_norm) - abs(th_target_norm)) + dth_old**2

        # optionally transform state
        obs = ret[0]
        if ret[1] > 0:
            return obs, 1-costs, *ret[2:]
        else:
            return ret

    def to_action_space(self, action: np.ndarray):
        """ for input in [-1, 1] """
        if action.dtype == np.int32 or action.dtype == np.int64:
            return super().to_action_space(action)
        else:
            # map to push left or right {0, 1}
            action = 0.5 * (action[0:1] + 1)
            action = np.clip(action, 0, 1)
            action = np.round(action) # {0, 1}
            return super().to_action_space(action)

    def get_state(self):
        # specifically, of the pole; note x is vertical axis; cos(th)
        _, _, th, dth = self.state
        state = np.array([np.cos(th), np.sin(th), dth], dtype=np.float32)
        return state

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        #frame = self.render(mode='rgb_array')
        if frame is None:
            frame = self.get_observation() * 255 # debug: visualise agent inputs
        if self.state_cues:
            return frame

        W, H = frame.shape[0:2]
        l_to_pel = H // 4

        if z_type == 'none':
            c = (255, 255, 255)
            cues = [0]
        elif len(cues) == 1:
            c = (128, 128, 128)
        else:
            dth = int(255 * (cues[1] + 16) / 32) # ang vel ranges from [-inf,inf]
            c = (255 - dth, 0, dth)

        T = H // 64
        offs = 0#int(2 * T) #int(0.03 * W if T < 2 else 2 * T)
        h_from_top = int(cues[0] * l_to_pel)
        h_top = int(1.0 * l_to_pel + int(l_to_pel * (-1 * self.target_height + 1)))
        cv.arrowedLine(frame, (W - offs, h_top), (W - offs, h_top + h_from_top), c, thickness=T)

        if annotate:
            shortened_ztype = ' '.join([x[0:3] for x in z_type.replace('ground_truth_', '').split('-')])
            cv.putText(frame, f"{shortened_ztype}",
                        (10, H-5),
                        cv.FONT_HERSHEY_SIMPLEX,0.4,(0,128,128),T, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self):
        x, y, ang_vel = self.get_state() # pole state
        height_from_top = self.target_height - x  # range [0,2]
        torque = 0.0
        if 0.7 < x < 0.95:
            torque = 0.1 if ang_vel < 0 else -0.1 # make sure to oppose
        elif 0.5 < x < 0.7:
            torque = 0.8 if ang_vel < 0 else -0.8
        elif -0.5 < x < -0.7:
            torque = 0.3 if ang_vel > 0 else -0.3 # make sure to algin
        elif -0.7 < x < -1:
            torque = 2 if ang_vel > 0 else -2
        return self.get_state() if self.state_cues else [height_from_top, ang_vel, torque]

    def create_policy_args(self, z_type):
        z_class_size = 0
        if z_type == 'none':
            z_in_size = 0 
        elif 'state' in z_type:
            self.state_cues = True
            z_in_size = 3  # pole state: x, y, ang_vel
        elif 'height-angle' in z_type:
            z_in_size = 2 
        elif 'act' in z_type or 'height' in z_type or 'angle' in z_type or 'noise' in z_type:
            z_in_size = 1  
        else:
            raise ValueError(f'invalid z_type - {z_type}')

        return dict(z_type=z_type,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    z_in_size=z_in_size,
                    z_class_size=z_class_size)

    def slice_cues(self, z_type):
        """ returns subset of cues; only use in *_cues fns """
        if 'height-angle' in z_type:
            return (0, 2)
        elif 'height' in z_type or 'noise' in z_type:
            return (0, 1)
        elif 'angle' in z_type:
            return (1, 2)
        elif 'act' in z_type:
            return (2, 3)
        elif 'state' in z_type:
            return 0, len(self.get_state())
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ return angular velocity, target height, and total steps above target """
        return cues[-1], self.steps_above_height

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        ang_vel, steps_above_height = zip(*progress_metrics)
        #pct_above_height = steps_above_height[-1] / len(steps_above_height)
        pct_above_height = max(steps_above_height) / len(steps_above_height)
        with open(join(agent_dir, f"t-{step}_max_progress.txt"), 'w') as fp:
            fp.write(f"percent above target height ({np.cos(self.th_initial)} -> {self.target_height}): {pct_above_height:.3f}")

        pickle.dump([ang_vel], open(join(agent_dir, f"t-{step}_key_cues.pkl"), 'wb'))
        pickle.dump([self.th_initial, pct_above_height, step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))