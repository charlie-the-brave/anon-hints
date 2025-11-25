import cv2 as cv
import numpy as np
import pickle
from os.path import join
from torch import FloatTensor
from matplotlib import pyplot as plt

from env.wrappers import ControlEnv


# note: make sure to use environment pgp-dmc for mujoco support
font = cv.FONT_HERSHEY_SIMPLEX
MAX_TIMEOUT = 200
MAX_TIMEOUT_GOAL = 15


class CheetahEnv(ControlEnv):
    """ A version of pendulum-v1 that returns computes custom state cues """
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        assert 'v4' in env_name
        self.is_ood = 'v4r' in env_name
        if self.is_ood: # setup random goal in reset
            env_name = env_name.replace('v4r', 'v4')

        super().__init__(env_name, enable_normalised_actions, needs_pixels, frame_skip)
        self.goal_position = np.zeros(2)
        self.goal_distance = -1
        self.state_info = 4 * [8]
        self.v_max = 0.
        self.goal_time = 0
        self.goal_speed = 5
        self.z_distance = 0.
        self.distance_travelled = -1
        self.goal_distance = -1
        self.initial_com = None
        self.t_timeout = 0
        self.state_cues = False

    def check_goal_reached(self):
        return self.state_info[1] > self.goal_speed and self.distance_travelled >= self.goal_distance

    def reset(self, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        #kwargs['reset_noise_scale'] = 1.0
        obs = super().reset(**kwargs)
        self.state_info = 4 * [8]
        self.v_max = 0.
        self.t_timeout = 0
        self.goal_time = 0
        self.goal_speed = 5#10
        self.z_distance = 0.
        self.distance_travelled = 0.
        self.goal_distance = 50 
        self.inital_com = self.get_body_com("torso").copy()
        return obs

    def _step(self, action):
        action = self.to_action_space(action)
        if self.state_info[1] < 1.:
            self.t_timeout += 1
        #todo: consider moving done overwrite here
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        ret = super().step(action)
        self.state_info = list(ret[3].values()) # x, dx, rew, cost
        self.v_max = max(self.state_info[1], self.v_max)
        if self.check_goal_reached():
            self.goal_time += 1 

        com_torso = self.get_body_com("torso").copy()
        self.distance_travelled = np.linalg.norm(com_torso[:2] - self.inital_com[:2])
        self.z_distance = np.linalg.norm(com_torso[2] - self.inital_com[2])
        # done is always false, so overwrite based on success or timeout
        done_by_timeout = self.t_timeout > MAX_TIMEOUT or self.distance_travelled >= self.goal_distance
        return *ret[:2], done_by_timeout, *ret[3:]

    def _step_ood(self, u):
        pass

    def get_state(self):
        # truncate since env wrappers may append to unwrapped info
        return self.state_info[:4]

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        if frame is None:
            frame = self.get_observation() * 255
        if self.state_cues:
            return frame

        H, W = frame.shape[0:2]
        W, H = W - W // 8, H // 8 # box corner offset

        speed_norm = self.safe_ratio(cues[1], self.goal_speed)
        pct_distance = self.safe_ratio(cues[4], self.goal_distance) if len(cues) > 4 else 0
        rew, cost = cues[1:3]

        if z_type == 'none':
            c_velocity = (0,0,0)
        else:
            c_velocity = (255,0,0)

        # visualise 2d speed
        L = 3 * H
        T = frame.shape[1] // 64
        scale = 0.2 if T < 3 else 0.8
        offs = int(0.22*W)
        speed = int(H * speed_norm)
        cv.line(frame, (W - L, H), (W, H), (200,200,200,0.1), thickness=2 * T)
        cv.line(frame, (W - L // 2, H), (W - L // 2 + speed, H), c_velocity, thickness=T)
        cv.putText(frame, f"v:{abs(speed):.0f}",(W - L - offs, H), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        # visualise rewards vs cost
        rbar = 2 + int(L * self.safe_ratio(rew, 1000))
        cbar = 2 + int(L * self.safe_ratio(cost, 1000))
        cv.line(frame, (W - L, H + 4*T), (W, H + 4*T), (200,200,200,0.1), thickness=3 * T)
        cv.line(frame, (W - L, H + 4*T), (W - L + cbar, H + 4*T), (200,0,0), thickness=2 * T)
        cv.line(frame, (W - L, H + 4*T), (W - L + rbar, H + 4*T), (0,200,0), thickness=T)
        cv.putText(frame, f"r:",(W - L - offs, H + 4*T), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        # visualise progress to goal distance
        bar = int(0.8 * L * min(1.2, pct_distance)) # full is 120+%
        c_goal_distance = (0,255,0) if self.check_goal_reached() else (255,0,0)
        cv.line(frame, (W - L, H + 8*T), (W, H + 8*T), (200,200,200,0.1), thickness=2 * T)
        cv.line(frame, (W - L, H + 8*T), (W - L + bar, H + 8*T), c_goal_distance, thickness=T)
        cv.putText(frame, f"d:{self.distance_travelled:.0f}",(W - L - offs, H + 8*T), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self):
        # state-based torso height
        # action-based limit joint angle/vel of external limbs
        x, dx, rew, cost = self.get_state()
        # todo: think of direct ways to provide action constraints
        #       conditioning will not be enough
        return self.get_state() if self.state_cues else [self.v_max, dx, rew, cost, self.distance_travelled, self.z_distance]

    def create_policy_args(self, z_type):
        z_class_size = 0
        if z_type == 'none':
            z_in_size = 0 
        elif 'height' in z_type or 'dist' in z_type or 'cost' in z_type or 'max' in z_type or 'speed' in z_type or 'rew' in z_type:
            z_in_size = 1
        elif 'state' in z_type:
            self.state_cues = True
            z_in_size = len(self.state_info)
        else:
            raise ValueError(f'invalid z_type - {z_type}')

        return dict(z_type=z_type,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    z_in_size=z_in_size,
                    z_class_size=z_class_size)

    def slice_cues(self, z_type):
        """ returns subset of cues; only use in *_cues fns """
        if 'max' in z_type:
            return (0, 1)
        elif 'speed' in z_type:
            return (1, 2)
        elif 'rew' in z_type:
            return (2, 3)
        elif 'cost' in z_type:
            return (3, 4)
        elif 'dist' in z_type:
            return (4, 5)
        elif 'height' in z_type:
            return (5, 6)
        elif 'state' in z_type:
            return (0, len(self.state_info))
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ return angular velocity, target height, and total steps above target """
        return cues[0:2]

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        vmax, vel = zip(*progress_metrics)
        avg_vel = np.mean(vel)
        with open(join(agent_dir, f"t-{step}_max_progress.txt"), 'w') as fp:
            fp.write(f"max velocity ({np.max(vmax)} -> {self.goal_speed}): {avg_vel:.3f}")

        pct_distance = self.safe_ratio(self.distance_travelled, self.goal_distance)
        pickle.dump([np.max(vmax), avg_vel], open(join(agent_dir, f"t-{step}_cues.pkl"), 'wb'))
        pickle.dump([self.goal_distance, avg_vel, pct_distance, step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))
