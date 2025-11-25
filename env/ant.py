import cv2 as cv
import numpy as np
import pickle
from os.path import join
from torch import FloatTensor
from matplotlib import pyplot as plt

from env.wrappers import ControlEnv

# note: make sure to use environment pgp-dmc for mujoco support
font = cv.FONT_HERSHEY_SIMPLEX

class AntEnv(ControlEnv):
    """ A version of pendulum-v1 that returns computes custom state cues """
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        assert 'v4' in env_name
        self.is_ood = 'v4r' in env_name
        if self.is_ood: # setup random goal in reset
            env_name = env_name.replace('v4r', 'v4')

        super().__init__(env_name, enable_normalised_actions, needs_pixels, frame_skip)
        self.goal_position = np.zeros(2)
        self.z_goal = 0.72 # just a little under stated height af 75cm
        self.goal_distance = -1
        self.state_cues = False

    def check_goal_reached(self):
        return self.state_info['distance_from_origin'] > self.goal_distance

    def reset(self, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        #kwargs['reset_noise_scale'] = 1.0
        obs = super().reset(**kwargs)
        self.pos_initial = self.get_body_com("torso")[:2].copy()
        xy = self.get_body_com("torso")[:2].copy()
        self.state_info = dict(x_position=xy[0], y_position=xy[1], x_velocity=0, y_velocity=0, distance_from_origin=0)
        self.goal_distance = 50#50 * np.random.uniform(0, 1)
        return obs

    def _step(self, action):
        action = self.to_action_space(action)
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        #x, y, _, _, _, self.distance_travelled = self.get_state()
        #distance_travelled = np.linalg.norm(np.array([x,y]) - self.pos_initial, ord=2)
        ret = super().step(action)
        self.state_info = ret[3]
        return *ret[:2], ret[2] or self.state_info['distance_from_origin'] > self.goal_distance, *ret[3:]

    def _step_ood(self, u):
        pass

    def get_state(self):
        state = self.state_info
        x, y, z = state['x_position'], state['y_position'], self.state_vector()[2]
        dx, dy = state['x_velocity'], state['y_velocity']
        # torso angular velocities?
        #forces = self.contact_forces()
        return x, y, z, dx, dy, state['distance_from_origin']

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        if frame is None:
            frame = self.get_observation() * 255
        if self.state_cues:
            return frame

        H, W = frame.shape[0:2]
        W, H = W - W // 8, H // 8 # box corner offset

        z_dist_norm = cues[2]
        speed_norm = self.safe_ratio(cues[1], 10)
        pct_distance = self.safe_ratio(cues[0], self.goal_distance)
        distance_travelled = cues[0]

        if z_type == 'none':
            c_velocity = (0,0,0)
            c_zdistance = (0,0,0)
        else:
            c_velocity = (255,0,0)
            c_zdistance = (0,0,128)

        L = 3 * H
        T = frame.shape[1] // 64
        scale = 0.2 if T < 3 else 0.8
        offs = int(0.20*W)
        speed = int(H * speed_norm)
        cv.line(frame, (W - L, H), (W, H), (200,200,200,0.1), thickness=2 * T)
        cv.line(frame, (W - L // 2, H), (W - L // 2 + speed, H), c_velocity, thickness=T)
        cv.putText(frame, f"v:{speed:.0f}",(W - L - offs, H), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        z_bar = int(H * (z_dist_norm - 1))
        cv.line(frame, (W - L, H + 4*T), (W, H + 4*T), (200,200,200,0.1), thickness=2 * T)
        cv.line(frame, (W - L // 2, H + 4*T), (W - L // 2 + z_bar, H + 4*T), c_zdistance, thickness=T)
        cv.putText(frame, f"z:{z_dist_norm:.1f}",(W - L - offs, H + 4*T), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        # visualise progress to goal distance
        bar = int(0.8 * L * min(1.2, pct_distance)) # full is 120+%
        c_goal_distance = (0,255,0) if self.check_goal_reached() else (255,0,0) # toggle colour after 100%
        cv.line(frame, (W - L, H + 8*T), (W, H + 8*T), (200,200,200,0.1), thickness=2 * T)
        cv.line(frame, (W - L, H + 8*T), (W - L + bar, H + 8*T), c_goal_distance, thickness=T)
        cv.putText(frame, f"d:{distance_travelled:.0f}",(W - L - offs, H + 8*T), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self):
        # state-based torso height
        # action-based limit joint angle/vel of external limbs
        x, y, z, dx, dy, distance_travelled = self.get_state()
        # todo: think of direct ways to provide action constraints
        #       conditioning will not be enough
        speed = np.sqrt(dx**2 + dy**2)
        z_dist_norm = self.safe_ratio(z, self.z_goal)
        return self.state_vector() if self.state_cues else [distance_travelled, speed, z_dist_norm]

    def compute_actions(self):
        forward = np.array([
            np.arange(self.action_space.shape[0]) % 2,
            np.arange(self.action_space.shape[0]) % 2,
            np.arange(self.action_space.shape[0]) % 2,
            (np.arange(self.action_space.shape[0])+1) % 2,
            (np.arange(self.action_space.shape[0])+1) % 2,
            (np.arange(self.action_space.shape[0])+1) % 2,
        ])
        return np.stack([forward, -1 * forward], axis=0)

    def create_policy_args(self, z_type):
        z_class_size = 0
        if z_type == 'none':
            z_in_size = 0 
        elif 'height' in z_type or 'speed' in z_type or 'distance' in z_type:
            z_in_size = 1
        elif 'state' in z_type:
            self.state_cues = True
            z_in_size = len(list(self.state_vector()))
        elif 'act' in z_type:
            z_in_size = len(list(self.compute_actions().flatten()))
        else:
            raise ValueError(f'invalid z_type - {z_type}')

        return dict(z_type=z_type,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    z_in_size=z_in_size,
                    z_class_size=z_class_size)

    def slice_cues(self, z_type):
        """ returns indices corresponding to target cues; only use in *_cues fns """
        if 'distance' in z_type:
            return (0, 1)
        elif 'speed' in z_type:
            return (1, 2)
        elif 'height' in z_type:
            return (2, 3)
        elif 'state' in z_type:
            return 0, len(self.state_vector())
        elif 'act' in z_type:
            return (0, len(self.compute_actions().flatten()))
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ return angular velocity, target height, and total steps above target """
        return cues[0:2]

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        distance, vel = zip(*progress_metrics)
        avg_vel = np.mean(vel)
        pct_distance = self.safe_ratio(np.max(distance), self.goal_distance)
        with open(join(agent_dir, f"t-{step}_max_progress.txt"), 'w') as fp:
            fp.write(f"distance travelled ({np.max(distance)} -> {self.goal_distance}): {avg_vel:.3f}")

        pickle.dump([np.max(distance), np.mean(vel)], open(join(agent_dir, f"t-{step}_cues.pkl"), 'wb'))
        pickle.dump([self.goal_distance, pct_distance, step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))
