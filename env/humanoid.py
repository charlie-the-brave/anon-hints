import cv2 as cv
import numpy as np
import pickle
from os.path import join
from torch import FloatTensor
from matplotlib import pyplot as plt

from env.wrappers import ControlEnv

# note: make sure to use environment pgp-dmc for mujoco support
font = cv.FONT_HERSHEY_SIMPLEX

# TODO: consider merging with ant and cheetah into generic mujoco env
#       need to expose center of mass

class HumanoidEnv(ControlEnv):
    """ A version of Humanoid-v4 that returns computes custom state cues """
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
        return self.distance_travelled > self.goal_distance

    def reset(self, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        #kwargs['reset_noise_scale'] = 1.0
        obs = super().reset(**kwargs)
        self.state = np.zeros(5)
        self.distance_travelled = 0.
        self.goal_distance = 50 * np.random.uniform(0, 1)
        self.frame_initial = self.draw_cues(z_type='none', cues=self.compute_cues())
        return obs

    def _step(self, action):
        action = self.to_action_space(action)
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        ret = super().step(action)
        info = ret[3]
        self.state = np.array([info["x_position"], info["y_position"], self.data.qpos[2],#.flat.copy()[2],
                      info["x_velocity"], info["x_velocity"]])
        self.distance_travelled = ret[3]["distance_from_origin"]
        return ret

    def _step_ood(self, u):
        pass

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        if frame is None:
            frame = self.get_observation() * 255
        if self.state_cues:
            return frame

        W, H = frame.shape[0:2]
        W, H = W - W // 8, H // 8 # offset things (note that vertical axis is flipped)

        z_dist_norm = cues[2]
        speed_norm = self.safe_ratio(cues[1], 10)
        pct_distance = self.safe_ratio(cues[0], self.goal_distance)

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
        cv.putText(frame, f"d:{cues[0]:.0f}",(W - L - offs, H + 8*T), font, scale, (0,0,0), T // 2, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self):
        # state-based torso height
        # action-based limit joint angle/vel of external limbs
        x, y, z, dx, dy = self.state
        # todo: think of direct ways to provide action constraints
        #       conditioning will not be enough
        speed = np.sqrt(dx**2 + dy**2)
        z_dist_norm = self.safe_ratio(z, self.z_goal)
        return self.get_state() if self.state_cues else [self.distance_travelled, speed, z_dist_norm]

    def get_state(self):
        return np.concatenate(
            (
                self.data.qpos.flat, # joint positions
                self.data.qvel.flat, # joint velocities
                self.data.cinert.flat, # COM inertia
                self.data.cvel.flat, # COM velocities
                self.data.qfrc_actuator.flat, # actuator forces
                self.data.cfrc_ext.flat, # contact forces
            )
        )

    def create_policy_args(self, z_type):
        z_class_size = 0
        if z_type == 'none':
            z_in_size = 0 
        elif 'act' in z_type or 'height' in z_type or 'speed' in z_type or 'distance' in z_type:
            z_in_size = 1
        elif 'state' in z_type:
            self.state_cues = True
            z_in_size = len(list(self.get_state()))
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
        elif 'state' in z_type: # TODO: fix
            return (0, len(self.get_state()))
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ return angular velocity, target height, and total steps above target """
        return cues[0:2]

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        distance, vel = zip(*progress_metrics)
        avg_vel = np.mean(vel)
        pct_distance = self.safe_ratio(self.distance_travelled, self.goal_distance)
        with open(join(agent_dir, f"t-{step}_max_progress.txt"), 'w') as fp:
            fp.write(f"distance travelled ({np.max(distance)} -> {self.goal_distance}): {avg_vel:.3f}")

        pickle.dump([np.max(distance), np.mean(vel)], open(join(agent_dir, f"t-{step}_key_cues.pkl"), 'wb'))
        pickle.dump([self.goal_distance, pct_distance, step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))
