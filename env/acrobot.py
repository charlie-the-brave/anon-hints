import cv2 as cv
import numpy as np
import pickle
from os.path import join
from torch import FloatTensor, randn_like
from gym import spaces

from env.wrappers import ControlEnv

# note: make sure to use environment pgp-dmc for mujoco support
font = cv.FONT_HERSHEY_SIMPLEX

class AcrobotEnv(ControlEnv):
    """ A version of pendulum-v1 that returns computes custom state cues """
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        self.is_ood = 'v1r' in env_name or 'v2r' in env_name
        # select goal angle (radians, counter-clockwise)
        self.goal = np.deg2rad(180)
        if self.is_ood: # setup random goal in reset
            env_name = env_name.replace('v1r', 'v1').replace('v2r', 'v1')

        super().__init__(env_name, enable_normalised_actions, needs_pixels, frame_skip)
        self.target_height = np.cos(self.goal) # in the base frame
        self.timer = 0
        self.state_cues = False

    def reset(self, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        obs = super().reset(**kwargs)
        self.timer = 0
        self.initial_height = self.get_state()[4] # upper link pos is relative to lower link
        return obs

    def _step(self, action):
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        """ map from [-1,1] to {0, 1, 2} """
        self.timer += 1
        action = 0.5 * (action[0:1] + 1)
        action = np.round(2 * np.clip(action, 0, 1)) # {0, 1, 2}
        action = super().to_action_space(action)
        return super().step(action)

    def _step_ood(self, action):
        self.timer += 1
        if self.is_ood:
            # apply mirror offsets to state
            self.state[0] += np.deg2rad(180)
            self.state[1] += np.deg2rad(180)
        return super().step(int(action[0] if isinstance(action, np.ndarray) and len(action.shape) > 0 else action))

    def get_state(self):
        # note y is vertical axis; cos(th)
        th0, th1, dth0, dth1 = self.state[:4]
        th1 += th0 # relative to first joint!
        # in original env, the state reps [y0, x0, y1, x1, dth0, dth1]
        # we use ordering [x0, y0, dth0, x1, y1, dth1]
        state = np.array([np.sin(th0), np.cos(th0), dth0, np.sin(th1), np.cos(th1), dth1], dtype=np.float32)
        return state

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        """ visualise state-based cues """
        #frame = self.render(mode='rgb_array')
        if frame is None:
            frame = self.get_observation() * 255 # debug: visualise agent inputs
        if self.state_cues:
            return frame

        W, H = frame.shape[0:2]
        l_to_pel = H // 4
        h_center = H // 2

        if z_type == 'none':
            c = (0, 0, 0)
        else:
            c = (255, 0, 0)

        # goal is reached at negative height
        T = H // 64
        offs = int(0.05 * W)
        h_from_top = int(cues[0] * l_to_pel + 2)
        h_top = h_center - int(l_to_pel * self.target_height)
        cv.line(frame, (W - offs, H - h_top), (W - offs, H - h_top - h_from_top), c, thickness=T)
        cv.circle(frame, (W - offs, H - h_top - h_from_top), color=c, radius=1, thickness=2*T)

        L, V_MAX = W // 6, 28
        _w = W - offs - W // 10
        scale = 0.8
        bar = int(L // 2 * self.safe_ratio(cues[2], V_MAX))
        cv.line(frame, (_w - L, H//16), (_w, H//16), (200,200,200,0.1), thickness=2*T)
        cv.line(frame, (_w - L // 2, H//16), (_w - L // 2 + bar, H//16), (255,0,0,0.1), thickness=T)
        cv.putText(frame, f"dth",(W - int(0.1 * W), H//16), font, scale, (0,0,0), T//2, lineType=cv.LINE_AA)

        if annotate:
            shortened_ztype = ' '.join([x[0:3] for x in z_type.replace('ground_truth_', '').split('-')])
            cv.putText(frame, f"{shortened_ztype}",
                        (10, H-5),
                        font, scale,(0,128,128),T//2, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self):
        x0, y0, dth0, x1, y1, dth1 = self.get_state()[:6]
        height_from_top = self.target_height - y0 - y1  # range [0,2]
        dth_joint = dth0 + dth1 # linearly combine since kinematic chain
        torque = 0.0
        return self.get_state() if self.state_cues else [height_from_top, dth0, dth1, dth_joint, torque]

    def create_policy_args(self, z_type):
        # angle - short for angular velocity
        z_class_size = 0
        if z_type == 'none':
            z_in_size = 0 
        elif 'state-partial' in z_type:
            z_in_size = self.get_state()[2:].shape[0]
        elif 'angle-joint' in z_type:
            z_in_size = 1  # angle from base to end of second link
        elif 'act' in z_type or 'angle-first' in z_type or 'angle-second' in z_type:
            z_in_size = 1  
        elif 'angle' in z_type:
            z_in_size = 2  # angular velocity at each joint
        elif 'state' in z_type:
            self.state_cues = True
            z_in_size = self.get_state().shape[0]
        else:
            raise ValueError(f'invalid z_type - {z_type}')

        return dict(z_type=z_type,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    z_in_size=z_in_size,
                    z_class_size=z_class_size)

    def slice_cues(self, z_type):
        """ returns indices corresponding to target cues; only use in *_cues fns """
        # angle - short for angular velocity

        if 'state-partial' in z_type:
            return (2, len(self.get_state()))
        elif 'height' in z_type:
            return (0, 1)
        elif 'angle-first' in z_type:
            return (1, 2)
        elif 'angle-second' in z_type:
            return (2, 3)
        elif 'angle-joint' in z_type:
            return (3, 4)
        elif 'angle' in z_type:
            return (2, 4)
        elif 'act' in z_type:
            return (4, 5)
        elif 'state' in z_type:
            return (0, len(self.get_state()))
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ return angular velocity, target height, and total steps above target """
        return cues[1], self.timer

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        ang_vel, time_steps = zip(*progress_metrics)
        with open(join(agent_dir, f"t-{step}_max_progress.txt"), 'w') as fp:
            fp.write(f"time from ({self.initial_height} -> {self.target_height}): {time_steps[-1]:.3f}")

        pickle.dump([ang_vel], open(join(agent_dir, f"t-{step}_key_cues.pkl"), 'wb'))
        pickle.dump([self.initial_height, time_steps[-1], step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))
