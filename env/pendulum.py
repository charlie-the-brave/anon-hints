import cv2 as cv
import numpy as np
from torch import FloatTensor
import pickle
from os.path import join
from gym import spaces
from matplotlib import pyplot as plt

from env.wrappers import ControlEnv


font = cv.FONT_HERSHEY_SIMPLEX


class PendulumEnv(ControlEnv):
    """ A version of pendulum-v1 that returns computes custom state cues """
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        self.is_ood = 'v1r' in env_name or 'v2r' in env_name
        self.exclude_upright = 'v2r' in env_name
        # select goal angle (radians, counter-clockwise)
        self.goal = np.deg2rad(0)
        if self.is_ood: # setup random goal in reset
            env_name = env_name.replace('v1r', 'v1').replace('v2r', 'v1')

        super().__init__(env_name, enable_normalised_actions, needs_pixels, frame_skip)
        self.target_height = np.cos(self.goal)
        self.steps_above_height = 0
        self.state_cues = False

    def reset(self, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        if self.is_ood:
            self.goal = 2 * np.pi * np.random.uniform(-0.25, 0.25)
            self.target_height = np.cos(self.goal)
        obs = super().reset(**kwargs)
        self.steps_above_height = 0
        self.th_initial = self.state[0]
        #self.frame_initial = self.draw_cues(z_type='none', cues=None)
        return obs

    def _step(self, action):
        # this is a truncated env (see timelimit.py)
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        # count as balance for small angle and low ang vel
        th, dth = self.state

        # convert angles to range [-pi,pi]
        th_target_norm = ((self.goal + np.pi) % (2 * np.pi)) - np.pi
        th_norm = ((th + np.pi) % (2 * np.pi)) - np.pi

        vel_thresh = 4 if self.is_ood else 2
        if abs(th_norm - th_target_norm) < np.deg2rad(15) and abs(dth) < vel_thresh: # angles in world (no need to shift by -pi)
            self.steps_above_height += 1
        else:
            self.steps_above_height = 0

        return super().step(action)

    def _step_ood(self, u):
        # count as balance for small angle and low ang vel only at initial
        th_old, dth_old = self.state
        # score: successful balance - swing up and hold
        #        if swing up, hold, and drop then don't count
        # take step but overwrite costs
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
        # use log to penalise distance more aggressively
        # reduce penalty for velocity to avoid trivial solns at 0,+/-pi
        costs = np.log(abs(abs(th_norm) - abs(th_target_norm))) #+ 0.05*dth_old**2

        # optionally transform state
        obs = ret[0]
        return obs, -costs, *ret[2:]

    def to_action_space(self, action: np.ndarray):
        """ input in [-1, 1] """
        # map to torque range, [-2, 2]
        action *= 2
        return super().to_action_space(action)

    def get_state(self):
        # note x is vertical axis; cos(th)
        th, dth = self.state
        state = np.array([np.cos(th), np.sin(th), dth], dtype=np.float32)
        return state

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        """ visualise state-based cues """
        if frame is None:
            frame = self.get_observation() * 255 # debug: visualise agent inputs
        if self.state_cues:
            return frame

        H, W = frame.shape[0:2]
        l_to_pel = H // 4
        L, V_MAX = W // 6, 8 

        if z_type == 'none':
            c = (255, 255, 255)
            cues = [0, 0]
        elif len(cues) == 1:
            c = (128, 128, 128)
        else:
            dth = int(255 * (cues[1] + V_MAX) / 2 / V_MAX) # ang vel from [-8,8] to [0,255]
            c = (255 - dth, 0, dth) if cues[1] <= V_MAX else (0, 0, 0)

        T = H // 64
        offs = int(0.07 * W)
        h_from_top = int(cues[0] * l_to_pel)
        h_top = l_to_pel + int(l_to_pel * (-1 * self.target_height + 1))
        cv.arrowedLine(frame, (W - offs, h_top), (W - offs, h_top + h_from_top), (128, 128, 128), thickness=T)

        _w = W - offs - W // 10
        scale = 0.8
        bar = int(L // 2 * self.safe_ratio(cues[1], V_MAX))
        cv.line(frame, (_w - L, H//16), (_w, H//16), (200,200,200,0.1), thickness=2*T)
        cv.line(frame, (_w - L // 2, H//16), (_w - L // 2 + bar, H//16), c, thickness=T)
        cv.putText(frame, f"dth",(W - int(0.1 * W), H//16), font, scale, (0,0,0), T//2)

        if self.is_ood:
            goal_in_im = self.goal - np.pi # transform so that +x-axis is up and +y-axis is left
            x_targ, y_targ = int(l_to_pel * np.cos(goal_in_im)), int(l_to_pel * np.sin(goal_in_im))
            # ghost pend
            cv.line(frame, (W//2, H//2), (y_targ + W//2, x_targ + H//2), (200,200,200,0.1), thickness=2*T)
            # x marks the spot
            #cv.line(frame, (y_targ + W//2 - 5, x_targ + H//2 - 5), (y_targ + W//2 + 5, x_targ + H//2 + 5), (128,128,128,0.3), thickness=2)
            #cv.line(frame, (y_targ + W//2 - 5, x_targ + H//2 + 5), (y_targ + W//2 + 5, x_targ + H//2 - 5), (128,128,128,0.3), thickness=2)

        if annotate:
            shortened_ztype = ' '.join([x[0:3] for x in z_type.replace('ground_truth_', '').split('-')])
            cv.putText(frame, f"{shortened_ztype}",
                        (10, H-5),
                        cv.FONT_HERSHEY_SIMPLEX,0.4,(0,128,128),1)

        return frame

    def compute_cues(self):
        x, y, ang_vel = self.get_state()
        height_from_top = self.target_height - x  # range [0,2]
        torque = 0
        if height_from_top > (1 + np.sqrt(2))/2:
            torque = 1
        elif 0.5 < x < 0.7:
            torque = 0.8 if ang_vel < 0 else -0.8
        elif -0.5 < x < -0.7:
            torque = 0.3 if ang_vel > 0 else -0.3 # make sure to algin
        elif -0.7 < x < -1:
            torque = 2 if ang_vel > 0 else -2
        return self.get_state() if self.state_cues else [height_from_top, ang_vel, torque]

    def create_policy_args(self, z_type):
        # angle - short for angular velocity
        z_class_size = 0
        if z_type == 'none':
            z_in_size = 0 
        elif 'test' in z_type:
            z_in_size = len(self.compute_cues())
        elif 'state-partial' in z_type:
            z_in_size = len(self.get_state()[:2])  # x, y
        elif 'state' in z_type:
            z_in_size = len(self.get_state())  # x, y, ang_vel
        elif 'height-angle' in z_type:
            z_in_size = 2
        elif 'act' in z_type or 'height' in z_type or 'angle' in z_type or 'noise' in z_type:
            z_in_size = 1  # height or angle
        else:
            raise ValueError(f'invalid z_type - {z_type}')

        if 'state' in z_type:
            self.state_cues = True

        return dict(z_type=z_type,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    z_in_size=z_in_size,
                    z_class_size=z_class_size)

    def slice_cues(self, z_type):
        """ returns indices corresponding to target cues; only use in *_cues fns """
        # angle - short for angular velocity
        if 'test' in z_type:
            return (0, len(self.compute_cues()))
        elif 'height-angle' in z_type:
            return (0,2)
        elif 'height' in z_type or 'noise' in z_type:
            return (0,1)
        elif 'angle' in z_type:
            return (1,2)
        elif 'act' in z_type:
            return (2,3)
        elif 'state-partial' in z_type:
            return (0, len(self.get_state()[:2]))
        elif 'state' in z_type:
            return (0, len(self.get_state()))
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

        sorting_condition = self.goal if self.is_ood else self.th_initial
        pickle.dump([ang_vel], open(join(agent_dir, f"t-{step}_key_cues.pkl"), 'wb'))
        pickle.dump([sorting_condition, pct_above_height, step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))