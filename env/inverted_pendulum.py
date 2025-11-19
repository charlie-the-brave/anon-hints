import cv2 as cv
import numpy as np
import pickle
from os.path import join
from env.wrappers import ControlEnv
from torch import FloatTensor, randn_like


font = cv.FONT_HERSHEY_SIMPLEX
# note: make sure to use environment pgp-dmc for mujoco support

class CartPendulumEnv(ControlEnv):
    """ A version of pendulum-v1 that returns computes custom state cues """
    def __init__(self, env_name, enable_normalised_actions, needs_pixels, frame_skip=0):
        assert 'v4' in env_name
        self.is_ood = 'v4r' in env_name
        if self.is_ood: # setup random goal in reset
            env_name = env_name.replace('v4r', 'v4')
        self.goal = 0. # vertical alignment
        self.state_cues = False

        super().__init__(env_name, enable_normalised_actions, needs_pixels, frame_skip)
        self.target_height = 2 * np.cos(self.goal)
        self.steps_above_height = 0
        self.goal_steps_steps_above_height = 50

    def check_goal_reached(self, context=None):
        # count as balance for small angle and low ang vel
        if context is None: context = self.compute_cues()
        height_from_top, dth0, dth1 = context[:3]
        return height_from_top < 0.2 and abs(dth0 + dth1) < 2

    def reset(self, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        obs = super().reset(**kwargs)
        self.steps_above_height = 0
        self.th_initial = self.data.qpos[2]
        self.frame_initial = self.draw_cues(z_type='none', cues=self.compute_cues())
        return obs

    def _step(self, action):
        action = self.to_action_space(action)
        return self._step_ood(action) if self.is_ood else self._step_original(action)

    def _step_original(self, action):
        if self.check_goal_reached(): # angles in world (no need to shift by -pi)
            self.steps_above_height += 1
        else:
            self.steps_above_height = 0

        ret = super().step(action)

        return *ret[:2], ret[2] or self.steps_above_height > 100, *ret[3:]


    def _step_ood(self, action):
        ret = super().step(action)
        n = len(self.data.qpos) - 1 # ignoring cart at 0th element
        ood_obs = np.copy(ret[0])
        # apply mirror offset to angles
        ood_obs[1:n+1] = np.sin(self.data.qpos[1:] + np.deg2rad(180))
        ood_obs[n+1:2*n+1] = np.cos(self.data.qpos[1:] + np.deg2rad(180))
        return ood_obs, *ret[1:]

    def get_state(self):
        # zero degrees is aligned w y-axis: x=sin(th)
        x, _, y = self.data.site_xpos[0] # cart pos
        th0, th1 = self.data.qpos[1:3] # vertical angles in frame of cart then first pole
        dth0, dth1 = self.data.qvel[1:3]
        th1 += th0 # relative to the base frame
        state = np.array([np.sin(th0), np.cos(th0), dth0, np.sin(th1), np.cos(th1), dth1], dtype=np.float32)
        return state

    def draw_cues(self, z_type, cues, frame=None, annotate=False):
        """ visualise state-based cues """
        #frame = self.render(mode='rgb_array')
        if frame is None:
            frame = self.get_observation() * 255

        W, H = frame.shape[0:2]
        l_to_pel = H // 4


        if z_type == 'none':
            c_balanced = (0, 128, 0) if self.check_goal_reached(cues) else (128, 0, 0)
        else:
            c_balanced = (0, 255, 0) if self.check_goal_reached(cues) else (255, 0, 0)

        T = H // 64
        offs = int(0.03 * W)
        h_from_top = int(cues[0] * l_to_pel)
        h_top = H - l_to_pel * (int(self.target_height) + 1)
        cv.arrowedLine(frame, (W - offs, h_top), (W - offs, h_top + h_from_top), c_balanced, thickness=T)

        L, V_MAX = W // 6, 28
        _w = W - offs - W // 10
        scale = 0.8
        bar = int(L // 2 * self.safe_ratio(cues[2], V_MAX))
        cv.line(frame, (_w - L, H//16), (_w, H//16), (200,200,200,0.1), thickness=2*T)
        cv.line(frame, (_w - L // 2, H//16), (_w - L // 2 + bar, H//16), c_balanced, thickness=T)
        cv.putText(frame, f"dth",(W - int(0.1 * W), H//16), font, scale, (0,0,0), T//2, lineType=cv.LINE_AA)

        if annotate:
            shortened_ztype = ' '.join([x[0:3] for x in z_type.replace('ground_truth_', '').split('-')])
            cv.putText(frame, f"{shortened_ztype}",
                        (10, H-5),
                        cv.FONT_HERSHEY_SIMPLEX,0.4,(0,128,128),1, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self):
        # reference upper pendulum
        x0, y0, dth0, x, y, dth1 = self.get_state()[:6]
        dth_joint = dth0 + dth1
        height_from_top = self.target_height - y0 - y  # range [0,2]
        torque = 0.0
        if 0.7 < x < 0.95:
            torque = 0.1 if dth0 < 0 else -0.1 # make sure to oppose
        elif 0.5 < x < 0.7:
            torque = 0.8 if dth0 < 0 else -0.8
        elif -0.5 < x < -0.7:
            torque = 0.3 if dth0 > 0 else -0.3 # make sure to algin
        elif -0.7 < x < -1:
            torque = 2 if dth0 > 0 else -2
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
        elif 'act' in z_type or 'angle-first' in z_type or 'angle-second' in z_type or 'height' in z_type:
            z_in_size = 1 
        elif 'angle' in z_type:
            z_in_size = 2  # angular velocity at each joint (second relative to first)
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
        if 'height' in z_type:
            return (0, 1)
        elif 'angle-first' in z_type:
            return (1, 2)
        elif 'angle-second' in z_type:
            return (2, 3)
        elif 'angle-joint' in z_type:
            return (3, 4)  # second relative to first
        elif 'angle' in z_type:
            return (2, 4)  # first & second relative to first
        elif 'act' in z_type:
            return (4, 5)
        elif 'state-partial' in z_type:
            return (2, len(self.get_state()))
        elif 'state' in z_type:
            return (0, len(self.get_state()))
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ return angular velocity, target height, and total steps above target """
        return cues[-1], self.steps_above_height

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        ang_vel, steps_above_height = zip(*progress_metrics)
        max_above_height = max(steps_above_height) / self.goal_steps_steps_above_height
        with open(join(agent_dir, f"t-{step}_max_progress.txt"), 'w') as fp:
            fp.write(f"percent above target height ({np.cos(self.th_initial)} -> {self.target_height}): {max_above_height:.3f}")

        pickle.dump([ang_vel], open(join(agent_dir, f"t-{step}_cues.pkl"), 'wb'))
        pickle.dump([self.th_initial, max_above_height, step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))
