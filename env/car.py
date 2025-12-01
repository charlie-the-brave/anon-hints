import cv2 as cv
import numpy as np
import pickle
from os.path import join
from gym import spaces
from matplotlib import pyplot as plt

from env.wrappers import ControlEnv
from env.vis_track import vis_track


class CarEnv(ControlEnv):
    def __init__(self, env_name, enable_normalised_actions, frame_skip=0):
        # env already has pixel states
        super().__init__(env_name, enable_normalised_actions, needs_pixels=False, frame_skip=frame_skip)
        self.max_timesteps = 5000
        self.R = 2
        self.ZOOM = 2.7
        self.SCALE = 3.6
        self.track_iloc = 0
        self.max_on_track = 0
        self.lookahead = 0
        self._update_track_iloc() # keep
        self.track_polys = [p for p in self.road_poly if p[1][0] != 255]
        self.turn_types = ['sharp', 'right angle', 'curve', 'straight']
        #self.RESIZE_SHAPE = (64,64,3)
        self.RESIZE_SHAPE = (96,96,3)
        self.observation_space = spaces.Box(0, 1.0, self.RESIZE_SHAPE, np.float32)

    def _rot_mat(self, th):
        """ generates 2d rotation mat from radian angle """
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def _transform_to_pixels(self, xy, rot, scale, shift, image_h):
        # rotate, scale, and shift point into ~car coordinate frame
        p = scale * (xy @ rot) + shift
        # vertical flip (y=0) to get origin at bottom left
        p[:, 1] = image_h * np.ones_like(p[:, 1]) - p[:, 1]
        return p

    def _resize(self, obs):
        obs = obs.astype('float') / 255.
        if self.RESIZE_SHAPE[:2] == obs.shape[:2]:
            return obs
        #return cv.resize(obs, (96,96), obs, interpolation=cv.INTER_AREA)
        return cv.resize(obs, self.RESIZE_SHAPE[:2], obs, interpolation=cv.INTER_AREA)

    def _update_track_iloc(self):
        """ updates track tile index based on small window of tiles around car """
        L = len(self.track) - 1
        R = L // 60 # use wide window in case car moved a lot
        xy = np.array(self.car.hull.position)
        track_pos_xy = [self.track[s % L][2:4] for s in range(self.track_iloc - R, self.track_iloc + R + 1)]
        distances = (np.array(track_pos_xy) - xy)**2
        # index closest tile in window [iloc - R, iloc + R]
        self.track_iloc = (np.argmin(distances.sum(axis=1)) + self.track_iloc - R + 1) % L

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        self.track_iloc = 0
        self.max_on_track = 0
        self._update_track_iloc()
        # skip curb polys and update track position
        self.track_polys = [p for p in self.road_poly if p[1][0] != 255]
        return self._resize(ret)

    def _step(self, action):
        """ overrides wrappera to return downsized image """
        self.t += 1
        action = self.to_action_space(action)
        ret = super().step(action)
        return self._resize(ret[0]), *ret[1:]

    def draw_cues(self, z_type, cues, frame=None, annotate=True):
        """ illustrates cues on image frame, purely for inspection, not policy input! """
        if frame is None:
            frame = self.get_observation() * 255

        if z_type != 'none':
            # scale and shift points using world-in-car transforms
            H = frame.shape[0]
            zoom = 0.1 * self.SCALE * max(1 - self.t, 0) + self.ZOOM * self.SCALE * min(self.t, 1)
            scale, rot = zoom, self._rot_mat(self.car.hull.angle) # crucially, positive angle
            shift = np.array([600 / 2, 400 / 4]) + np.array(-1 * scale * self.car.hull.position) @ rot

            track_iloc = (self.track_iloc + self.lookahead) % len(self.track)
            xyc = self._transform_to_pixels(np.array([self.car.hull.position]), rot, scale, shift, H)
            xyt = self._transform_to_pixels(np.array(self.track_polys[track_iloc][0]), rot, scale, shift, H)
            xyc = xyc.astype(np.int32)
            xyt = xyt.astype(np.int32)

            # illustrate car, tile, and state-based cues
            cv.circle(frame, tuple(xyc.squeeze(0)), 3, (255, 0, 255), thickness=2)
            color = (0, 255, 255) if cues[1] == 1 else (255, 0, 0) # on track indicated by cyan
            for xy in xyt: cv.circle(frame, tuple(xy), 3, color, thickness=2)

            i_cues = np.array(cues).copy()[None,] # position and heading in magenta
            i_cues[:, 4:6] = self._transform_to_pixels(i_cues[:, 2:4] + i_cues[:, 4:6], rot, scale, shift, H)
            i_cues[:, 2:4] = self._transform_to_pixels(i_cues[:, 2:4], rot, scale, shift, H)
            i_cues = i_cues.squeeze(0).astype(np.int32)
            cv.arrowedLine(frame, tuple(i_cues[2:4]), tuple(i_cues[4:6]), color, thickness=2)

            if annotate and H > 64:
                # show labels, if available
                cv.putText(frame, f"{z_type}",
                            (10, 20),
                            cv.FONT_HERSHEY_COMPLEX,0.5,(0,128,128),1, lineType=cv.LINE_AA)
                cv.putText(frame, f"on track? {cues[1] == 1}",
                            (10, 40),
                            cv.FONT_HERSHEY_COMPLEX,0.5,(0,128,128),1, lineType=cv.LINE_AA)

        return frame

    def compute_cues(self, track_iloc=-1):
        R = self.R # use small window to compute curvature
        L = len(self.track) - 1
        if track_iloc == -1:
            self._update_track_iloc()
            track_iloc = (self.track_iloc + self.lookahead) % len(self.track)


        # (!) assuming looped track, compute curvature, tangents, and road type
        _, heading, x, y = self.track[track_iloc]

        # curvature: rate of change of unit tangent vec wrt arc len; dT/ds, where T=(df/dt)/|df/dt| and s=arc len
        c = 0  # cumulative curvature
        for s in range(track_iloc - R, track_iloc + R + 1):
            s = s % L # wrap to beginning of track
            T = [-1 * np.sin(self.track[s][1]), np.cos(self.track[s][1])]
            #c += (abs(self.track[s][1]) - abs(self.track[s + 1][1])) / np.sqrt(np.dot(T, T)) # odd results
            c += (abs(self.track[s][1] - self.track[s + 1][1])) / np.sqrt(np.dot(T, T))

        # containment test based on normals of tile edges, car spans ~3 tiles (they're small)
        on_track_label = False
        for s in range(self.track_iloc - 3, self.track_iloc + 2):
            s = s % len(self.track_polys) # wrap to beginning of track
            current_tile = np.array(self.track_polys[s][0])
            N = np.zeros_like(current_tile)  # normals of each tile edge
            N[:-1] = current_tile[1:] - current_tile[:-1]  # compute tile edges
            N[-1] = current_tile[0] - current_tile[-1]
            N = (N / np.linalg.norm(N, axis=-1, keepdims=True)) @ self._rot_mat(np.deg2rad(90)) # cw rotation
            V = np.array(self.car.hull.position) - current_tile  # point wrt to tile corner
            on_track_label |= np.all(np.diagonal(V @ N.T) > 0)

        tangent = np.array([-1 * np.sin(heading), np.cos(heading)])
        # todo: transform to pixel space?
        # scale and shift points using world-in-car transforms
        #H = frame.shape[0]
        #zoom = 0.1 * self.SCALE * max(1 - self.t, 0) + self.ZOOM * self.SCALE * min(self.t, 1)
        #scale, rot = zoom, self._rot_mat(self.car.hull.angle) # crucially, positive angle
        #shift = np.array([600 / 2, 400 / 4]) + np.array(-1 * scale * self.car.hull.position) @ rot
        #i_cues[:, 4:6] = self._transform_to_pixels(tangent, rot, scale, shift, H)
        #i_cues[:, 2:4] = self._transform_to_pixels(i_cues[:, 2:4], rot, scale, shift, H)

        if c > 1.20:
            speed_label = self.turn_types.index('sharp')
            speed = 0.2
        elif c > 0.90:
            speed_label = self.turn_types.index('right angle')
            speed = 0.4
        elif c > 0.10:
            #elif c > 0.05:
            speed_label = self.turn_types.index('curve')
            speed = 0.8
        else:
            speed_label = self.turn_types.index('straight')
            speed = 1

        # todo 7/19: distance to corner

        return [float(c), on_track_label, x, y, tangent[0], tangent[1], speed]

    def create_policy_args(self, z_type):
        z_class_size = 0 # on/off track
        if z_type == 'none':
            z_in_size = 0
        elif 'tangent-label' in z_type:
            z_class_size = 1
            z_in_size = 5  # tangent + onoff
        elif 'curvature-tangent' in z_type:
            z_in_size = 5  # curvature + tangent
        elif 'curvature-label' in z_type:
            z_class_size = 1
            z_in_size = 2  # curvature + onoff
        elif 'tangent' in z_type:
            z_in_size = 4  # tangent
        elif 'curvature' in z_type:
            z_in_size = 1 # curvature
        elif 'label' in z_type:
            z_class_size = 1
            z_in_size = 1 # onoff
        elif 'speed' in z_type:
            z_in_size = 1
        elif 'noise' in z_type:
            z_class_size = 1
            z_in_size = 7 # other
        else:
            raise ValueError(f'invalid z_type - {z_type}')

        return dict(z_type=z_type,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    z_in_size=z_in_size,
                    z_class_size=z_class_size)

    def slice_cues(self, z_type):
        """ returns subset of cues; only use in *_cues fns """
        if 'curvature-label' in z_type:
            return (0, 2) # curvature + label
        elif 'tangent' in z_type:
            return (4, 6) # tangent
        elif 'curvature' in z_type:
            return (0, 1) # curvature
        elif 'label' in z_type:
            return (1, 2) # onoff
        elif 'speed' in z_type:
            return (6, 7) # speed
        else:
            return (0, len(self.compute_cues()))

    def get_progress(self, cues):
        """ returns track curvature, car track position, and percent completion """
        self.max_on_track = max(self.max_on_track, self.tile_visited_count)
        return cues[0], np.array(self.car.hull.position), self.max_on_track / len(self.track)

    def save_progress(self, step, agent_dir, progress_metrics, z_type):
        # record first corner curvature
        iloc = 0
        while self.compute_cues(iloc)[0] < 0.05: iloc += 1
        curv = [self.compute_cues(iloc)[0]]
        while curv[-1] > 0.15:
            iloc += 1
            curv.append(self.compute_cues(iloc)[0])
        first_curv = max(curv)

        track_curvature, track_pos, pct_on_track = zip(*progress_metrics)
        vis_track(self, step, outpath=agent_dir, track_pos=track_pos)
        pickle.dump([track_curvature, track_pos], open(join(agent_dir, f"t-{step}_cues.pkl"), 'wb'))
        pickle.dump([first_curv, pct_on_track[-1], step], open(join(agent_dir, f"t-{step}_max_progress.pkl"), 'wb'))


#
# an old environment for back-compat
#

import torch
import gym
from gym.spaces import Box
from collections import deque
from copy import deepcopy
class CarRacing(gym.Wrapper):
    def __init__(self, frame_skip=0, frame_stack=4):
        self.env = gym.make("CarRacing-v1")
        super().__init__(self.env)

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frame_buf = deque(maxlen=frame_stack)

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes = 0

    @property
    def action_space(self):
        # note: for pwn, brake is hardcoded as negative gas -_-
        return Box(low=0, high=1, shape=(2,))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(self.frame_stack, 96, 96))

    def to_action_space(self, original_action, is_discrete=False):
        # todo: seperate into two parts, normalise and remap to real world action
        #       in this case, carracing policy generates only two actions (see
        #       action spec above)
        original_action = original_action.squeeze(0).cpu().to(torch.float).numpy()
        original_action = original_action * 2 - 1  # map from [0, 1] to [-1, 1]

        action = np.zeros(3)

        action[0] = original_action[0]

        # Separate acceleration and braking
        action[1] = max(0, original_action[1])
        action[2] = max(0, -original_action[1])

        return action

    def postprocess(self, original_observation):
        # convert to grayscale
        grayscale = np.array([0.299, 0.587, 0.114])
        observation = np.dot(original_observation, grayscale) / 255.0

        return observation

    def shape_reward(self, reward):
        return np.clip(reward, -1, 1)

    def get_observation(self):
        return np.array(self.frame_buf)

    def get_total_reward(self):
        # unshaped total reward from action repeats
        return self.total_reward

    def create_policy_args(self):
        return dict(z_type='none',
                    observation_space=self.observation_space,
                    action_space=self.action_space)

    def reset(self):

        self.t = 0
        self.last_reward_step = 0
        self.n_episodes += 1
        self.total_reward = 0

        first_frame = self.postprocess(self.env.reset())

        for _ in range(self.frame_stack):
            self.frame_buf.append(first_frame)

        return self.get_observation()

    def step(self, action):
        self.t += 1

        total_reward = 0
        for _ in range(self.frame_skip + 1):
            new_frame, reward, done, info = self.env.step(action)
            self.total_reward += reward
            reward = self.shape_reward(reward)
            total_reward += reward

            if reward > 0:
                self.last_reward_step = self.t

        if self.t - self.last_reward_step > 30:
            done = True

        reward = total_reward / (self.frame_skip + 1)

        real_frame = deepcopy(new_frame)

        new_frame = self.postprocess(new_frame)
        self.frame_buf.append(new_frame)

        return self.get_observation(), reward, done, info  #, real_frame
