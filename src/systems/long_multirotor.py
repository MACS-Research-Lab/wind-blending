from typing import Iterable

import numpy as np

from .multirotor import MultirotorTrajEnv



class LongTrajEnv:

    def __init__(self, waypoints: Iterable[np.ndarray], base_env: MultirotorTrajEnv):
        self.waypoints = waypoints
        self.base_env = base_env
        self.current_waypoint_idx = None


    def reset(self):
        self.current_waypoint_idx = 0
        normed_wp = self.waypoints[0] * 2 / (self.base_env.state_range[:3]+1e-6)
        waypt_vec = self.waypoints[self.current_waypoint_idx] - np.array([0,0,0])
        self.base_env._des_unit_vec = waypt_vec / (np.linalg.norm(waypt_vec)+1e-6)
        self.base_env.reset(uav_x=np.concatenate([np.zeros(12, np.float32), self.waypoints[self.current_waypoint_idx]]))
        return np.concatenate((self.base_env.state, normed_wp))


    def step(self, u: np.ndarray):
        assert self.current_waypoint_idx is not None, "Make sure to call the reset() method first."
        # coming from a tanh NN policy function
        u = np.clip(u, a_min=-1., a_max=1.)
        u = self.base_env.unnormalize_action(u)
        u = u + self.waypoints[self.current_waypoint_idx]
        u = self.base_env.normalize_action(u)

        done = False
        reward = 0
        # TODO: return info dict with done flags
        s, reward, _, info = self.base_env.step(u)

        if info.get('reached'):
            self.current_waypoint_idx += 1
            # if full traj is finished
            if self.current_waypoint_idx == len(self.waypoints):
                done = True
                normed_wp = self.waypoints[-1] * 2 / (self.base_env.state_range[:3]+1e-6)
                reward += 100
            else:
                self.base_env.reset(uav_x=np.concatenate([self.base_env.x[:12], self.waypoints[self.current_waypoint_idx]]))
                waypt_vec = self.waypoints[self.current_waypoint_idx] - self.waypoints[self.current_waypoint_idx-1]
                self.base_env._des_unit_vec = waypt_vec / np.linalg.norm(waypt_vec) # do I need to norm this?
                normed_wp = self.waypoints[self.current_waypoint_idx] * 2 / (self.base_env.state_range[:3]+1e-6)
        else:
            normed_wp = self.waypoints[self.current_waypoint_idx-1] * 2 / (self.base_env.state_range[:3]+1e-6)
        s = np.concatenate((s, normed_wp))

        # reward calculation

        # done calculations
        if not done:
            # tipped, out of bounds from the info dict
            if info.get('tipped') or info.get('outofbounds') or info.get('outoftime'):
                done = True
                # can set negative reward here

        # state is a 12+3 element vector, where last 3
        # elements are the normalized next waypoint
        return s, reward, done, info
    
        