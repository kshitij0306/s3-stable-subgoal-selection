import numpy as np
import argparse
from collections import deque
from gym import spaces

import envs.create_maze_env


def get_goal_sample_fn(env_name, evaluate):
    if env_name == 'AntMaze':
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntMazeSparse':
        return lambda: np.array([2., 9.])
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name in ['AntMaze', 'AntPush']:
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntMazeSparse':
        return lambda obs, goal: float(np.sum(np.square(obs[:2] - goal)) ** 0.5 < 1)
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def get_success_fn(env_name):
    if env_name in ['AntMaze', 'AntPush', 'AntFall']:
        return lambda reward: reward > -5.0
    elif env_name == 'AntMazeSparse':
        return lambda reward: reward > 1e-6
    else:
        assert False, 'Unknown env'


class GatherEnv(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.count = 0

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        obs = self.base_env.reset()
        self.count = 0
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }

    def step(self, a):
        obs, reward, done, info = self.base_env.step(a)
        self.count += 1
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }
        return next_obs, reward, done or self.count >= 500, info

    @property
    def action_space(self):
        return self.base_env.action_space


class EnvWithGoal(object):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.success_fn = get_success_fn(env_name)
        self.goal = None
        self.distance_threshold = 5 if env_name in ['AntMaze', 'AntPush', 'AntFall'] else 1
        self.count = 0
        self.early_stop = False if env_name in ['AntMaze', 'AntPush', 'AntFall'] else True
        self.early_stop_flag = False

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self, xy=None):
        # self.viewer_setup()
        self.early_stop_flag = False
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        
        obs = self.base_env.reset()
        if xy is not None:
            self.set_xy(xy)
        obs = self.base_env._get_obs() if hasattr(self.base_env, "_get_obs") else self.base_env.base_env._get_obs()
        self.count = 0
        self.goal = self.goal_sample_fn()
        self.desired_goal = self.goal if self.env_name in ['AntMaze', 'AntPush', 'AntFall'] else None
        

        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        done = self.early_stop_flag and self.count % 10 == 0
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }
        return next_obs, reward, done or self.count >= 500, info

    # def render(self):
        # self.base_env.render()
    def render(self, mode: str = "human"):
        """
        mode="human"     → open / update the on-screen Mujoco viewer
        mode="rgb_array" → return an (H,W,3) uint8 frame
        """
        if mode == "human":
            # Mujoco-py / gym-mujoco already knows how to draw a window
            return self.base_env.render(mode="human")
        elif mode == "rgb_array":
            # Off-screen rendering for video files
            return self.base_env.render(mode="rgb_array")
        else:
            raise NotImplementedError(
                f"EnvWithGoal: render mode '{mode}' not supported."
            )

    def set_xy(self, xy):
        self.base_env.set_xy(xy)

    @property
    def action_space(self):
        return self.base_env.action_space
