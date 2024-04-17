import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete

class Env(gym.Env):
    def __init__(self, env_config):
        pass

    def reset(self, seed=None, options=None):
        obs = self.calc_obs()
        info = {}
        return obs, info

    def step(self, action):
        obs = self._convert_obs()
        terminated = False
        return obs, reward, terminated, done, info

    def close(self):
        pass


if __name__ == '__main__':
    # print(calc_norm_values_obs())
    env_config = {}
    env = Env(env_config)
    # print("Reset")
    # print(f"obs: {obs}, info: {info}")

    for i in range(2):
        obs, info = env.reset()
        done = False
        while not done:
            test_action = env.action_space.sample()
            # print(f"test action {test_action}")
            obs, reward, terminated, done, info = env.step(test_action)