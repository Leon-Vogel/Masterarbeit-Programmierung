import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from read_taillard_data import get_taillard_with_uncert_proc_time
from agent_facade import AgentFacade
from create_neighbor import SelectNextJobAction, ShiftCurrentJobAction, ChangeCurrentJobDistanceAction
import numpy as np


class Env(gym.Env):
    def __init__(self, env_config):
        # self.observation_space = Box(-1, 1e6, shape=(62,))
        self.env_config = env_config
        self.observation_space = Box(-1, 1e6, shape=(47,))
        num_actions_SCJA = len([attr for attr in ShiftCurrentJobAction.__dict__ if not attr.startswith("__")])
        num_actions_CCJDA = len([attr for attr in ChangeCurrentJobDistanceAction.__dict__ if not attr.startswith("__")])
        num_actions_SNJA = len([attr for attr in SelectNextJobAction.__dict__ if not attr.startswith("__")])
        self.action_space = MultiDiscrete([num_actions_SCJA, num_actions_CCJDA, num_actions_SNJA])
        self.reward_definitions = env_config["reward_definitions"]
        # self.num_eps = 0
        # self.path_to_actions = rf"C:\Projekte\SUPPORT\Software\project_support\src\executable\isri_utils\paper2_shifting_jobs\rl_results\{self.env_config['exp_name']}\actions.txt"
        # self.actions_write_criterion = (self.env_config["timesteps_total_for_training"]/
        #                                 self.env_config["episode_len"] - 100) / self.env_config["num_workers"]

    def reset(self, seed=None, options=None):
        # verwende die Shortest Processing Time Priorirätsregel als Eröffnungsverfahren
        if self.env_config["DR"]:
            inst = np.random.choice(self.env_config["problem_instances"])
            ind = get_taillard_with_uncert_proc_time(inst)
        else:
            ind = get_taillard_with_uncert_proc_time(self.env_config["problem_instance"])
        ind = sorted(ind, key=lambda x: x["operations"][0]["expected_duration"])

        # instanziiere die Fassade (API), die vom gym-Env. für alle operationen
        # verwendet werden kann
        fitness_features = ["flowtime_sum", "worker_load", "job_uncertainty_mean"]
        fitness_weights = [0.4, 0.4, 0.2]
        self.af = AgentFacade(ind, fitness_features, fitness_weights,
                              episode_len=self.env_config["episode_len"],
                              reward_definitions=self.reward_definitions)
        obs, n_steps = self.af.get_first_observation_and_episode_len(ind)
        obs = self._convert_obs(obs)
        info = {}
        # self.actions = []
        return obs, info

    def step(self, action):
        action_SCJA = action[0]
        action_CCJDA = action[1]
        action_SNJA = action[2]
        obs, reward, done, info = self.af.perform_action_and_get_new_observation_and_reward(action_SCJA,
                                                                                      action_CCJDA,
                                                                                      action_SNJA)
        obs = self._convert_obs(obs)
        terminated = False
        # if self.num_eps > self.actions_write_criterion:
        #     self.actions.append(action.tolist())
        # if done:
        #     self.num_eps += 1
        #     if self.num_eps > self.actions_write_criterion:
        #         with open(self.path_to_actions, "a") as f:
        #             f.writelines(f"{action}," for action in self.actions)
        #             f.write("\n")

        return obs, reward, terminated, done, info

    def close(self):
        pass

    def _convert_obs(self, obs):
        # filter out data
        # normalize data - use env_config
            # run environment 100x mit random samples only
            # übernehme Wert in dictionary, falls dieser größer ist als voriger Wert
            # Falls Wert==0, dann ersetze durch 1
            # speichere als normalisation factor
        # convert into np.array
        obs = np.array(list(obs.values()))
        # clip auf 1
        return obs


def calc_norm_values_obs():
    env_config = {}
    env = Env(env_config)
    obs, info = env.reset()

    def save_max_value(obs_orig, obs_new):
        for key, val in obs_orig.items():
            if val < obs_new[key]:
                obs_orig[key] = obs_new[key]
        return obs_orig

    for i in range(100):
        print(f"Run {i}")
        obs_new, reward, terminated, done, info = env.step(env.action_space.sample())
        obs = save_max_value(obs, obs_new)
    return obs


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
            # print(f"obs: {obs}, reward: {reward}, termin: {terminated}, done: {done}, info: {info}")



# 'n_jobs': 20,
# 'n_machines': 5,
# 'worker_load': 251.13327846364882,
# 'flowtime_sum': 36830,
# 'flowtime_mean': 1841.5,
# 'flowtime_std': 732.3177247615955,
# 'job_uncertainty_mean': 0.06580190695209416,
# 'makespan': 2944.0,
# 'distance_pattern_levenshtein': 0,
# 'distance_pattern_mean_len_sequences': 1.0,
# 'processing_sum': 6486.0,
# 'processing_mean': 64.86,
# 'processing_std': 27.098406528127,
# 'slack_std_machines': 3.762656900360533,
# 'slack_std_jobs': 10.761993245442891,
# 'slack_mean': 61.50526315789474,
# 'slack_sum': 5843.0,
# 'slack_sum_q1': 1794.0,
# 'slack_sum_q2': 1483.0,
# 'slack_sum_q3': 1329.0,
# 'slack_sum_q4': 1237.0,
# 'slack_std_q1': 23.0259777680322,
# 'slack_std_q2': 23.465014043421558,
# 'slack_std_q3': 26.252197348272553,
# 'slack_std_q4': 29.835752067767213,
# 'has_pre_job': 1,
# 'has_post_job': 1,
# 'curr_processing_mean': 79.4,
# 'curr_processing_std': 26.23547217032695,
# 'curr_high_distance': 0,
# 'curr_uncertainty': 0.02966792921067522,
# 'curr_slack_mean': 47.6,
# 'curr_slack_std': 26.235472170326954,
# 'curr_high_slack_mean': 0,
# 'curr_high_slack_std': 1,
# 'curr_high_processing_mean': 1,
# 'curr_high_processing_std': 0,
# 'curr_high_uncertainty': 0,
# 'curr_pos_rel': 0.85,
# 'curr_flowtime': 2667,
# 'pre_processing_mean': 77.2,
# 'pre_processing_std': 28.75239120490677,
# 'pre_high_distance': 0,
# 'pre_uncertainty': 0.08844135927667585,
# 'pre_slack_mean': 49.8,
# 'pre_slack_std': 28.75239120490677,
# 'pre_high_slack_mean': 0,
# 'pre_high_slack_std': 1,
# 'pre_high_processing_mean': 1,
# 'pre_high_processing_std': 1,
# 'pre_high_uncertainty': 1,
# 'post_processing_mean': 68.6,
# 'post_processing_std': 32.63127334322092,
# 'post_high_distance': 0,
# 'post_uncertainty': 0.04064122992555612,
# 'post_slack_mean': 58.4,
# 'post_slack_std': 32.63127334322092,
# 'post_high_slack_mean': 0,
# 'post_high_slack_std': 1,
# 'post_high_processing_mean': 1,
# 'post_high_processing_std': 1,
# 'post_high_uncertainty': 0