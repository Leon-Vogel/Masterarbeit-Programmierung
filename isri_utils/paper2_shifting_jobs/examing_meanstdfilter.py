import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ray.rllib.algorithms.ppo import PPO
import numpy as np
import pandas as pd
import json
from environment import Env


EXPERIMENT = "dist1.02_3obj_smallerR_200k"
TRIAL = "trial"
PATH_TO_TRIAL = os.path.join(r'.\rl_results', EXPERIMENT, TRIAL)


json_config_path = os.path.join(PATH_TO_TRIAL, 'params.json')
with open(json_config_path) as json_config_file:
    config = json.load(json_config_file)
num_workers_orig = config['num_workers']
config['num_workers'] = 1
del config['env']

max_ch = 0
ch_path_existing = False
for ch_path in os.listdir(PATH_TO_TRIAL):
    if "checkpoint" in ch_path:
        ch_path_existing = True
        if int(ch_path.split('_')[1]) > max_ch:
            max_ch = int(ch_path.split('_')[1])
            checkpoint_path = os.path.join(PATH_TO_TRIAL, ch_path)
if not ch_path_existing:
    print(f"No Checkpoint in {PATH_TO_TRIAL}.")
agent = PPO(env=Env, config=config)
agent.restore(checkpoint_path)
print(f"Agent restored from saved model at {checkpoint_path}")

env = Env(config["env_config"])
raw_obs, info = env.reset()

# https://www.johndcook.com/blog/standard_deviation/
# https://github.com/ray-project/ray/blob/master/rllib/utils/filter.py
MeanStdFilter = agent.workers.local_worker().filters['default_policy']
# print(agent.local_evaluator.filters)
print(type(MeanStdFilter.running_stats))
print("n")
print(MeanStdFilter.running_stats.n)
print("\nnum_pushes")
print(MeanStdFilter.running_stats.num_pushes)
print("\nmean")
print(MeanStdFilter.running_stats.mean)
# print("\nmean_array")
# print(MeanStdFilter.running_stats.mean_array)
print("\nstd")
print(MeanStdFilter.running_stats.std)
# print("\nstd_array")
# print(MeanStdFilter.running_stats.std_array)
print("Raw Observation:", raw_obs)
print(dir(MeanStdFilter))
print("Preprocessed Observation:", MeanStdFilter(raw_obs))