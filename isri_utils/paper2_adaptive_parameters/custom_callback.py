# Based on https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py

from typing import Dict, Tuple
import gymnasium as gym
import numpy as np
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.pg.pg import PGConfig

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        # assert episode.length == 0, (
        #     "ERROR: `on_episode_start()` callback should be called right "
        #     "after env reset!"
        # )
        # Create lists to store angles in
        episode.user_data["crossover"] = []
        episode.user_data["mutation"] = []
        episode.user_data["pressure"] = []


    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        # assert episode.length > 0, (
        #     "ERROR: `on_episode_step()` callback should not be called right "
        #     "after env reset!"
        # )
        
        info = episode.last_info_for()
        episode.user_data['crossover'] = info['crossover']
        episode.user_data['mutation'] = info['muatation']
        episode.user_data['pressure'] = info['pressure']
    
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        episode.hist_data["crossover"] = episode.user_data["crossover"]
        episode.hist_data["mutation"] = episode.user_data["mutation"]
        episode.hist_data["pressure"] = episode.user_data["pressure"]