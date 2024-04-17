from _dir_init import *
from misc_utils.action_masking_model import TorchActionMaskModel
from ray import tune
from ray.air.config import RunConfig, CheckpointConfig, ScalingConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.ppo import PPOConfig
torch, nn = try_import_torch()
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.algorithms.soo.nonconvex.ga import GeneticAlgorithm
from gymnasium.utils.env_checker import check_env
import pymoo_utils
from isri_utils.paper2_adaptive_parameters.rl_agent import adaptation_env
from cec_optimization_default import cecProblem
from cec2017 import functions
import numpy as np
from ray import tune
import datetime
from problem_factory import get_permutation_problem
from custom_callback import MyCallbacks

#INPUT************************************************************************************************
RESULT_PATH = r'./isri_utils/paper2_adaptive_parameters/results/'
EXP_NAME = 'Trained_on_ta21_to_ta33_new_obs_space'
CHECKPOINT_SAVE_FREQUENCY = 10000
TRAINING_STEPS = 300000
N_EVALS = 6000
POP_SIZE = 100
DIMS = 30
CPUS = 25
PROBLEM = 'pfssp'
#*****************************************************************************************************

# RL Setup
environment = get_permutation_problem(PROBLEM)
tune.register_env("HPAdaptation", lambda config: environment)

def trial_name_creator(trial):
    return f'trial_{trial.trial_id}'

def trial_dirname_creator(trial):
    return f'trial_{trial.trial_id}'

def stopper(trial_id, result):
    if result['timesteps_total'] > TRAINING_STEPS:
        return True
    else:
        return False

algo_config = PPOConfig().rollouts(num_rollout_workers=CPUS).environment('HPAdaptation').training(
        lr=1.e-5,
        train_batch_size = 4096,
        entropy_coeff=0,
        gamma=0.999
        ).framework('torch').reporting(keep_per_episode_custom_metrics=True)

tuner = tune.Tuner(
        'PPO',
        tune_config=tune.tune_config.TuneConfig(
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
            num_samples=1,
            max_concurrent_trials=31
        ),
        param_space=algo_config,
        run_config=RunConfig(
            name=EXP_NAME,
            local_dir=RESULT_PATH,  # Local dir to save training results to.
            stop=stopper,  # e.g. stop={"training_iteration": 10}
            # callbacks=[CustomTuneCallback(self.exp_dir_configs, self.exp_checkpoint_sub_dir)],
            # progress_reporter=CustomReporter(sresults_logger, max_report_frequency=100),
            # Progress reporter for reporting intermediate experiment progress. Defaults to CLIReporter if running in command-line
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=CHECKPOINT_SAVE_FREQUENCY,
                checkpoint_at_end=True,
            ),
        ),
    )

if __name__ == "__main__":
    results = tuner.fit()
    # custom_metrics = results.metrics["custom_metrics"]