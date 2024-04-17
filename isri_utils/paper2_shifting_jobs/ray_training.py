import itertools
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from isri_utils.paper2_shifting_jobs.environment import Env
from misc_utils.action_masking_model import TorchActionMaskModel
from ray import tune
from ray.air.config import RunConfig, CheckpointConfig, ScalingConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks

#********************************************       SETTINGS     ***************************************************
RESULT_PATH = os.path.abspath(r'.\rl_results')
EXP_NAME = 'dist1.02_3obj_smallerR_POL_DR_ta30'
CHECKPOINT_SAVE_FREQUENCY = 100
TIMESTEPS_TOTAL_FOR_TRAINING = 300e3
#*****************************************************************************************************


def trial_name_creator(trial):
    return f't_epslen{trial.config["env_config"]["episode_len"]}'
    # return "trial"


def trial_dirname_creator(trial):
    return f't_epslen{trial.config["env_config"]["episode_len"]}'
    # return "trial"


def stopper(trial_id, result):
    if result['timesteps_total'] > TIMESTEPS_TOTAL_FOR_TRAINING:
        return True
    else:
        return False


env_config = {
        'curriculum_learning': False,
        'use_action_masking': False,
        "episode_len": 75,
        # "problem_instance": "ta041", # "ta041" "ta001"
        "DR": True, # False
        "problem_instances": ["ta031", "ta032", "ta033", "ta034", "ta035", ],
        # "exp_name": EXP_NAME,
        # "timesteps_total_for_training": TIMESTEPS_TOTAL_FOR_TRAINING,
        # "num_workers": NUM_WORKERS,
        "reward_definitions": {
            "reward_for_improving_the_last_schedule": 0.2,
            "reward_for_improving_the_initial_schedule": 0.2,
            "reward_factor_for_final_improvement": 2,
            "punishment_for_final_deterioration": 2,
            "pain_of_living_per_step": 0.1
        }
    }

# env_config_grid = []
# episode_lens = [50, 100, 200]
# for epslen in episode_lens:
#     env_config_temp = env_config.copy()
#     env_config_temp["episode_len_factor"] = epslenf
#     env_config_grid.append(env_config_temp)

# env_config_grid = []
# for eps_len, inst in itertools.product([50, 100, 200], ["ta011", "ta041", "ta081"]):
#     env_config_temp = env_config.copy()
#     env_config_temp["episode_len"] = eps_len
#     env_config_temp["problem_instance"] = inst
#     env_config_grid.append(env_config_temp)


algo_config = {
        "observation_filter": "MeanStdFilter",
        "env": Env,
        "env_config": env_config,
        # "env_config": tune.grid_search(env_config_grid), # , tune.choice(env_configs),
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.95,  # tune.quniform(0.95, 1.0, 0.01),
        "kl_target": 0.005, # tune.grid_search([0.003, 0.01, 0.02]), # 0.003,
        "kl_coeff": 0.6, # tune.grid_search([0.2, 0.8]), # 0.3,  # tune.quniform(0.5, 0.8, 0.1),
        "shuffle_sequences": True,
        "vf_loss_coeff": 0.5, # tune.grid_search([0.5, 1.0]),  # tune.quniform(0.5, 1, 0.1),
        "entropy_coeff": 0.001,  # tune.quniform(0.000, 0.005, 0.001),
        # "entropy_coeff_schedule": tune.grid_search([[[0, 0.01], [1e5, 0.008], [2e5, 0.002]], [[0, 0.003], [1e6, 0.003]]]), # [[0, 0.005], [3e5, 0.001]], # tune.grid_search([[[0, 0.01], [1500000, 0.000]], [[0, 0.01], [2500000, 0.000]], [[0, 0.005], [2500000, 0.005]]]),  # None
        "clip_param": 0.1, # tune.grid_search([0.1, 0.25]), # 0.1,  # tune.quniform(0.1, 0.3, 0.05),
        "vf_clip_param": 10, # tune.grid_search([10, 20]),
        # tune.grid_search([20, 300, 50, 500, 200, 100, 150]), LinearSchedule(schedule_timesteps=300000, initial_p=50, final_p=500, framework='torch'),
        "grad_clip": 40,
        # "lr_schedule": [[0, 2e-4], [7e5, 4e-5]],  # [[0, 3e-4], [4000000, 5e-5]],
        "lr": 1e-4, # 5e-5, #tune.grid_search([1e-4, 0.5e-4, 0.1e-4]), #1e-4, # 1.7e-4, # tune.quniform(0.00008, 0.0003, 0.00001),  # 1.7e-4, # tune.loguniform(0.00009, 0.0004),
        "gamma": 0.99,  # tune.quniform(0.91, 0.96, 0.01),
        "num_workers": 20,  # if 0, then use same CPU for worker and driver
        # "num_envs_per_worker": 1,
        "framework": "torch",
        "eager_tracing": True,
        "rollout_fragment_length": 'auto',
        "train_batch_size": 1024, #1024, # tune.grid_search([1024, 2048, 4096, 8192]),  # 1024 oder 2048
        "sgd_minibatch_size": 64, #32,  # tune.choice([32, 64]),
        "num_sgd_iter": 15, # 15,  # tune.randint(10, 20), # 15
        "model": {
            "fcnet_hiddens": [256, 256], #tune.choice([[256, 256], [512, 512]]), # [256, 256],  # tune.choice([[128, 128], [64, 64], [256, 256]]),
            "fcnet_activation": "relu"
        }}
if env_config['curriculum_learning']:
    algo_config["callbacks"] = MyCallbacks
if env_config['use_action_masking']:
    ModelCatalog.register_custom_model('action_mask_model', TorchActionMaskModel)
    algo_config_actionmasking = {
        "model": {
            "custom_model": "action_mask_model",
            "vf_share_layers": True,
            #  "custom_model_config": {"no_masking": not USE_ACTION_MASKING},  # needed for the TorchModel
        },
    }
    algo_config['model'] = {**algo_config['model'], **algo_config_actionmasking['model']}

tuner = tune.Tuner(
        'PPO',
        tune_config=tune.tune_config.TuneConfig(
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
            num_samples=2,
            max_concurrent_trials=32
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
                num_to_keep=1,
                checkpoint_frequency=CHECKPOINT_SAVE_FREQUENCY,
                checkpoint_at_end=True,
            ),
        ),
    )

if __name__ == "__main__":
    results = tuner.fit()