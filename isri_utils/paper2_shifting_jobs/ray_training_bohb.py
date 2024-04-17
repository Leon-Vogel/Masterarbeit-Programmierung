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
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB


#********************************************       SETTINGS     ***************************************************
RESULT_PATH = os.path.abspath(r'.\rl_results')
EXP_NAME = 'dist1.02_hpsearch_narrowactions'
CHECKPOINT_SAVE_FREQUENCY = 10000
#*****************************************************************************************************

def trial_name_creator(trial):
    return f't_epslenf{trial.config["env_config"]["episode_len_factor"]}'
    # return "trial"


def trial_dirname_creator(trial):
    return f't_epslenf{trial.config["env_config"]["episode_len_factor"]}'


def stopper(trial_id, result):
    if result['timesteps_total'] > 1.5e6:
        return True
    else:
        return False


env_config = {
        'curriculum_learning': False,
        'use_action_masking': False,
        "episode_len_factor": 5,
        "reward_definitions": {
            "reward_for_improving_the_last_schedule": 0.2,
            "reward_for_improving_the_initial_schedule": 2.5,
            "reward_factor_for_final_improvement": 0.5,
            "punishment_for_final_deterioration": 2,
        }
    }

# env_config_grid = []
# episode_len_factors = [5]
# for epslenf in episode_len_factors:
#     env_config_temp = env_config.copy()
#     env_config_temp["episode_len_factor"] = epslenf
#     env_config_grid.append(env_config_temp)

# env_config_grid = []
# REWARD_FOR_IMPROVING_THE_LAST_SCHEDULES = [0.1, 0.2]
# REWARD_FOR_IMPROVING_THE_INITIAL_SCHEDULES = [2.5, 5]
# REWARD_FACTOR_FOR_FINAL_IMPROVEMENTS = [0.5, 1.0]
# PUNISHMENT_FOR_FINAL_DETERIORATIONS = [1, 2]
# for r1, r2, r3, r4 in itertools.product(REWARD_FOR_IMPROVING_THE_LAST_SCHEDULES,
#                                         REWARD_FOR_IMPROVING_THE_INITIAL_SCHEDULES,
#                                         REWARD_FACTOR_FOR_FINAL_IMPROVEMENTS,
#                                         PUNISHMENT_FOR_FINAL_DETERIORATIONS):
#     env_config_temp = env_config.copy()
#     env_config_temp["reward_definitions"] = {
#         "reward_for_improving_the_last_schedule": r1,
#         "reward_for_improving_the_initial_schedule": r2,
#         "reward_factor_for_final_improvement": r3,
#         "punishment_for_final_deterioration": r4,
#     }
#     env_config_grid.append(env_config_temp)


algo_config = {
        "observation_filter": "MeanStdFilter",
        "env": Env,
        "env_config": env_config,
        # "env_config": tune.grid_search(env_config_grid), # , tune.choice(env_configs),
        "use_critic": True,
        "use_gae": True,
        "lambda": tune.quniform(0.95, 1.0, 0.01), #0.95,  # tune.quniform(0.95, 1.0, 0.01),
        "kl_target": 0.005, # tune.grid_search([0.003, 0.01, 0.02]), # 0.003,
        "kl_coeff": 0.6, # tune.grid_search([0.2, 0.8]), # 0.3,  # tune.quniform(0.5, 0.8, 0.1),
        "shuffle_sequences": True,
        "vf_loss_coeff": tune.quniform(0.5, 1, 0.1), # 0.5, # tune.grid_search([0.5, 1.0]),  # tune.quniform(0.5, 1, 0.1),
        "entropy_coeff": 0.003,  # tune.quniform(0.000, 0.005, 0.001),
        # "entropy_coeff_schedule": tune.grid_search([[[0, 0.01], [1e5, 0.008], [2e5, 0.002]], [[0, 0.003], [1e6, 0.003]]]), # [[0, 0.005], [3e5, 0.001]], # tune.grid_search([[[0, 0.01], [1500000, 0.000]], [[0, 0.01], [2500000, 0.000]], [[0, 0.005], [2500000, 0.005]]]),  # None
        "clip_param": tune.quniform(0.1, 0.3, 0.05), # 0.1, # tune.grid_search([0.1, 0.25]), # 0.1,  # tune.quniform(0.1, 0.3, 0.05),
        "vf_clip_param": tune.randint(7, 30), # 10, # tune.grid_search([10, 20]),
        # tune.grid_search([20, 300, 50, 500, 200, 100, 150]), LinearSchedule(schedule_timesteps=300000, initial_p=50, final_p=500, framework='torch'),
        "grad_clip": 40,
        # "lr_schedule": [[0, 2e-4], [7e5, 4e-5]],  # [[0, 3e-4], [4000000, 5e-5]],
        "lr": tune.quniform(0.000005, 0.0001, 0.000001), #1e-4, # 1.7e-4, # tune.quniform(0.00008, 0.0003, 0.00001),  # 1.7e-4, # tune.loguniform(0.00009, 0.0004),
        "gamma": 0.99,  # tune.quniform(0.91, 0.96, 0.01),
        "num_workers": 0,  # if 0, then use same CPU for worker and driver
        # "num_envs_per_worker": 1,
        "framework": "torch",
        "eager_tracing": True,
        "rollout_fragment_length": 'auto',
        "train_batch_size": tune.choice([1024, 2048, 4096, 8192]), # 1024, #1024, # tune.grid_search([1024, 2048, 4096, 8192]),  # 1024 oder 2048
        "sgd_minibatch_size": tune.choice([32, 64, 128, 256]), #32,  # tune.choice([32, 64]),
        "num_sgd_iter": tune.randint(10, 30), # 15,  # tune.randint(10, 20), # 15
        "model": {
            "fcnet_hiddens": [256, 256], # tune.choice([[128, 128], [64, 64], [256, 256]])
            "fcnet_activation": "relu"
        }
}
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

bohb_hyperband = HyperBandForBOHB(
        time_attr="timesteps_total",
        max_t=1.5e6,
        reduction_factor=4,
        stop_last_trials=False,
    )

bohb_search = TuneBOHB(
    # space=config_space,  # If you want to set the space manually
)
bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=31)

tuner = tune.Tuner(
        'PPO',
        tune_config=tune.tune_config.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
            num_samples=100,
            # max_concurrent_trials=2
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