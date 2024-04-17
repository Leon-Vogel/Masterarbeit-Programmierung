'''
Dieses Skript durchläuft mit allen trainierten Agents (checkpoints) in dem Pfad PATH_TO_EXPERIMENT eine Episode und
speichert die dabei ermittelten Metriken unter demselben Pfad ab.
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ray.rllib.algorithms.ppo import PPO
import numpy as np
import pandas as pd
import json
from environment import Env
from copy import deepcopy


#********************************************       SETTINGS     ***************************************************
EXPLORING_AGENT_DURING_INFERENCE = False
EXPERIMENT = "dist1.02_3obj_smallerR_POL_DR_ta30"
NUMBER_OF_RUNS_PER_AGENT = 20
TEST_ON_UNSEEN_DATA = False
#*******************************************************************************************************************
PATH_TO_EXPERIMENT = os.path.join(r'.\rl_results', EXPERIMENT)
PATHS = [path.name for path in os.scandir(PATH_TO_EXPERIMENT) if path.is_dir()]


if __name__ == '__main__':
    columns_for_evaluation_matrix = ["flowtime_sum_max", "flowtime_sum_min", "flowtime_sum_median", "flowtime_sum_mean",
         "flowtime_sum_std", "worker_load_max", "worker_load_min", "worker_load_median", "worker_load_mean",
         "worker_load_std",
         "job_uncertainty_mean_max", "job_uncertainty_mean_min", "job_uncertainty_mean_median",
                                     "job_uncertainty_mean_mean", "job_uncertainty_mean_std",
         "fitness_score_max", "fitness_score_min", "fitness_score_median", "fitness_score_mean",
         "fitness_score_std", "return_max", "return_min", "return_median", "return_mean", "return_std",
         "episode_len", "problem_instance", "reward_for_improving_the_last_schedule",
         "reward_for_improving_the_initial_schedule",
         "reward_factor_for_final_improvement", "punishment_for_final_deterioration",
         "number_of_runs", "train_batch_size",
         "lr", "num_workers", "sgd_minibatch_size", "num_sgd_iter", "entropy_coeff",
         "clip_param", "kl_coeff", "kl_target", "lambda", "vf_clip_param",
         "vf_loss_coeff", "path", "actions_first_run"]  # "terminated",
    df = pd.DataFrame(columns=columns_for_evaluation_matrix)

    i = 1
    for path in PATHS:
        agent_path = os.path.join(PATH_TO_EXPERIMENT, path)
        json_config_path = os.path.join(agent_path, 'params.json')
        with open(json_config_path) as json_config_file:
            config = json.load(json_config_file)
        num_workers_orig = config['num_workers']
        config['num_workers'] = 1
        config["env_config"]["DR"] = False
        config["env_config"]["problem_instance"] = "ta041"
        del config['env']
        if TEST_ON_UNSEEN_DATA:
            config["env_config"]["problem_instance"] = config["env_config"]["problem_instance"].replace("1", "2")
        # für Action Masking und Curriculum Learning
        # ModelCatalog.register_custom_model('action_mask_model', TorchActionMaskModel)
        # if "callbacks" in config:
        #     del config["callbacks"]

        # Init agent
        max_ch = 0
        ch_path_existing = False
        for ch_path in os.listdir(agent_path):
            if "checkpoint" in ch_path:
                ch_path_existing = True
                if int(ch_path.split('_')[1]) > max_ch:
                    max_ch = int(ch_path.split('_')[1])
                    checkpoint_path = agent_path + "\\" + ch_path
        if not ch_path_existing:
            print(f"No Checkpoint in {agent_path}.")
            continue
        agent = PPO(env=Env, config=config)
        agent.restore(checkpoint_path)
        MeanStdFilter = agent.workers.local_worker().filters['default_policy']
        print(f"Agent restored from saved model at {checkpoint_path}")

        # TODO: For-Schleife wieder löschen
        config_orig = deepcopy(config)
        for inst in ["ta031", "ta032", "ta033", "ta034", "ta035", "ta036", "ta037", "ta038", "ta039", "ta040"]: # ["ta041", "ta042", "ta043", "ta044", "ta045", "ta046"]
            config = deepcopy(config_orig)
            config["env_config"]["problem_instance"] = inst
            # Initialize env
            env = Env(config['env_config'])
            print(f"Environment was initialized.")

            flowtime_sum = []
            worker_load = []
            job_uncertainty_mean = []
            fitness_score = []
            returns = []
            j = 1
            for _ in range(NUMBER_OF_RUNS_PER_AGENT):
                reward_sum = 0
                if j == 1:
                    actions = []

                obs, info = env.reset()
                done = False
                while not done:
                    obs = MeanStdFilter(obs)
                    action = agent.compute_single_action(obs, explore=EXPLORING_AGENT_DURING_INFERENCE)
                    if j == 1:
                        actions.append(action.tolist())
                    obs, reward, terminated, done, info = env.step(action)
                    reward_sum += reward
                flowtime_sum.append(info["flowtime_sum"])
                worker_load.append(info["worker_load"])
                job_uncertainty_mean.append(info["job_uncertainty_mean"])
                fitness_score.append(info["fitness_score"])
                returns.append(reward_sum)
                j += 1

            # Prepare and store evaluation metrics
            eval_run_data = {**config, **config["env_config"], **config["env_config"]["reward_definitions"],
                                **{'num_workers': num_workers_orig, "path": agent_path,
                                   "number_of_runs": NUMBER_OF_RUNS_PER_AGENT}}
            eval_run_data = {**eval_run_data,
                             "return_max": np.max(returns),
                             "return_min": np.min(returns),
                             "return_std": np.std(returns),
                             "return_mean": np.mean(returns),
                             "return_median": np.median(returns),
                             "flowtime_sum_max": np.max(flowtime_sum),
                             "flowtime_sum_min": np.min(flowtime_sum),
                             "flowtime_sum_std": np.std(flowtime_sum),
                             "flowtime_sum_mean": np.mean(flowtime_sum),
                             "flowtime_sum_median": np.median(flowtime_sum),
                             "worker_load_max": np.max(worker_load),
                             "worker_load_min": np.min(worker_load),
                             "worker_load_std": np.std(worker_load),
                             "worker_load_mean": np.mean(worker_load),
                             "worker_load_median": np.median(worker_load),
                             "job_uncertainty_mean_max": np.max(job_uncertainty_mean),
                             "job_uncertainty_mean_min": np.min(job_uncertainty_mean),
                             "job_uncertainty_mean_std": np.std(job_uncertainty_mean),
                             "job_uncertainty_mean_mean": np.mean(job_uncertainty_mean),
                             "job_uncertainty_mean_median": np.median(job_uncertainty_mean),
                             "fitness_score_max": np.max(fitness_score),
                             "fitness_score_min": np.min(fitness_score),
                             "fitness_score_std": np.std(fitness_score),
                             "fitness_score_mean": np.mean(fitness_score),
                             "fitness_score_median": np.median(fitness_score),
                             "actions_first_run": [actions]
                             } # "terminated": terminated,
            eval_run_data = pd.DataFrame({k: v for k, v in eval_run_data.items() if k in df.columns}, index=[0])
            df = pd.concat([df, eval_run_data], axis=0, ignore_index=True)  # df.append(eval_run_data, ignore_index=True)

    UNSEEN_STRING = '_31-40_n20'
    if TEST_ON_UNSEEN_DATA:
        UNSEEN_STRING = '_unseen'
    df.to_excel(os.path.join(PATH_TO_EXPERIMENT, f"evaluation_{EXPERIMENT}{UNSEEN_STRING}.xlsx"), index=False)
    df.to_pickle(os.path.join(PATH_TO_EXPERIMENT, f"evaluation_{EXPERIMENT}{UNSEEN_STRING}.pkl"))
    # df_agg = aggregate_results(df)


