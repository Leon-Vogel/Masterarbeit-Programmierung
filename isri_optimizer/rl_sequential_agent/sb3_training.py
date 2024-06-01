from _dir_init import *
from isri_optimizer.rl_sequential_agent.environment import IsriEnv
from isri_optimizer.rl_sequential_agent.environment_no_cluster import IsriEnv_no_cluster
from sb3_contrib.ppo_mask import MaskablePPO
import pickle
from data_preprocessing import IsriDataset
from stable_baselines3.common.callbacks import CheckpointCallback
from isri_optimizer.rl_sequential_agent.custom_callback import CustomCallback
from stable_baselines3.dqn import DQN
from sklearn.cluster import KMeans
import torch as th
from itertools import product
from typing import Callable
from data_split import train_test


SAVE_FREQUENCY = 100_000
TOTAL_TRAINING_STEPS = 1000_000
MODEL_SAVE_DIR = f"./isri_optimizer/rl_sequential_agent/savefiles_test/"
JOBDATA_DIR = './isri_optimizer/instances/'
SAVEFILE = f"./isri_optimizer/rl_sequential_agent/savefiles_test/_best_chromosome"
N_TRAINING_INSTANCES = 500
GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl" #ToDo: Muss diese Datei auch aktualisiert werden?
N_TRIES = 1

data = train_test(min_length=20, max_length=100, path=GA_SOLUTIONS_PATH, N_TRAINING_INSTANCES=N_TRAINING_INSTANCES ,all_data=True, save=True)
# isri_dataset, test_dataset = data.get_mixed_data() #für unterschiedlich große Instanzen
isri_dataset, isri_dataset_test = data.get_data() #für gleichgroße Instanzen
# Loading instances and creating config
#isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
#isri_dataset.data['Jobdata'] = isri_dataset.data['Jobdata'][:N_TRAINING_INSTANCES]
#isri_dataset.data['Files'] = isri_dataset.data['Files'][:N_TRAINING_INSTANCES]
#isri_dataset.data['GAChromosome'] = isri_dataset.data['GAChromosome'][:N_TRAINING_INSTANCES]
#sri_dataset.data['GAFitness'] = isri_dataset.data['GAFitness'][:N_TRAINING_INSTANCES]


def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        if progress_remaining > 0.4:
            return progress_remaining * initial_value
        else:
            return 0.4*initial_value


    return func

env_config = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'classes', # simple, full, small, classes
    "diffsum_weight": 1/300,#1/30000, #0.1, #diffsum im tausender Bereich
    "diffsum_weight_sum": 1/30000, 
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1/4, #1.0
    "tardiness_weight_sum": 1/20, #1/20, 
    "TARDINESS_NORM": 1.0,
    "pca": None
}

env_config_variants = {
    "last_n": [3], #20
    "reward_type": ["dense"], #sparse dense combined sparse_sum , "sparse_sum"
    "n_classes": [8, 12, 15], # Muss mit Kmeans übereinstimmen
    "cluster_method": ["kmeans", "no_cluster", "neighbour"] #kmeans neighbour no_cluster
}


keys, values = zip(*env_config_variants.items())
combinations = [dict(zip(keys, combination)) for combination in product(*values)]
envs = {}
for combination in combinations:
    name = "_".join(f"{value}" for key, value in combination.items())
    new_dict = env_config.copy()
    new_dict.update(combination)
    envs[name] = new_dict


ppo_config = {
    'policy': 'MlpPolicy',
    #'learning_rate': 0.0003,
    'n_steps': 1024, #1024, #The number of steps to run for each environment per update 
    'batch_size': 128, #Minibatch size
    'n_epochs': 10, #Number of epoch when optimizing the surrogate loss
    'gamma': 0.9999, #Discount factor
    'gae_lambda': 0.95, # gae_lambda: float = 0.95
    'ent_coef': 0.01, #Entropy coefficient for the loss calculation
    'clip_range': 0.2, #Clipping parameter
    #'clip_range_vf': 0.2,
    #'tensorboard_log': MODEL_SAVE_DIR,
    'stats_window_size': 100, #Window size for the rollout logging
    'verbose': True,
    'policy_kwargs': dict(activation_fn=th.nn.LeakyReLU, 
                          net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])) #512, 
}


if __name__ == '__main__':
    for name, config in envs.items():
        for try_idx in range(N_TRIES):
            if config["cluster_method"] == "no_cluster":
                env = IsriEnv_no_cluster(config)
            else:
                env = IsriEnv(config)
            ppo_config['env'] = env
            callback = CustomCallback(path=f'{MODEL_SAVE_DIR}_{name}' + f'/{name}_best_model')
            model = MaskablePPO(**ppo_config, learning_rate=linear_schedule(0.0005) ,tensorboard_log=f'{MODEL_SAVE_DIR}_{name}') #, ent_coef=linear_schedule(0.001)
            model.learn(TOTAL_TRAINING_STEPS, tb_log_name=f'{name}', callback=callback)
            model.save(f'{MODEL_SAVE_DIR}_{name}' + f'/{name}_{TOTAL_TRAINING_STEPS}_{try_idx}')
            with open(f'{MODEL_SAVE_DIR}_{name}' + f'/{name}_config.txt', "w") as outfile:
                outfile.write(str(config))
                outfile.write("\n")
                outfile.write(str(ppo_config))
