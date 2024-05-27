from _dir_init import *
from isri_optimizer.rl_sequential_agent.environment import IsriEnv
from sb3_contrib.ppo_mask import MaskablePPO
import pickle
from data_preprocessing import IsriDataset
from stable_baselines3.common.callbacks import CheckpointCallback
from isri_optimizer.rl_sequential_agent.custom_callback import CustomCallback
from stable_baselines3.dqn import DQN
from sklearn.cluster import KMeans
import torch as th

SAVE_NAME = "sparse" #Logname
SAVE_FREQUENCY = 100_000
TOTAL_TRAINING_STEPS = 800_000
MODEL_SAVE_DIR = f"./isri_optimizer/rl_sequential_agent/savefiles_0526/{SAVE_NAME}"
JOBDATA_DIR = './isri_optimizer/instances/'
SAVEFILE = f"./isri_optimizer/rl_sequential_agent/savefiles_0526/{SAVE_NAME}_best_chromosome"
N_TRAINING_INSTANCES = 1
GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl" #ToDo: Muss diese Datei auch aktualisiert werden?
N_TRIES = 1

# Loading instances and creating config
isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
isri_dataset.data['Jobdata'] = isri_dataset.data['Jobdata'][:N_TRAINING_INSTANCES]
isri_dataset.data['Files'] = isri_dataset.data['Files'][:N_TRAINING_INSTANCES]
isri_dataset.data['GAChromosome'] = isri_dataset.data['GAChromosome'][:N_TRAINING_INSTANCES]
isri_dataset.data['GAFitness'] = isri_dataset.data['GAFitness'][:N_TRAINING_INSTANCES]


env_config_kmeans_n8 = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 1.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 8, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}
env_config_kmeans_n12 = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 1.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 12, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}
env_config_kmeans_n15 = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 3,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 1.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 15, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans", #kmeans model übergeben
    "reward_type": "sparse" #sparse dense combined
}

env_config_neighbour_n8 = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 1.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 8, # Muss mit Kmeans übereinstimmen
    "cluster_method": "neighbour" #kmeans model übergeben
}
env_config_neighbour_n12 = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 1.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 12, # Muss mit Kmeans übereinstimmen
    "cluster_method": "neighbour" #kmeans model übergeben
}
env_config_neighbour_n15 = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 1.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 15, # Muss mit Kmeans übereinstimmen
    "cluster_method": "neighbour" #kmeans model übergeben
}



env_config_kmeans_n8_tard = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 0.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 8, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}
env_config_kmeans_n12_tard = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 0.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 12, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}
env_config_kmeans_n15_tard = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 0.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 15, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}

env_config_neighbour_n8_tard = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 0.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 8, # Muss mit Kmeans übereinstimmen
    "cluster_method": "neighbour" #kmeans model übergeben
}
env_config_neighbour_n12_tard = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 0.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 12, # Muss mit Kmeans übereinstimmen
    "cluster_method": "neighbour" #kmeans model übergeben
}
env_config_neighbour_n15_tard = {
    "jpl": 20,  # Example values, please adjust according to your needs
    "conv_speed": 208,
    "n_machines": 12,
    "n_lines": 1,
    "window_size": 4,
    "isri_dataset": isri_dataset,
    "next_n": 15,
    "last_n": 2,
    "input_features": 13,  # Example number of features per job
    "obs_space": 'simple', # simple, full, small
    "diffsum_weight": 0.0, #diffsum im tausender Bereich
    "DIFFSUM_NORM": 1.0,
    "tardiness_weight": 1.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 15, # Muss mit Kmeans übereinstimmen
    "cluster_method": "neighbour" #kmeans model übergeben
}

ppo_config = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.003,
    'n_steps': 2048, #The number of steps to run for each environment per update 
    # 'batch_size': 128, #Minibatch size
    'n_epochs': 10, #Number of epoch when optimizing the surrogate loss
    'gamma': 0.9999, #Discount factor
    'gae_lambda': 0.95, # gae_lambda: float = 0.95
    'ent_coef': 0.00, #Entropy coefficient for the loss calculation
    'clip_range': 0.1, #Clipping parameter
    #'tensorboard_log': MODEL_SAVE_DIR,
    'stats_window_size': 2048, #Window size for the rollout logging
    'verbose': True,
    'policy_kwargs': dict(activation_fn=th.nn.ReLU, 
                          net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
}



config_dict = {'kmeans_n8': env_config_kmeans_n8, 'kmeans_n12': env_config_kmeans_n12, 'kmeans_n15': env_config_kmeans_n15, 
               'neighbour_n8':env_config_neighbour_n8, 'neighbour_n12':env_config_neighbour_n12, 'neighbour_n15':env_config_neighbour_n15, 
               'kmeans_n8_tard': env_config_kmeans_n8_tard, 'kmeans_n12_tard': env_config_kmeans_n12_tard, 
               'kmeans_n15_tard': env_config_kmeans_n15_tard, 
               'neighbour_n8_tard':env_config_neighbour_n8_tard, 'neighbour_n12_tard':env_config_neighbour_n12_tard, 
               'neighbour_n15_tard':env_config_neighbour_n15_tard} # die Namen der items werden für savefile namen benutzt
config_dict2 = {'kmeans_n15_relu': env_config_kmeans_n15} # die Namen der items werden für savefile namen benutzt
if __name__ == '__main__':
    # import cProfile, pstats
    # profiler = cProfile.Profile()    
    # profiler.enable() 
    for name, config in config_dict2.items():
        for try_idx in range(N_TRIES):
            env = IsriEnv(config)
            ppo_config['env'] = env
            #dqn_config['env'] = env
            # save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY, save_path=MODEL_SAVE_DIR + f"/_{SAVE_NAME}")
            callback = CustomCallback()
            model = MaskablePPO(**ppo_config, tensorboard_log=f'{MODEL_SAVE_DIR}_{name}')
            # model = DQN(**dqn_config)
            # logname = f"Training_multiple_instances_gamma_{ppo_config['gamma']}_lr_{ppo_config['learning_rate']}_clip_range_{ppo_config['clip_range']}"
            model.learn(TOTAL_TRAINING_STEPS, tb_log_name=f'{SAVE_NAME}_{name}', callback=callback)
            # model.learn(TOTAL_TRAINING_STEPS, callback=callback)
            # profiler.disable()
            
            model.save(f'{MODEL_SAVE_DIR}_{name}' + f'/{SAVE_NAME}_{name}_{TOTAL_TRAINING_STEPS}_{try_idx}')
            with open(f'{MODEL_SAVE_DIR}_{name}' + f'/{SAVE_NAME}_{name}_config.txt', "w") as outfile:
                outfile.write(str(config))
                outfile.write("\n")
                outfile.write(str(ppo_config))
            # stats = pstats.Stats(profiler)
            # stats.dump_stats('isri_optimizer/rl_sequential_agent/training_profile.prof')
