from _dir_init import *
from isri_optimizer.rl_sequential_agent.environment import IsriEnv
from sb3_contrib.ppo_mask import MaskablePPO
import pickle
from data_preprocessing import IsriDataset
from stable_baselines3.common.callbacks import CheckpointCallback
from isri_optimizer.rl_sequential_agent.custom_callback import CustomCallback
from stable_baselines3.dqn import DQN
from sklearn.cluster import KMeans

SAVE_NAME = "simple_obs_sparse_reward_single_instance_dqn" #Logname
SAVE_FREQUENCY = 100_000
TOTAL_TRAINING_STEPS = 250_000
MODEL_SAVE_DIR = f"./isri_optimizer/rl_sequential_agent/savefiles/{SAVE_NAME}"
JOBDATA_DIR = './isri_optimizer/instances/'
SAVEFILE = f"./isri_optimizer/rl_sequential_agent/savefiles/{SAVE_NAME}_best_chromosome"
N_TRAINING_INSTANCES = 1
GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl" #ToDo: Muss diese Datei auch aktualisiert werden?
N_TRIES = 10

# Loading instances and creating config
isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
isri_dataset.data['Jobdata'] = isri_dataset.data['Jobdata'][:N_TRAINING_INSTANCES]
isri_dataset.data['Files'] = isri_dataset.data['Files'][:N_TRAINING_INSTANCES]
isri_dataset.data['GAChromosome'] = isri_dataset.data['GAChromosome'][:N_TRAINING_INSTANCES]
isri_dataset.data['GAFitness'] = isri_dataset.data['GAFitness'][:N_TRAINING_INSTANCES]

    
#kmeans in notebook vorbereiten und per pickle übertragen

env_config = {
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
    "tardiness_weight": 2.0, 
    "TARDINESS_NORM": 1.0,
    "pca": None,
    "n_classes": 8, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}

ppo_config = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.005,
    'n_steps': 2048,
    'gamma': 0.9999,
    'ent_coef': 0.01,
    'tensorboard_log': MODEL_SAVE_DIR,
    'stats_window_size': 2048,
    'verbose': True,
    "policy_kwargs": dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
}

dqn_config = {
    'policy': 'MlpPolicy',
    'gamma': 0.9999,
    #'tensorboard_log': MODEL_SAVE_DIR,
    'stats_window_size': 2048,
    'verbose': True,
    'train_freq': 1024
    # "policy_kwargs": dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
}

if __name__ == '__main__':
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    for try_idx in range(N_TRIES):
        env = IsriEnv(env_config)
        ppo_config['env'] = env
        dqn_config['env'] = env
        # save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY, save_path=MODEL_SAVE_DIR + f"/_{SAVE_NAME}")
        callback = CustomCallback()
        model = MaskablePPO(**ppo_config)
        # model = DQN(**dqn_config)
        # logname = f"Training_multiple_instances_gamma_{ppo_config['gamma']}_lr_{ppo_config['learning_rate']}_clip_range_{ppo_config['clip_range']}"
        model.learn(TOTAL_TRAINING_STEPS, tb_log_name=SAVE_NAME, callback=callback)
        # model.learn(TOTAL_TRAINING_STEPS, callback=callback)
        # profiler.disable()
        model.save(MODEL_SAVE_DIR + f'/{SAVE_NAME}_{TOTAL_TRAINING_STEPS}_{try_idx}')
        with open(MODEL_SAVE_DIR + f'/{SAVE_NAME}_config.txt', "w") as outfile:
            outfile.write(str(env_config))
            outfile.write("\n")
            outfile.write(str(ppo_config))
        # stats = pstats.Stats(profiler)
        # stats.dump_stats('isri_optimizer/rl_sequential_agent/training_profile.prof')
