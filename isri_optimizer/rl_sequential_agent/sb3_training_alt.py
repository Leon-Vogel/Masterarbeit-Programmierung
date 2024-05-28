from _dir_init import *
from isri_optimizer.rl_sequential_agent.environment_alt import IsriEnv
from sb3_contrib.ppo_mask import MaskablePPO
import pickle
from data_preprocessing import IsriDataset
from stable_baselines3.common.callbacks import CheckpointCallback
from isri_optimizer.rl_sequential_agent.custom_callback_alt import CustomCallback
from stable_baselines3.dqn import DQN
import torch as th

SAVE_NAME = "alt_model_sparse_last20"
SAVE_FREQUENCY = 100_000
TOTAL_TRAINING_STEPS = 400_000
MODEL_SAVE_DIR = f"./isri_optimizer/rl_sequential_agent/savefiles_0526/{SAVE_NAME}"
JOBDATA_DIR = './isri_optimizer/instances/'
SAVEFILE = f"./isri_optimizer/rl_sequential_agent/savefiles_0526/{SAVE_NAME}_best_chromosome"
N_TRAINING_INSTANCES = 1
GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent/IsriDataset.pkl"
N_TRIES = 1

# Loading instances and creating config
isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
isri_dataset.data['Jobdata'] = isri_dataset.data['Jobdata'][:N_TRAINING_INSTANCES]
isri_dataset.data['Files'] = isri_dataset.data['Files'][:N_TRAINING_INSTANCES]
isri_dataset.data['GAChromosome'] = isri_dataset.data['GAChromosome'][:N_TRAINING_INSTANCES]
isri_dataset.data['GAFitness'] = isri_dataset.data['GAFitness'][:N_TRAINING_INSTANCES]

    
env_config = {
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
    "DIFFSUM_WEIGHT": 1.0,
    "DIFFSUM_NORM": 1/10_000,
    "TARDINESS_WEIGHT": 1.0,
    "pca": None,
    "TARDINESS_NORM": 1/10_000,
    "n_classes": 15, # Muss mit Kmeans übereinstimmen
    "cluster_method": "kmeans" #kmeans model übergeben
}

ppo_config = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.005,
    'n_steps': 2048, #The number of steps to run for each environment per update 
    # 'batch_size': 128, #Minibatch size
    'n_epochs': 10, #Number of epoch when optimizing the surrogate loss
    'gamma': 0.9999, #Discount factor
    'gae_lambda': 0.95, # gae_lambda: float = 0.95
    'ent_coef': 0.00, #Entropy coefficient for the loss calculation
    'clip_range': 0.2, #Clipping parameter
    #'tensorboard_log': MODEL_SAVE_DIR,
    'stats_window_size': 2048, #Window size for the rollout logging
    'verbose': True,
    'policy_kwargs': dict(#activation_fn=th.nn.LeakyReLU, 
                          net_arch=dict(pi=[512, 256, 128, 64], vf=[512, 256, 128, 64]))
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
    for _ in range(N_TRIES):
        env = IsriEnv(env_config)
        ppo_config['env'] = env
        dqn_config['env'] = env
        # save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY, save_path=MODEL_SAVE_DIR + f"/_{SAVE_NAME}")
        callback = CustomCallback()
        model = MaskablePPO(**ppo_config, tensorboard_log=f'{MODEL_SAVE_DIR}')
        # model = DQN(**dqn_config)
        # logname = f"Training_multiple_instances_gamma_{ppo_config['gamma']}_lr_{ppo_config['learning_rate']}_clip_range_{ppo_config['clip_range']}"
        model.learn(TOTAL_TRAINING_STEPS, tb_log_name=f'{SAVE_NAME}', callback=callback)
        # model.learn(TOTAL_TRAINING_STEPS, callback=callback)
        # profiler.disable()
        model.save(MODEL_SAVE_DIR + f'/{SAVE_NAME}_{TOTAL_TRAINING_STEPS}')
        with open(MODEL_SAVE_DIR + f'/{SAVE_NAME}_config.txt', "w") as outfile:
            outfile.write(str(env_config))
            outfile.write("\n")
            outfile.write(str(ppo_config))
        # stats = pstats.Stats(profiler)
        # stats.dump_stats('isri_optimizer/rl_sequential_agent/training_profile.prof')
