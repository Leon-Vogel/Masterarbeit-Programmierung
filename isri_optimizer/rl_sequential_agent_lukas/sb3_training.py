from _dir_init import *
from isri_optimizer.rl_sequential_agent_lukas.environment import IsriEnv
from sb3_contrib.ppo_mask import MaskablePPO
import pickle
from isri_optimizer.rl_sequential_agent_lukas.data_preprocessing import IsriDataset
from data_preprocessing import IsriDataset
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from isri_optimizer.rl_sequential_agent_lukas.custom_callback import CustomCallback
from stable_baselines3.dqn import DQN
#from isri_optimizer.rl_sequential_agent_lukas.custom_model import CustomDictTransformerFeatureExtractor, TransformerPointingPolicy, CustomLSTMExtractor, CustomLSTMExtractor2
import gymnasium as gym
import numpy as np
#from torch.utils.data import Dataset
from stable_baselines3.common.logger import configure

SAVE_NAME = "mlp_model_350_instances"
SAVE_FREQUENCY = 100_000
TOTAL_TRAINING_STEPS = 500_000
MODEL_SAVE_DIR = f"./isri_optimizer/rl_sequential_agent_lukas/savefiles/{SAVE_NAME}"
JOBDATA_DIR = './isri_optimizer/instances/'
SAVEFILE = f"./isri_optimizer/rl_sequential_agent_lukas/savefiles/{SAVE_NAME}_best_chromosome"
N_TRAINING_INSTANCES = 350
GA_SOLUTIONS_PATH = "./isri_optimizer/rl_sequential_agent_lukas/data/IsriDataDict.pkl"
N_TRIES = 10

# Loading instances and creating config
def load_data(GA_SOLUTIONS_PATH, N_TRAINING_INSTANCES):
    isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
    isri_dataset['Jobdata'] = isri_dataset['Jobdata'][:N_TRAINING_INSTANCES]
    isri_dataset['Files'] = isri_dataset['Files'][:N_TRAINING_INSTANCES]
    isri_dataset['GAChromosome'] = isri_dataset['GAChromosome'][:N_TRAINING_INSTANCES]
    isri_dataset['GAFitness'] = isri_dataset['GAFitness'][:N_TRAINING_INSTANCES]
    return isri_dataset

isri_dataset = load_data(GA_SOLUTIONS_PATH, N_TRAINING_INSTANCES)
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
    "obs_space": 'simple', # simple, full, small, transformer
    "goal": 'both', # Workload, both
    "use_heuristic": True 
}


policy_kwargs = dict(
    # features_extractor_class=CustomDictTransformerFeatureExtractor,
    # features_extractor_kwargs=dict(input_features_dim=13, hidden_dim=64, num_layers=3),
    # features_extractor_class=CustomLSTMExtractor2,
    # features_extractor_kwargs=dict(input_features_dim=13, hidden_dim=16, num_layers=3),
    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    # net_arch=dict(pi=[15,], vf=[15,])
)

ppo_config = {
    # 'policy': TransformerPointingPolicy,
    'policy': 'MlpPolicy',
    'learning_rate': 0.0001,
    'n_steps': 1024,
    'gamma': 0.9999,
    'tensorboard_log': MODEL_SAVE_DIR,
    'stats_window_size': 100,
    'verbose': True,
    "policy_kwargs": policy_kwargs
}

def make_env(env_id, seed):
    def _f():
        # Aus irgendeinem Grund muss das in jedem Prozess einzeln initialisiert werden ...
        gym.register("Isri", 'environment:IsriEnv', 'env_config')
        env = gym.make(env_id, env_config=env_config)
        env.seed = seed
        return env
    return _f

if __name__ == '__main__':
    for idx in range(N_TRIES):
        if env_config['obs_space'] == 'transformer':
            envs = [make_env('Isri', idx) for idx in range(4)]
            env = SubprocVecEnv(envs)
        else:
            env = IsriEnv(env_config)
        ppo_config['env'] = env
        logger = configure(MODEL_SAVE_DIR + f'/{SAVE_NAME}_{idx}', ['csv', "tensorboard"])
        callback = CustomCallback()
        model = MaskablePPO(**ppo_config)
        model.set_logger(logger)
        # logname = f"Training_multiple_instances_gamma_{ppo_config['gamma']}_lr_{ppo_config['learning_rate']}_clip_range_{ppo_config['clip_range']}"
        # model.learn(TOTAL_TRAINING_STEPS, tb_log_name=SAVE_NAME)
        model.learn(TOTAL_TRAINING_STEPS, callback=callback, tb_log_name=SAVE_NAME)
        model.save(MODEL_SAVE_DIR + f'/{SAVE_NAME}_{TOTAL_TRAINING_STEPS}_{idx}')
        with open(MODEL_SAVE_DIR + f'/{SAVE_NAME}_config.txt', "w") as outfile:
            outfile.write(str(env_config))
            outfile.write("\n")
            outfile.write(str(ppo_config))

