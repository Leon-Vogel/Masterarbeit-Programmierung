import os
import random
from typing import Callable

import numpy as np
import torch as T
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from CustomCallbacks import CustomCallback
from plantsim.plantsim import Plantsim
from ps_environment import Environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ToDo: Planerfüllung nicht mehr dokumentierien und nach anderen Redundanzen suchen
# ToDo: Maskierung von Skip bis alle Produkte freigegeben wurden

# pfad = 'E:\\Studium\Projekt\ActionMasking\PlantSimRL\simulations'
pfad = 'D:\\Studium\Projekt\ActionMasking\PlantSimRL\simulations'
# pfad = 'D:\\Studium\Projekt\ActionMasking\PlantSimRL\simulations'
erg = 'Ergebnisse\\Mask_Exp3\\'
mod = 'Models\\Mask_Exp3\\'  # _V1
net_arch = dict(pi=[256, 256, 128, 64], vf=[256, 256, 128, 64])
Training2 = {
    'Sim': [pfad + '\RL_Sim_Mask_Shaped.spp', pfad + '\RL_Sim_Mask_Auslastung.spp', pfad + '\RL_Sim_Mask_Dlz.spp', pfad + '\RL_Sim_Mask_Warteschlangen.spp'],
    'Logs': [erg + 'Mask_PPO_Shaped', erg + 'Mask_PPO_Auslastung', erg + 'Mask_PPO_Dlz', erg + 'Mask_PPO_Warteschlangen'],
    'Logname': [str(net_arch['pi']).replace(", ", "-") + '_Shaped_001_',
                str(net_arch['pi']).replace(", ", "-") + '_Auslastung_001_',
                str(net_arch['pi']).replace(", ", "-") + '_Dlz_001_',
                str(net_arch['pi']).replace(", ", "-") + '_Warteschlangen_001_'],
    'Model': [mod + 'Mask_PPO_Shaped', mod + 'Mask_PPO_Auslastung', mod + 'Mask_PPO_Dlz', mod + 'Mask_PPO_Warteschlangen']
}
Training = {
    'Sim': [pfad + '\RL_Sim_Mask_Auslastung.spp', pfad + '\RL_Sim_Mask_Warteschlangen.spp'],
    'Logs': [erg + 'Mask_PPO_Auslastung', erg + 'Mask_PPO_Warteschlangen'],
    'Logname': [str(net_arch['pi']).replace(", ", "-") + '_Auslastung_001_',
                str(net_arch['pi']).replace(", ", "-") + '_Warteschlangen_001_'],
    'Model': [mod + 'Mask_PPO_Auslastung', mod + 'Mask_PPO_Warteschlangen']
}
sim_count = len(Training['Sim'])

# os.makedirs(logs, exist_ok=True)
policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=net_arch)
# policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
eval_freq = 310000 # ToDo manuelles speichern der besten Modelle
n_eval_episodes = 2
learning_rate = 3e-4  # 4e-4
n_epochs = 10
batch_size = 128
n_steps = 1024  # 1024 384 2048
clip_range = 0.2
ent_coef = 0.001 # ToDo kleiner setzen und schauen ob Verhalten stabiler wird (linear schedule testen)
clip_range_vf = None
total_timesteps = 200000
visible = False  # False True
info_keywords = tuple(['Typ1', 'Typ2', 'Typ3', 'Typ4', 'Typ5', 'Avg', 'Warteschlangen', 'Auslastung'])
data = {'policy_kwargs': policy_kwargs, 'eval_freq': eval_freq, 'n_eval_episodes': n_eval_episodes,
        'learning_rate': learning_rate, 'n_epochs': n_epochs, 'n_steps': n_steps, 'clip_range': clip_range,
        'clip_range_vf': clip_range_vf, 'total_timesteps': total_timesteps, 'Sim': Training['Sim']}


# learning_rate-value
def lrsched():
    def reallr(progress):
        lr = learning_rate
        if progress < 0.5:
            lr = learning_rate * 0.8
        if progress < 0.3:
            lr = learning_rate * 0.6
        if progress < 0.1:
            lr = learning_rate * 0.4
        return lr

    return reallr


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


# clip-value
def clipsched():
    def realclip(progress):
        clip = clip_range
        if progress < 0.85:
            clip = clip_range
        if progress < 0.66:
            clip = clip_range  # * 0.8
        if progress < 0.33:
            clip = clip_range  # * 0.6
        return clip

    return realclip

def entsched():
    def realent(progress):
        ent = ent_coef
        if progress < 0.85:
            ent = ent_coef
        if progress < 0.66:
            ent = ent_coef  # * 0.8
        if progress < 0.33:
            ent = ent_coef  # * 0.6
        return ent

    return realent


def mask_fn(env: Environment) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


for i in range(sim_count):  # sim_count
    os.makedirs(Training['Logs'][i], exist_ok=True)
    with open(Training['Logs'][i] + '\\' + Training['Logname'][i] + '_Settings.txt', "w") as datei:
        # Die Werte in die Datei schreiben, einen pro Zeile
        for name, wert in data.items():
            datei.write(str(name) + ' = ' + str(wert) + "\n")
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell',
                        model=Training['Sim'][i], socket=None, visible=visible)
    env = Environment(plantsim)
    env = Monitor(env, Training['Logs'][i], info_keywords=info_keywords)
    env = ActionMasker(env, mask_fn)
    rollout_callback = CustomCallback(env)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000000, verbose=1)
    eval_callback = EvalCallback(eval_env=env, best_model_save_path=Training['Model'][i],
                                 log_path=Training['Logs'][i],
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, callback_on_new_best=stop_callback)

    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=Training['Logs'][i],
                        learning_rate=learning_rate, n_epochs=n_epochs,  # linear_schedule(learning_rate)
                        clip_range=clip_range,  # linear_schedule(clip_range)
                        device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
                        clip_range_vf=clip_range_vf, ent_coef=ent_coef,
                        batch_size=batch_size,
                        n_steps=n_steps, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=total_timesteps, callback=[rollout_callback, eval_callback],
                tb_log_name=Training['Logname'][i], progress_bar=True)
    model.save(Training['Model'][i] + '\\train_model')
    del model
'''
    # Evaluiere Agent, Ergebnisse Dokumentieren
    model = MaskablePPO.load(Training['Model'][i] + '\\best_model', env)
    
    obs = env.reset()
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    dones = False
    valid_action_array = np.array([1, 0])
    while not dones:
        action = model.predict(obs, action_masks=valid_action_array, episode_start=episode_starts,
                               deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones

    # Eval von 5 Plänen die nicht teil vom Training sind & Eval Rand actions
    open(Training['Logs'][i] + '\\' + Training['Logname'][i] + '_Testing.txt', "w")
    for j in range(sim_count):
        done = False
        reward_sum = 0
        steps = 0
        info = {}
        obs = env.reset(eval_mode=True, eval_step=j)
        episode_starts = np.ones((num_envs,), dtype=bool)
        while not done:
            steps += 1
            action = model.predict(obs, action_masks=valid_action_array, episode_start=episode_starts,
                                   deterministic=True)
            obs, rewards, done, info = env.step(action)
            episode_starts = done
            reward_sum += rewards
        with open(Training['Logs'][i] + '\\' + Training['Logname'][i] + '_Testing.txt', "a") as datei:
            # Die Werte in die Datei schreiben, einen pro Zeile
            datei.write('Steps = ' + str(steps) + "\n")
            datei.write('Return = ' + str(reward_sum) + "\n")
            datei.write('Info = ' + str(info) + "\n")
    # Eval von Random Actions
    open(Training['Logs'][i] + '\\' + Training['Logname'][i] + '_Random.txt', "w")
    for j in range(5):
        done = False
        reward_sum = 0
        steps = 0
        info = {}
        obs = env.reset(eval_mode=True, eval_step=j)
        while not done:
            steps += 1
            action = random.randint(0, 4)
            obs, rewards, done, info = env.step(action)
            reward_sum += rewards
        with open(Training['Logs'][i] + '\\' + Training['Logname'][i] + '_Random.txt', "a") as datei:
            # Die Werte in die Datei schreiben, einen pro Zeile
            datei.write('Steps = ' + str(steps) + "\n")
            datei.write('Return = ' + str(reward_sum) + "\n")
            datei.write('Info = ' + str(info) + "\n")
'''