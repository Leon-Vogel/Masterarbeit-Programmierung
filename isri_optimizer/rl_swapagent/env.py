from _dir_init import *
import os
import pickle
import numpy as np
import gymnasium as gym
from typing import Dict
# from gymnasium.spaces import Graph, Box, Discrete, MultiDiscrete, Dict, Tuple, Sequence

from isri_optimizer.isri_metaheuristic_simple import fast_sim_diffsum, plan_by_due_date


class LocalOperators:
    def __init__(self):
        pass

    @staticmethod
    def swap(genome: np.ndarray, pos1: int, pos2: int):
        if pos1 < 0 or pos2 < 0 or pos1 >= len(genome) or pos2 >= len(genome):
            raise ValueError("pos1 und pos2 müssen gültige Indizes im Array sein")

        # Tausche die Elemente an den Positionen pos1 und pos2
        genome[pos1], genome[pos2] = genome[pos2], genome[pos1]
        return genome

    @staticmethod
    def two_opt():
        raise NotImplementedError

    @staticmethod
    def relocate():
        raise NotImplementedError


class IsriEnv(gym.Env):
    def __init__(self, env_config: Dict):
        self._read_config(env_config)
        self.local_operators = LocalOperators()
        self.genome = np.array(list(self.jobdata.keys()))  # Permutation der Keys in Jobdata als np.array, z.B. [23039847988, 23039847987...]
        self.num_jobs = self.jpl
        # TODO: Sorge dafür, dass zunächst nur die Jobs beachtet werden, die auf die Linie passen und der Rest weggeschnitten wird aus jobdata ODER bei reset

        # self.observation_space = gym.spaces.? # TODO
        self.action_space = gym.spaces.Discrete(self.num_jobs*self.num_jobs)

    def reset(self, *, seed=None, options=None):
        self.genome = plan_by_due_date(jobs=self.jobdata, jpl=self.jpl,
                                       n_lines=self.n_lines)  # Sortierung nach Due Dates
        # TODO: Sorge dafür, dass zunächst nur die Jobs beachtet werden, die auf die Linie passen und der Rest weggeschnitten wird aus jobdata ODER bei init
        obs = self._convert_genome_to_obs()
        info = self._get_info()

        # Calc init parameters for diffsum and tardiness
        diffsum, tardiness = fast_sim_diffsum(
            ind=self.genome,  # Permutation bzw. Reihenfolge der Jobs
            jobs=self.jobdata,  # Informationen über alle Jobs (times & due_date). Nicht sortiert.
            jpl=self.jpl,  # Jobs per Line
            conv_speed=self.conv_speed,  # Geschwindigkeit der Linie bzw. Zeit pro Aufgabe am Arbeitsplatz
            n_machines=self.n_machines,  # Anzahl Maschinen pro Linie
            n_lines=self.n_lines,  # Anzahl Linien
            window_size=self.window_size # Wie viele Nachbarn werden berücksichtigt, default=4
        )
        print(f"diffsum: {diffsum}, tardiness: {tardiness}")
        self.best_diffsum = -diffsum  # Minus wegen Pymoo (kann nur minimieren). Wir nutzen den pos. Wert
        self.best_tardiness = tardiness
        self.best_reward = 0
        self.timesteps = 0
        return obs, info

    def step(self, action: int):
        self.genome = self._perform_local_operator(action)
        print(f"self.genome: {self.genome}")
        reward = self._get_reward()
        obs = self._convert_genome_to_obs()
        terminated = False
        info = self._get_info()
        self.timesteps += 1
        if self.timesteps >= self.T:
            terminated = True
        return obs, reward, terminated, False, info

    def render(self):
        # Gantt-Diagramm zeigen + Metriken
        pass

    def close(self):
        pass

    def _read_config(self, env_config: Dict):
        # Lese config-Dict ein und speichere die Einträge als Instanzvariablen
        self.jpl = env_config["jpl"]  # Jobs per Line
        self.conv_speed = env_config["conv_speed"]  # Geschwindigkeit der Linie bzw. Zeit pro Aufgabe am Arbeitsplatz
        self.n_machines = env_config["n_machines"]  # Anzahl Maschinen pro Linie
        self.n_lines = env_config["n_lines"]  # Anzahl Linien
        self.window_size = env_config["window_size"]
        self.jobdata = env_config["jobdata"]  # Informationen über alle Jobs (times & due_date). Nicht sortiert.
        self.T = env_config["T"]  # Horizon, max Anzahl an Steps
        self.DIFFSUM_WEIGHT = env_config["DIFFSUM_WEIGHT"]
        self.DIFFSUM_NORM = env_config["DIFFSUM_NORM"]
        self.TARDINESS_WEIGHT = env_config["TARDINESS_WEIGHT"]
        self.TARDINESS_NORM = env_config["TARDINESS_NORM"]

    def _project_fitnessscores_to_reward(self, diffsum: float, tardiness: float):
        # Erwarte positiven Wert für diffsum
        if diffsum < 0:
            diffsum = -1 * diffsum
        # Wir wollen diffsum_weight maximieren und tardiness minimieren
        diffsum_delta = (diffsum - self.best_diffsum) / self.DIFFSUM_NORM
        tardiness_delta = (self.best_tardiness - tardiness) / self.TARDINESS_NORM
        r = self.DIFFSUM_WEIGHT * diffsum_delta + self.TARDINESS_WEIGHT * tardiness_delta
        return r

    def _get_reward(self):
        diffsum, tardiness = fast_sim_diffsum(
            ind=self.genome,  # Permutation bzw. Reihenfolge der Jobs
            jobs=self.jobdata,  # Informationen über alle Jobs (times & due_date). Nicht sortiert.
            jpl=self.jpl,  # Jobs per Line
            conv_speed=self.conv_speed,  # Geschwindigkeit der Linie bzw. Zeit pro Aufgabe am Arbeitsplatz
            n_machines=self.n_machines,  # Anzahl Maschinen pro Linie
            n_lines=self.n_lines,  # Anzahl Linien
            window_size=self.window_size  # Wie viele Nachbarn werden berücksichtigt, default=4
        )
        diffsum = -diffsum  # Minus wegen Pymoo (kann nur minimieren). Wir nutzen den pos. Wert
        print(f"diffsum: {diffsum}, tardiness: {tardiness}")
        r = self._project_fitnessscores_to_reward(diffsum, tardiness)
        if r > self.best_reward:
            reward = r - self.best_reward  # Weise nur die Differenz zum bisher besten Support als Reward zu
            self.best_tardiness = tardiness
            self.best_diffsum = diffsum
            self.best_reward = r
        else:
            reward = 0  # wenn keine Verbesserung erzielt werden konnte, dann reward = 0
        return reward

    def _get_info(self):
        return None

    def _perform_local_operator(self, action):
        position_in_flattened_array = action
        pos1 = position_in_flattened_array % self.num_jobs  # col
        pos2 = position_in_flattened_array // self.num_jobs  # row
        new_genome = self.local_operators.swap(self.genome, pos1, pos2)
        return new_genome

    def _convert_genome_to_obs(self):
        obs = self.genome  # TODO
        return obs


if __name__ == '__main__':
    # with open(os.path.join(PROJECT_ROOT_DIR, "isri_optimizer", 'rl_swapagent', 'instance_15'), 'rb') as file:
    #     jobdata = pickle.load(file)
    # env_config = {
    #     "jpl": 48,
    #     "conv_speed": 208,
    #     "n_machines": 12,
    #     "n_lines": 1,
    #     "window_size": 4,
    #     "jobdata": jobdata[0],
    #     "T": 200,
    #     "DIFFSUM_WEIGHT": 0.2, # Wert von GA
    #     "DIFFSUM_NORM": 50,
    #     "TARDINESS_WEIGHT": 0.8, # Wert von GA
    #     "TARDINESS_NORM": 50,
    # }
    with open(os.path.join(PROJECT_ROOT_DIR, "isri_optimizer", 'rl_swapagent', 'instance_easy'), 'rb') as file:
        jobdata = pickle.load(file)
    env_config = {
        "jpl": 10,
        "conv_speed": 208,
        "n_machines": 4,
        "n_lines": 1,
        "window_size": 4,
        "jobdata": jobdata,
        "T": 5,
        "DIFFSUM_WEIGHT": 0.2,  # Wert von GA
        "DIFFSUM_NORM": 50,
        "TARDINESS_WEIGHT": 0.8,  # Wert von GA
        "TARDINESS_NORM": 50,
    }
    env = IsriEnv(env_config)
    obs, info = env.reset()
    done = False
    i = 0
    while not done:
        action = np.random.randint(0, env_config["jpl"] * env_config["jpl"])
        obs, reward, done, _, info = env.step(action)
        i += 1
        print(f"Performed {i} steps.")

