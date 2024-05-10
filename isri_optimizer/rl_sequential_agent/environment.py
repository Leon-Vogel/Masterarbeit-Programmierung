from _dir_init import *
import gymnasium as gym
import numpy as np
from isri_optimizer.isri_metaheuristic_simple import fast_sim_diffsum, find_next_jobs_by_due_date
from typing import Dict
from misc_utils.copy_helper import fast_deepcopy
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from clustering import cluster_kmeans


def map_to_rgb(array: np.ndarray):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    cmap = plt.colormaps.get_cmap('RdYlGn')
    colors = cmap(normalized_array)
    return colors


class IsriEnv(gym.Env):
    def __init__(self, env_config: Dict):
        self._read_config(env_config)
        self.genome = []  # Permutation der Keys in Jobdata als np.array, z.B. [23039847988, 23039847987...]
        self.num_jobs = self.jpl
        self.steps = 0
        # Observation Size Large: Für die nächsten und letzten Aufträge jeweils die Arbeitszeiten + deadline und eine Distanzmatrix aufgeteilt in die Arbeitszeiten
        # observation_shape = (self.next_n + self.last_n) * self.features_per_job + self.next_n * self.next_n * (self.features_per_job - 1)
        # Observation Size Small: Pro next Job euclidische Distanz zum letztverplanten + Deadline + Distanzmatrix (see scipy.spatial.pdist)
        observation_shape = self.get_obs_space(env_config['obs_space'])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=observation_shape)
        self.action_space = gym.spaces.Discrete(self.next_n)
        self.unplanned_jobs_sorted = []
        self.plan_horizon = 0
        self.selected_index = None
        self.obs_type = env_config['obs_space']
        self.deadline_gap = 0
        self.workload_gap = 0
        self.balance_punishement = 0
        self.jobclasses = {idx: [] for idx in range(self.n_classes)}
        if self.cluster_method == "kmeans":
            self.cluster = cluster_kmeans(self.n_classes)
        


    def reset(self, *, seed=None, options=None):
        self.genome = []
        self.steps = 0
        random_index = random.choice(list(range(0, len(self.dataset.data['Jobdata']))))
        self.selected_index = random_index
        self.jobdata = self.dataset.data['Jobdata'][random_index]
        for job in self.jobdata:
            times = self.jobdata[job]['times']
            cluster = self.cluster_fn.predict(times)
            self.jobclasses[cluster].append(job)

        for cluster in range(self.n_classes):
            cluster_jobs = self.jobclasses[cluster]
            sorted_products = sorted(cluster_jobs, key=lambda x: self.jobdata[x]['due_date'])
            self.jobclasses[cluster] = sorted_products

        self.curr_target_values = self.dataset.data['GAFitness'][random_index]
        self.unplanned_jobs_sorted = sorted(self.jobdata, key=lambda x: self.jobdata[x]['due_date'])
        self.plan_horizon = len(self.jobdata) * self.conv_speed + self.n_machines * self.conv_speed
        first_job = self.unplanned_jobs_sorted.pop(0)
        self.genome.append(first_job)
        obs = self.make_obs()
        info = self._get_info()
        self.invalid_action_penalty = 0
        return obs, info

    def step(self, action: int):
        self._add_job_to_genome(action)
        # print(f"self.genome: {self.genome}")
        reward = self._get_reward_dense()
        obs = self.make_obs()
        terminated = False
        # info = self._get_info()
        if self.steps >= len(self.jobdata) - 1:
            terminated = True
        return obs, reward, terminated, False, {}

    def render(self):
        fig, axs = plt.subplots(6, 4)
        fig.tight_layout(h_pad=0.5)
        workload_data = np.array([self.jobdata[job_id]['times'] for job_id in self.genome])
        n_jobs = workload_data.shape[0]
        deadlines = np.array([self.jobdata[job_id]['due_date'] for job_id in self.genome])
        colors = map_to_rgb(deadlines)
        for idx, ax in enumerate(list(axs.flatten()[:12])):
            ax.bar(np.arange(0, n_jobs, 1), workload_data[:, idx], width=0.5, color=colors, alpha=0.8)
            ax.set_title(f'AP {idx} - RL')
            ax.set_xlabel('Produkte')
            ax.set_ylabel('MTM Zeit')
        
        ga_chromosome = self.dataset.data['GAChromosome'][self.selected_index]
        workload_data = np.array([self.jobdata[job_id]['times'] for job_id in ga_chromosome])
        deadlines = np.array([self.jobdata[job_id]['due_date'] for job_id in ga_chromosome])
        colors = map_to_rgb(deadlines)
        for idx, ax in enumerate(list(axs.flatten()[12:])):
            ax.bar(np.arange(0, n_jobs, 1), workload_data[:, idx], width=0.5, color=colors, alpha=0.8)
            ax.set_title(f'AP {idx} - GA')
            ax.set_xlabel('Produkte')
            ax.set_ylabel('MTM Zeit')
        
        plt.show()

    def close(self):
        pass

    def _read_config(self, env_config: Dict):
        # Lese config-Dict ein und speichere die Einträge als Instanzvariablen
        self.jpl = env_config["jpl"]  # Jobs per Line
        self.conv_speed = env_config["conv_speed"]  # Geschwindigkeit der Linie bzw. Zeit pro Aufgabe am Arbeitsplatz
        self.n_machines = env_config["n_machines"]  # Anzahl Maschinen pro Linie
        self.n_lines = env_config["n_lines"]  # Anzahl Linien
        self.window_size = env_config["window_size"]
        self.dataset = env_config["isri_dataset"]
        self.next_n = env_config["next_n"]  # Horizon, max Anzahl an Steps
        self.last_n = env_config["last_n"]
        self.features_per_job = env_config['input_features']
        self.diffsum_weight = env_config["diffsum_weight"]
        self.DIFFSUM_NORM = env_config["DIFFSUM_NORM"]
        self.tardiness_weight = env_config["tardiness_weight"]
        self.TARDINESS_NORM = env_config["TARDINESS_NORM"]
        self.pca = env_config['pca']
        self.n_classes = env_config['n_classes']
        self.cluster_fn = env_config['clustering']

    def _project_fitnessscores_to_reward(self, diffsum: float, tardiness: float):
        # Wir wollen diffsum_weight maximieren und tardiness minimieren
        diffsum_reward = diffsum * self.DIFFSUM_NORM * self.DIFFSUM_WEIGHT
        tardiness_reward = -tardiness * self.TARDINESS_NORM * self.TARDINESS_WEIGHT
        r = diffsum_reward + tardiness_reward
        return r

    def _get_reward_sparse(self):
        if self.steps == len(self.jobdata) - 1:
            diffsum, tardiness = fast_sim_diffsum(np.array(self.genome), jobs=self.jobdata, jpl=self.jpl,
                                                  conv_speed=self.conv_speed, n_machines=self.n_machines)
            diffsum_difference = (diffsum - self.curr_target_values[0]) / self.curr_target_values[0]
            tardiness_difference = (self.curr_target_values[1] - tardiness) / self.curr_target_values[1]
            # balance_reward = np.abs(diffsum_difference - tardiness_difference) # Belohnen wenn Ziele im selben Maß erreicht werden
            balance_reward = 0
            self.workload_gap = diffsum_difference
            self.deadline_gap = tardiness_difference
            self.balance_punishement = balance_reward
            return (diffsum_difference + tardiness_difference - balance_reward) * 100
        else:
            return -self.invalid_action_penalty
        
    def _get_reward_dense(self):
        # Letzte Diffsum Änderung
        last_job_times = np.array(self.jobdata[self.genome[-2]]['times']) / self.conv_speed
        planned_job_times =np.array(self.jobdata[self.genome[-1]]['times']) / self.conv_speed
        diffsum = np.sum(np.abs(last_job_times - planned_job_times))

        # Tardiness Änderung
        last_job_deadline = self.jobdata[self.genome[-1]['due_date']]
        current_finish_time = (self.steps + self.n_machines) * self.conv_speed
        tardiness = -np.exp((current_finish_time - last_job_deadline) / 3600) # 3600 Sekunden = 1 Stunde
        return diffsum * self.diffsum_weight + tardiness * self.tardiness_weight

    def _get_info(self):
        return None

    def _add_job_to_genome(self, action):
        selected_job = self.jobclasses[action].pop(0)
        self.unplanned_jobs_sorted.remove(selected_job)
        self.genome.append(selected_job)
        self.steps += 1

    def action_masks(self):
        return np.array([len(self.jobclasses[idx]) > 0 for idx in range(self.n_classes)])
        # return np.array([idx < len(self.unplanned_jobs_sorted) for idx in range(self.next_n)])
    
    def get_obs_space(self, obs_space):
        if obs_space == 'simple':
            obs_shape = (1, self.n_classes * 14 + self.last_n * 12 + 1) # Plus 1 für Step
            # obs_shape = (self.next_n + self.last_n, self.features_per_job)
        elif obs_space == 'full':
            obs_shape_len = (self.next_n + self.last_n) * self.features_per_job + self.next_n * self.next_n * (self.features_per_job - 1)
            obs_shape = (1, obs_shape_len)
        elif obs_space == 'small':
            obs_shape_len = int(self.next_n * 2 + ((self.next_n * (self.next_n - 1)) / 2))
            obs_shape = (1, obs_shape_len)
        else:
            raise ValueError(f"Unknown Observation type: {obs_space}")
        return obs_shape

    def make_obs(self):
        if self.obs_type == 'simple':
            return self._make_obs_simple()
        elif self.obs_type == 'full':
            return self._make_obs_full()
        elif self.obs_type == 'small':
            return self._make_obs_small()

    def _make_obs_full(self):
        # Extract times for the last self.last_n entries of planned_jobs
        last_n_array = np.ones((self.last_n, self.features_per_job)) * -10
        size_last_n = min((len(self.genome), self.last_n))
        last_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.genome[-size_last_n:]])
        last_n_array[-size_last_n:, :] = last_n_times
        # last_n_array /= self.conv_speed

        # next n jobs by due date
        next_n_array = np.ones((self.next_n, self.features_per_job)) * -10

        size_next_n = min((len(self.unplanned_jobs_sorted), self.next_n))
        next_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.unplanned_jobs_sorted[:size_next_n]])
        
        dist_matrix = np.zeros((self.next_n, self.next_n, (self.features_per_job-1))) # -1 weil das letzte Feature die Deadline ist

        if size_next_n > 0:
            # Difference instead of times
            next_n_times[:size_next_n, :self.n_machines] -= last_n_array[-1, :self.n_machines] * self.conv_speed
            # next_n_times[:size_next_n, -1] -= self.conv_speed * self.steps 
            next_n_array[:size_next_n, :] = next_n_times

            # Distance Matrix
            points_reshaped = next_n_times[:, :-1].reshape(size_next_n, 1, 12)
            points_transpose = next_n_times[:, :-1].reshape(1, size_next_n, 12)

            # Calculate the absolute difference for each pair of points
            distance_matrix = np.abs(points_reshaped - points_transpose)
            dist_matrix[:size_next_n, :size_next_n, :] = distance_matrix

        # next_n_array /= self.conv_speed
        obs = np.concatenate([last_n_array.flatten(), next_n_array.flatten(), dist_matrix.flatten()], axis=0)
        return obs
    
    def _make_obs_simple(self):
        # Extract times for the last self.last_n entries of planned_jobs
        last_n_array = np.ones((self.last_n, self.features_per_job)) * -1
        size_last_n = min((len(self.genome), self.last_n))
        last_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.genome[-size_last_n:]])
        last_n_array[-size_last_n:, :] = last_n_times

        # Normalisierung last n (MTM / Max Zeit, Deadline - vergangene Zeit / conv speed)
        last_n_array[-size_last_n:, :self.n_machines] /= self.conv_speed
        last_n_array[-size_last_n:, -1] = (last_n_array[-size_last_n:, -1] - self.conv_speed * self.steps) / self.conv_speed

        # next n jobs by due date
        next_n_array = np.ones((self.next_n, self.features_per_job)) * -1

        size_next_n = min((len(self.unplanned_jobs_sorted), self.next_n))
        next_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.unplanned_jobs_sorted[:size_next_n]])

        if size_next_n > 0:
            # Next N Jobs Data ( MTM + Deadline)
            next_n_times[:, :self.n_machines] /= self.conv_speed
            next_n_times[:, -1] = (next_n_times[:, -1] - self.conv_speed * self.steps) / self.conv_speed
            next_n_array[:size_next_n, :] = next_n_times

        obs = np.concatenate((last_n_array.flatten(), next_n_array.flatten(), np.array([self.steps / len(self.jobdata)])))
        return obs
    
    def _make_obs_classes(self):
        """
        Observation Space: [n_classes * (min_deadline, anzahl, aufwände_next)] + last_n_features + , step/n_products
        min_deadline: für die Klasse deadline des nächsten Produktes (Produkt mit der minimalen Deadline)
        anzahl: Anzahl Produkte in der Klasse die noch produziert werden müssen
        aufwände next: die Bearbeitungszeiten des nächsten Produkts in der Klasse
        fortschritt: step/n_products - Prozentueller Fortschritt
        last_n_features: für die letzten n Produkte welche Aufwände + Deadline verplant wurden
        """

        obs = np.zeros((self.n_classes, 14))  # 14 = 1 deadline + 1 Anzahl + 12 Aufwände next
        for cls in range(self.n_classes):
            class_products = self.jobclasses[cls]
            next_product = class_products[0]
            next_deadline = self.jobdata[next_product]['due_date']
            next_times = self.jobdata[next_product]['times']
            amount = len(class_products)
            row = np.array([next_deadline, amount] + next_times) #concat?
            obs[cls, :] = row
        
        # Extract times for the last self.last_n entries of planned_jobs
        last_n_array = np.ones((self.last_n, self.features_per_job)) * -1
        size_last_n = min((len(self.genome), self.last_n))
        last_n_times = np.array([self.jobdata[job]['times'] for job in self.genome[-size_last_n:]])
        last_n_array[-size_last_n:, :] = last_n_times
        last_n_array[-size_last_n:, :] /= self.conv_speed

        progress = np.array(self.steps / len(self.jobdata))

        obs_flat = np.concatenate([obs.flatten(), last_n_array.flatten(), progress])
        return obs_flat



class GraphEnv(gym.Env):
    def __init__(self, env_config: Dict):
        self._read_config(env_config)
        self.genome = []  # Permutation der Keys in Jobdata als np.array, z.B. [23039847988, 23039847987...]
        self.num_jobs = self.jpl
        self.steps = 0
        # Observation Size Large: Für die nächsten und letzten Aufträge jeweils die Arbeitszeiten + deadline und eine Distanzmatrix aufgeteilt in die Arbeitszeiten
        # observation_shape = (self.next_n + self.last_n) * self.features_per_job + self.next_n * self.next_n * (self.features_per_job - 1)
        # Observation Size Small: Pro next Job euclidische Distanz zum letztverplanten + Deadline + Distanzmatrix (see scipy.spatial.pdist)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.num_jobs, self.features_per_job))
        self.action_space = gym.spaces.Discrete(self.num_jobs)
        self.unplanned_jobs_sorted = []
        self.plan_horizon = 0
        self.selected_index = None
        self.obs_type = env_config['obs_space']
        self.deadline_gap = 0
        self.workload_gap = 0
        self.balance_punishement = 0
        self.curr_goal = 'workload' # Curriculum Learning 'workload' -> 'both'

    def reset(self, *, seed=None, options=None):
        self.genome = []
        self.steps = 0
        random_index = random.choice(list(range(0, len(self.dataset.data['Jobdata']))))
        self.selected_index = random_index
        self.jobdata = self.dataset.data['Jobdata'][random_index]
        self.job_ids = list(self.jobdata.keys())
        self.curr_target_values = self.dataset.data['GAFitness'][random_index]
        self.job_features = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']] for job in self.job_ids])
        obs = self.make_obs()
        info = self._get_info()
        self.invalid_action_penalty = 0
        return obs, info

    def step(self, action: int):
        self._add_job_to_genome(action)
        # print(f"self.genome: {self.genome}")
        reward = self._get_reward_sparse()
        obs = self.make_obs()
        terminated = False
        # info = self._get_info()
        if self.steps >= len(self.jobdata) - 1:
            terminated = True
        return obs, reward, terminated, False, {}

    def render(self):
        fig, axs = plt.subplots(6, 4)
        fig.tight_layout(h_pad=0.5)
        workload_data = np.array([self.jobdata[job_id]['times'] for job_id in self.genome])
        n_jobs = workload_data.shape[0]
        deadlines = np.array([self.jobdata[job_id]['due_date'] for job_id in self.genome])
        colors = map_to_rgb(deadlines)
        for idx, ax in enumerate(list(axs.flatten()[:12])):
            ax.bar(np.arange(0, n_jobs, 1), workload_data[:, idx], width=0.5, color=colors, alpha=0.8)
            ax.set_title(f'AP {idx} - RL')
            ax.set_xlabel('Produkte')
            ax.set_ylabel('MTM Zeit')
        
        ga_chromosome = self.dataset.data['GAChromosome'][self.selected_index]
        workload_data = np.array([self.jobdata[job_id]['times'] for job_id in ga_chromosome])
        deadlines = np.array([self.jobdata[job_id]['due_date'] for job_id in ga_chromosome])
        colors = map_to_rgb(deadlines)
        for idx, ax in enumerate(list(axs.flatten()[12:])):
            ax.bar(np.arange(0, n_jobs, 1), workload_data[:, idx], width=0.5, color=colors, alpha=0.8)
            ax.set_title(f'AP {idx} - GA')
            ax.set_xlabel('Produkte')
            ax.set_ylabel('MTM Zeit')
        
        plt.show()

    def close(self):
        pass

    def _read_config(self, env_config: Dict):
        # Lese config-Dict ein und speichere die Einträge als Instanzvariablen
        self.jpl = env_config["jpl"]  # Jobs per Line
        self.conv_speed = env_config["conv_speed"]  # Geschwindigkeit der Linie bzw. Zeit pro Aufgabe am Arbeitsplatz
        self.n_machines = env_config["n_machines"]  # Anzahl Maschinen pro Linie
        self.n_lines = env_config["n_lines"]  # Anzahl Linien
        self.window_size = env_config["window_size"]
        self.dataset = env_config["isri_dataset"]
        self.next_n = env_config["next_n"]  # Horizon, max Anzahl an Steps
        self.last_n = env_config["last_n"]
        self.features_per_job = env_config['input_features']

    def _get_reward_sparse(self):
        if self.steps == len(self.jobdata) - 1:
            diffsum, tardiness = fast_sim_diffsum(np.array(self.genome), jobs=self.jobdata, jpl=self.jpl,
                                                  conv_speed=self.conv_speed, n_machines=self.n_machines)
            diffsum_difference = (diffsum - self.curr_target_values[0]) / self.curr_target_values[0]
            tardiness_difference = (self.curr_target_values[1] - tardiness) / self.curr_target_values[1]
            # balance_reward = np.abs(diffsum_difference - tardiness_difference) # Belohnen wenn Ziele im selben Maß erreicht werden
            balance_reward = 0
            self.workload_gap = diffsum_difference
            self.deadline_gap = tardiness_difference
            self.balance_punishement = balance_reward
            if self.curr_goal == 'workload':
                return diffsum_difference * 100 - self.invalid_action_penalty
            elif self.curr_goal == 'both':
                return (diffsum_difference + tardiness_difference - balance_reward) * 100
        else:
            return -self.invalid_action_penalty
        
    def _get_reward_dense(self):
        if self.steps == len(self.jobdata) - 1:
            diffsum, tardiness = fast_sim_diffsum(np.array(self.genome), jobs=self.jobdata, jpl=self.jpl,
                                                  conv_speed=self.conv_speed, n_machines=self.n_machines)
            diffsum_difference = (diffsum - self.curr_target_values[0]) / self.curr_target_values[0]
            tardiness_difference = (self.curr_target_values[1] - tardiness) / self.curr_target_values[1]
            # balance_reward = np.abs(diffsum_difference - tardiness_difference) # Belohnen wenn Ziele im selben Maß erreicht werden
            balance_reward = 0
            self.workload_gap = diffsum_difference
            self.deadline_gap = tardiness_difference
            self.balance_punishement = balance_reward
            if self.curr_goal == 'workload':
                return diffsum_difference * 10
            elif self.curr_goal == 'both':
                return (diffsum_difference + tardiness_difference - balance_reward) * 100
        else:
            last_job_times = np.array(self.jobdata[self.genome[-2]]['times']) / self.conv_speed
            planned_job_times =np.array(self.jobdata[self.genome[-1]]['times']) / self.conv_speed
            diffsum = np.sum(np.abs(last_job_times - planned_job_times))
            return diffsum

    def _get_info(self):
        return None

    def _add_job_to_genome(self, action):
        try:
            selected_job = self.unplanned_jobs_sorted.pop(action)
            self.invalid_action_penalty = 0
        except IndexError:
            selected_job = self.unplanned_jobs_sorted.pop(len(self.unplanned_jobs_sorted) - 1) # Falls ungültige Aktion -> nimm den letzten
            self.invalid_action_penalty = 10
        self.genome.append(selected_job)
        self.steps += 1

    def action_masks(self):
        return np.array([idx < len(self.unplanned_jobs_sorted) for idx in range(self.next_n)])
    
    def get_obs_space(self, obs_space):
        if obs_space == 'simple':
            obs_shape = (1, (self.next_n + self.last_n) * (self.n_machines + 1) + 1) # Plus 1 für Step
            # obs_shape = (self.next_n + self.last_n, self.features_per_job)
        elif obs_space == 'full':
            obs_shape_len = (self.next_n + self.last_n) * self.features_per_job + self.next_n * self.next_n * (self.features_per_job - 1)
            obs_shape = (1, obs_shape_len)
        elif obs_space == 'small':
            obs_shape_len = int(self.next_n * 2 + ((self.next_n * (self.next_n - 1)) / 2))
            obs_shape = (1, obs_shape_len)
        else:
            raise ValueError(f"Unknown Observation type: {obs_space}")
        return obs_shape

    def make_obs(self):
        if self.obs_type == 'simple':
            return self._make_obs_simple()
        elif self.obs_type == 'full':
            return self._make_obs_full()
        elif self.obs_type == 'small':
            return self._make_obs_small()

    def _make_obs_full(self):
        # Extract times for the last self.last_n entries of planned_jobs
        last_n_array = np.ones((self.last_n, self.features_per_job)) * -10
        size_last_n = min((len(self.genome), self.last_n))
        last_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.genome[-size_last_n:]])
        last_n_array[-size_last_n:, :] = last_n_times
        # last_n_array /= self.conv_speed

        # next n jobs by due date
        next_n_array = np.ones((self.next_n, self.features_per_job)) * -10

        size_next_n = min((len(self.unplanned_jobs_sorted), self.next_n))
        next_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.unplanned_jobs_sorted[:size_next_n]])
        
        dist_matrix = np.zeros((self.next_n, self.next_n, (self.features_per_job-1))) # -1 weil das letzte Feature die Deadline ist

        if size_next_n > 0:
            # Difference instead of times
            next_n_times[:size_next_n, :self.n_machines] -= last_n_array[-1, :self.n_machines] * self.conv_speed
            # next_n_times[:size_next_n, -1] -= self.conv_speed * self.steps 
            next_n_array[:size_next_n, :] = next_n_times

            # Distance Matrix
            points_reshaped = next_n_times[:, :-1].reshape(size_next_n, 1, 12)
            points_transpose = next_n_times[:, :-1].reshape(1, size_next_n, 12)

            # Calculate the absolute difference for each pair of points
            distance_matrix = np.abs(points_reshaped - points_transpose)
            dist_matrix[:size_next_n, :size_next_n, :] = distance_matrix

        # next_n_array /= self.conv_speed
        obs = np.concatenate([last_n_array.flatten(), next_n_array.flatten(), dist_matrix.flatten()], axis=0)
        return obs
    
    def _make_obs_simple(self):
        # Extract times for the last self.last_n entries of planned_jobs
        last_n_array = np.ones((self.last_n, self.features_per_job)) * -1
        size_last_n = min((len(self.genome), self.last_n))
        last_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.genome[-size_last_n:]])
        last_n_array[-size_last_n:, :] = last_n_times

        # Normalisierung last n (MTM / Max Zeit, Deadline - vergangene Zeit / conv speed)
        last_n_array[-size_last_n:, :self.n_machines] /= self.conv_speed
        last_n_array[-size_last_n:, -1] = (last_n_array[-size_last_n:, -1] - self.conv_speed * self.steps) / self.conv_speed

        # next n jobs by due date
        next_n_array = np.ones((self.next_n, self.features_per_job)) * -1

        size_next_n = min((len(self.unplanned_jobs_sorted), self.next_n))
        next_n_times = np.array([self.jobdata[job]['times'] + [self.jobdata[job]['due_date']]
                                 for job in self.unplanned_jobs_sorted[:size_next_n]])

        if size_next_n > 0:
            # Next N Jobs Data ( MTM + Deadline)
            next_n_times[:, :self.n_machines] /= self.conv_speed
            next_n_times[:, -1] = (next_n_times[:, -1] - self.conv_speed * self.steps) / self.conv_speed
            next_n_array[:size_next_n, :] = next_n_times

        obs = np.concatenate((last_n_array.flatten(), next_n_array.flatten(), np.array([self.steps / len(self.jobdata)])))
        return obs
    
    def _make_obs_small(self):
        last_job_times = np.array(self.jobdata[self.genome[-1]]['times']) / self.conv_speed
        next_n_array = np.ones((self.next_n, 2)) * -1
        size_next_n = min((len(self.unplanned_jobs_sorted), self.next_n))
        dist_matrix_size = int((self.next_n * (self.next_n - 1)) / 2)
        dist_matrix_full = np.ones(dist_matrix_size) * -1 # see scipy.spatial.pdist

        if size_next_n > 0:
            next_n_times = np.array([self.jobdata[job]['times'] for job in self.unplanned_jobs_sorted[:size_next_n]])
            next_n_times /= self.conv_speed # Normalisierung
            workload_differences = cdist(last_job_times.reshape(1, self.n_machines), next_n_times, metric='minkowski', p=1)
            deadline_norm = np.array([self.jobdata[job]['due_date'] for job in self.unplanned_jobs_sorted[:size_next_n]])
            deadline_norm -= self.steps * self.conv_speed # Normalisierung
            dist_matrix = pdist(next_n_times, metric='minkowski', p=1)
            next_n_array[:size_next_n, 0] = workload_differences
            next_n_array[:size_next_n, 1] = deadline_norm
            dist_matrix_full[:dist_matrix.shape[0]] = dist_matrix


        obs = np.concatenate([next_n_array.flatten(), dist_matrix_full.flatten()], axis=0)
        return obs

