from _dir_init import *
import gymnasium as gym
import numpy as np
from isri_optimizer.isri_metaheuristic_simple import fast_sim_diffsum, find_next_jobs_by_due_date
from typing import Dict
from misc_utils.copy_helper import fast_deepcopy
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from clustering import cluster_kmeans, cluster_neighbour

LOGPATH = './isri_optimizer/rl_sequential_agent/test_log.txt'


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
        self.action_space = gym.spaces.Discrete(self.n_classes)
        self.unplanned_jobs_sorted = []
        self.plan_horizon = 0
        self.selected_index = None
        self.obs_type = env_config['obs_space']
        self.deadline_gap = 0
        self.workload_gap = 0
        self.deadline_r = 0
        self.diffsum_r = 0
        #self.reward_log = -100000
        self.balance_punishement = 0
        self.jobclasses = {idx: [] for idx in range(self.n_classes)}
        if self.cluster_method == "kmeans":
            self.cluster = cluster_kmeans(self.n_classes)
        if self.cluster_method == "neighbour":
            self.cluster = cluster_neighbour(self.n_classes)
        


    def reset(self, *, seed=None, options=None):
        self.genome = []
        self.steps = 0
        # self.reward_log = -100000
        random_index = random.choice(list(range(0, len(self.dataset.data['Jobdata']))))
        self.selected_index = random_index
        self.jobdata = self.dataset.data['Jobdata'][random_index]
        for job in self.jobdata:
            times = self.jobdata[job]['times']
            cluster = int(self.cluster.label(times))
            self.jobclasses[cluster].append(job)

        for cluster in range(self.n_classes):
            cluster_jobs = self.jobclasses[cluster]
            sorted_products = sorted(cluster_jobs, key=lambda x: self.jobdata[x]['due_date'])
            self.jobclasses[cluster] = sorted_products

        self.curr_target_values = self.dataset.data['GAFitness'][random_index]
        self.unplanned_jobs_sorted = sorted(self.jobdata, key=lambda x: self.jobdata[x]['due_date'])
        self.plan_horizon = len(self.jobdata) * self.conv_speed + self.n_machines * self.conv_speed
        #first_job = self.unplanned_jobs_sorted.pop(0)
        #self.genome.append(first_job)
        obs = self.make_obs()
        info = self._get_info()
        self.invalid_action_penalty = 0
        return obs, info

    def step(self, action: int):
        self._add_job_to_genome(action)
        #self.log_to_file(LOGPATH, str(action))
        #self._add_job_to_genome_heurist(action)
        # print(f"self.genome: {self.genome}")
        reward = self._get_reward()
        obs = self.make_obs()
        #self.reward_log += reward
        #self.log_to_file(LOGPATH, str(obs))
        #self.log_to_file(LOGPATH, f'Reward: {reward}')
        terminated = False
        # info = self._get_info()
        if self.steps >= len(self.jobdata):
            terminated = True
            
            #self.log_to_file(LOGPATH, 'New Episode')
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
        self.DIFFSUM_NORM = env_config["DIFFSUM_NORM"] #Was ist das Norm?
        self.tardiness_weight = env_config["tardiness_weight"]
        self.TARDINESS_NORM = env_config["TARDINESS_NORM"]
        self.pca = env_config['pca']
        self.n_classes = env_config['n_classes']
        self.cluster_method = env_config['cluster_method']
        self.use_heuristic = True
        self.reward_type = env_config['reward_type']

    def _project_fitnessscores_to_reward(self, diffsum: float, tardiness: float):
        # Wir wollen diffsum_weight maximieren und tardiness minimieren
        diffsum_reward = diffsum * self.DIFFSUM_NORM * self.DIFFSUM_WEIGHT
        tardiness_reward = -tardiness * self.TARDINESS_NORM * self.TARDINESS_WEIGHT
        r = diffsum_reward + tardiness_reward
        return r

    @staticmethod
    def log_to_file(path, message):#
        if os.path.isfile(path):
            mode = 'a'
        else:
            mode = 'w'
        with open(path, mode) as f:
            f.write(message)
            f.write("\n")
        
    def _get_reward(self):
        if self.reward_type == 'sparse':
            return self._get_reward_sparse()
        elif self.reward_type == 'sparse_sum':
            return self._get_reward_sparse_sum()
        elif self.reward_type == 'dense':
            return self._get_reward_dense()
        elif self.reward_type == 'combined':
            return (self._get_reward_dense()*0.1)+self._get_reward_sparse()

    def _get_reward_sparse(self):
        if self.steps == len(self.jobdata): # - 1
            diffsum, tardiness = fast_sim_diffsum(np.array(self.genome), jobs=self.jobdata, jpl=self.jpl,
                                                  conv_speed=self.conv_speed, n_machines=self.n_machines)
            diffsum_difference = (diffsum - self.curr_target_values[0]) / self.curr_target_values[0]
            tardiness_difference = (self.curr_target_values[1] - tardiness) / self.curr_target_values[1]
            # balance_reward = np.abs(diffsum_difference - tardiness_difference) # Belohnen wenn Ziele im selben Maß erreicht werden
            self.workload_gap = diffsum_difference            
            self.deadline_gap = tardiness_difference
            #return (diffsum_difference + tardiness_difference - balance_reward) * 100
            reward = (diffsum_difference + tardiness_difference) * 1
            reward = float(reward)
            return reward
        else:
            return 0
        
        
    def _get_reward_sparse_sum(self):
        if self.steps == len(self.jobdata): # 
            diffsum, tardiness = fast_sim_diffsum(np.array(self.genome), jobs=self.jobdata, jpl=self.jpl,
                                                  conv_speed=self.conv_speed, n_machines=self.n_machines)
            diffsum_difference = (diffsum - self.curr_target_values[0]) / self.curr_target_values[0]
            tardiness_difference = (self.curr_target_values[1] - tardiness) / self.curr_target_values[1]
            # balance_reward = np.abs(diffsum_difference - tardiness_difference) # Belohnen wenn Ziele im selben Maß erreicht werden
            balance_reward = 0
            self.workload_gap = diffsum_difference            
            self.deadline_gap = tardiness_difference
            self.balance_punishement = balance_reward
            #if tardiness > 0:
            #    tardiness = -tardiness
            #else:
            #    tardiness = 0
            #return (diffsum_difference + tardiness_difference - balance_reward) * 100
            
            self.deadline_r += (-tardiness)/20
            self.diffsum_r += (-diffsum)/30000
            
            reward = self.diffsum_r + self.deadline_r
            reward = float(reward)
            return reward
        else:
            return 0
        
    def _get_reward_dense(self):
        # Letzte Diffsum Änderung
        if self.steps > 1:
            last_job_times = np.array(self.jobdata[self.genome[-2]]['times']) / self.conv_speed
            planned_job_times =np.array(self.jobdata[self.genome[-1]]['times']) / self.conv_speed
            diffsum = np.sum(np.abs(last_job_times - planned_job_times))

            # Tardiness Änderung
            last_job_deadline = self.jobdata[self.genome[-1]]['due_date']
            current_finish_time = (self.steps + self.n_machines) * self.conv_speed
            #tardiness = -np.exp((current_finish_time - last_job_deadline) / 3600) # 3600 Sekunden = 1 Stunde
            tardiness = ((current_finish_time - last_job_deadline) / 3600) # 3600 Sekunden = 1 Stunde
            if tardiness > 0:
                tardiness = -tardiness
            else:
                tardiness = 0# -tardiness*0.1

            self.deadline_r += tardiness * self.tardiness_weight
            self.diffsum_r += diffsum * self.diffsum_weight
            if self.steps == len(self.jobdata):
                diffsum_temp, tardiness_temp = fast_sim_diffsum(np.array(self.genome), jobs=self.jobdata, jpl=self.jpl,
                                                  conv_speed=self.conv_speed, n_machines=self.n_machines)
                diffsum_difference = (diffsum_temp - self.curr_target_values[0]) / self.curr_target_values[0]
                tardiness_difference = (self.curr_target_values[1] - tardiness_temp) / self.curr_target_values[1]
                # balance_reward = np.abs(diffsum_difference - tardiness_difference) # Belohnen wenn Ziele im selben Maß erreicht werden
                balance_reward = 0
                self.workload_gap = diffsum_difference            
                self.deadline_gap = tardiness_difference
            try:
                reward = tardiness * self.tardiness_weight + diffsum * self.diffsum_weight 
                reward = float(reward)
            except TypeError:
                print('Debug')
            return reward
        else:
            return 0
    
    def _get_info(self):
        return None

    def _add_job_to_genome(self, action):
        selected_job = self.jobclasses[action].pop(0)
        self.unplanned_jobs_sorted.remove(selected_job)
        self.genome.append(selected_job)
        self.steps += 1

    def _add_job_to_genome_heurist(self, action):
        try:
            selected_job = self.jobclasses[action].pop(0)
            self.unplanned_jobs_sorted.remove(selected_job)
            self.invalid_action_penalty = 0
        except IndexError:
            selected_job = self.unplanned_jobs_sorted.pop(len(self.unplanned_jobs_sorted) - 1) # Falls ungültige Aktion -> nimm den letzten
            self.invalid_action_penalty = 10
        self._update_genome(selected_job)
        self.steps += 1

    def _update_genome(self, job):
        if self.last_n > len(self.genome):
            options = list(range(len(self.genome) + 1))
        else:
            options = list(range(len(self.genome) - self.last_n, len(self.genome) + 1))
        
        if self.use_heuristic:
            # Platziere den Job an der Stelle mit maximaler Workload difference innerhalb der last_n Stellen
            results = []
            for option in options:
                new_ind = fast_deepcopy(self.genome)
                new_ind.insert(option, job)
                new_ind = np.array(new_ind)
                diffsum, tardiness = fast_sim_diffsum(new_ind, self.jobdata, self.jpl, self.conv_speed,
                                                    self.n_machines, n_lines=1, window_size=self.window_size)
                results.append(diffsum)
            best_diffsum = np.argmin(results)
            self.genome.insert(options[best_diffsum], job)

    def action_masks(self):
        return np.array([len(self.jobclasses[idx]) > 0 for idx in range(self.n_classes)])
        # return np.array([idx < len(self.unplanned_jobs_sorted) for idx in range(self.next_n)])
    
    def get_obs_space(self, obs_space):
        if obs_space == 'classes':
            obs_shape = (1, self.n_classes * 14 + self.last_n * 13 + 1) # Plus 1 für Step
            # obs_shape = (self.next_n + self.last_n, self.features_per_job)
        elif obs_space == 'simple':
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
        if self.obs_type == 'classes':
            return self._make_obs_classes()
        elif self.obs_type == 'simple':
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
        #Deadline etc. für leere Klassen ist -1, 
        obs = np.zeros((self.n_classes, 14))#*-1 # 14 = 1 deadline + 1 Anzahl + 12 Aufwände next
        for cls in range(self.n_classes):
            class_products = self.jobclasses[cls]
            if len(class_products) > 0:
                next_product = class_products[0]
                next_deadline = self.jobdata[next_product]['due_date']
                next_deadline = (next_deadline - self.conv_speed * self.steps ) / (self.conv_speed*100) #Next deadline ändern und leere klassen wieder auf 0?
                next_times = self.jobdata[next_product]['times']
                
                amount = len(class_products)/len(self.jobdata)
                row = np.array([next_deadline, amount] + next_times) #concat?
                row[2:] = row[2:]/self.conv_speed
                obs[cls, :] = row
        
        # Extract times for the last self.last_n entries of planned_jobs
        last_n_array = np.zeros((self.last_n, self.features_per_job))# * -1
        size_last_n = min((len(self.genome), self.last_n))
        if size_last_n > 0:
            last_n_times = np.array([self.jobdata[job]['times']+[self.jobdata[job]['due_date']] for job in self.genome[-size_last_n:]])
            last_n_array[-size_last_n:, :] = last_n_times

            last_n_array[-size_last_n:, :-1] /= self.conv_speed #Normalisierung mit conveyer speed

            last_n_array[-size_last_n:, -1] = (last_n_array[-size_last_n:, -1] - self.conv_speed * self.steps) / (self.conv_speed*100)

            #Verspätung zu den letzten Jobs hinzufügen

        progress = np.array(self.steps / len(self.jobdata)).reshape([1,])

        obs_flat = np.concatenate([obs.flatten(), last_n_array.flatten(), progress])
        return obs_flat



