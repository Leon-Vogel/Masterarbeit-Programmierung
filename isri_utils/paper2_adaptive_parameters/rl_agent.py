import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import torch as th
from torch.nn import functional as F
import numpy as np
import gymnasium as gym
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from gymnasium.utils.env_checker import check_env
from gymnasium.spaces.multi_discrete import MultiDiscrete
from pymoo_utils.algorithm_factory import get_alg
from pymoo.core.variable import Real
from scipy.spatial.distance import cdist


def reward_function(algorithm, function_min=500, **kwargs):
    curr_fitness = algorithm.pop.get('F')
    curr_min = np.min(curr_fitness)
    # curr_mean = np.mean(curr_fitness)
    # advantage_min = last_min - curr_min
    # advantage_mean = last_mean - curr_mean

    # Verbesserung wird wichtiger je weiter die Episode fortgeschritten ist
    # advantage_min = advantage_min * algorithm.termination.perc
    # advantage_mean = advantage_mean * algorithm.termination.perc

    # In der letzten Episode gibt es eine Strafe für Funktionswert
    if not algorithm.has_next():
        return -(curr_min - function_min) / function_min
    # Sonst nur Belohnung für Verbesserung (Fokus auf )
    else:
        # return advantage_mean + advantage_min
        return 0

def linear_schedule(initial_value: float, decrease_factor: float = 0.95):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def get_val( var):
    if isinstance(var, Real):
        return var.value
    elif isinstance(var, float):
        return var
    elif isinstance(var, np.float32):
        return var


def set_val(var, val):
    if isinstance(var, Real):
        var.value = val
        return var
    elif isinstance(var, float):
        return val
    elif isinstance(var, np.float32):
        return val

def get_observation(algorithm, min_hist: list, mean_hist: list):
    """
    Berechnet die Observation sowohl in der Environment (unten) als auch in iteration Callback (pymoo.utils)
    """
    fitness_vals = np.array(list(algorithm.pop.get("F")))
    fitness_min = np.min(fitness_vals, axis=0)
    fitness_vals_norm = fitness_vals / fitness_min
    fitness_means = np.mean(fitness_vals_norm, axis=0)
    fitness_var = np.var(fitness_vals_norm, axis=0)

    n_iter = algorithm.n_iter
    n_eval = algorithm.evaluator.n_eval

    advantage_history = [1, 5, 10]
    mean_advantages = np.zeros(len(advantage_history))
    min_advantages = np.zeros(len(advantage_history))
    for idx, value in enumerate(advantage_history):
        if len(min_hist) > value:
            last_min = min_hist[-value]
            last_mean = mean_hist[-value]
            min_advantages[idx] = (last_min - fitness_min) / last_min
            mean_advantages[idx] = (last_mean - fitness_means) / last_mean
    
    min_hist.append(fitness_min)
    mean_hist.append(fitness_means)

    observation = np.hstack([fitness_means, fitness_var, mean_advantages,
                            min_advantages, n_iter,
                            get_val(algorithm.mating.mutation.prob),
                            get_val(algorithm.mating.crossover.prob),
                            n_eval]).flatten()
    np.nan_to_num(observation, copy=False, nan=0, posinf=0, neginf=0)
    observation = np.array(observation, dtype=np.float32) # WICHTIG!
    return observation

def decode_action(action):
    # CX Prob
    if action[0] == 0:
        crossover = 0.05
    elif action[0] == 1:
        crossover = 0.6
    elif action[0] == 2:
        crossover = 0.95
    
    # Mutation Prob
    if action[1] == 0:
        mutation = 0.05
    elif action[1] == 1:
        mutation = 0.4
    elif action[1] == 2:
        mutation = 0.6
    
    """
    # Pop Size
    if action[2] == 0:
        pop_size = 50
    elif action[2] == 1:
        pop_size = 100
    elif action[2] == 2:
        pop_size = 200
    """

    # Selection Alg
    if action[2] == 0:
        pressure = 2
    elif action[2] == 1:
        pressure = 4
    elif action[2] == 2:
        pressure = 10


    return crossover, mutation, pressure


class adaptation_env(gym.Env):
    def __init__(self, param, k=3):
        super(adaptation_env, self).__init__()
        self.param = param
        n_objs = len(param['obj_weights'])
        self.k = k # Adaption alle k steps
        self._function_min = None
        # Observation hängt von der Anzahl Objectives ab: Pro obj 5 Statistiken + 3 Allgemeine Werte (iter, curr_mut, curr_cx)
        self.observation_space = Box(np.ones(5*n_objs + 7) * -np.inf, np.ones(5*n_objs + 7) * np.inf)
        self.action_space = MultiDiscrete(np.array([3, 3, 3]))
        
        self._setup_algorithm(param)

        #Helper Variables
        self._min_hist = []
        self._mean_hist = []


    def step(self, action):
        # Decode Action
        crossover, mutation, pressure = decode_action(action)
                
        # Set Params accordingly
        self.algorithm.mating.crossover.prob = set_val(
            self.algorithm.mating.crossover.prob, crossover
        )
        self.algorithm.mating.mutation.prob = set_val(
            self.algorithm.mating.mutation.prob, mutation
        )
        """
        self.algorithm.pop_size = pop_size
        self.algorithm.n_offsprings = pop_size
        """

        self.algorithm.mating.selection.pressure = pressure

        # Neuer Alg Step
        for _ in range(self.k):
            self.algorithm.next()

            if self.algorithm.has_next():
                done = False
            else:
                done = True
                break

        # Make observation
        observation = self._get_obs()

        # Reward
        reward = self._get_reward()

        info = {
            'crossover': crossover,
            'mutation': mutation,
            "pressure": pressure}

        return observation, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_algorithm(self.param)
        self._min_hist = []
        self._mean_hist = []
        observation = self._get_obs()
        info = {}
        return observation, info
        
    def close(self):
        pass
    
    def _get_obs(self):
         # Get Observation
        observation = get_observation(self.algorithm, self._min_hist, self._mean_hist)
        return observation
    
    def _get_reward(self):
        # Get Reward
        return reward_function(self.algorithm, self._function_min)

    def _setup_algorithm(self, param, **kwargs):
        n_objs = len(param['obj_weights'])
        self.algorithm = get_alg('GA', n_objs, param['crossover'], param['mutation'], param['sampling'], False)
        probl = param['problem_getter'](param, None)
        min_value_idx = param['function'].index(probl.function)
        self._function_min = param['min_values'][min_value_idx]
        kwargs['termination'] = ('n_eval', param['n_evals'])
        self.algorithm.setup(probl, **kwargs)
        self.algorithm.pop_size = param["pop_size"]
        self.algorithm.n_offsprings = param['pop_size']
        self.algorithm.next()

if __name__ == '__main__':
    import time

    start_imports = time.time()
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.mutation.pm import PolynomialMutation
    from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
    from pymoo.algorithms.soo.nonconvex.ga import GeneticAlgorithm
    from gymnasium.utils.env_checker import check_env
    from cec_optimization_default import cecProblem
    from cec2017 import functions
    
    end_imports = time.time()
    import_time = end_imports - start_imports
    print('Import time: %.2f' %import_time)
    crossover = SimulatedBinaryCrossover(1)
    mutation = PolynomialMutation(1)
    sampling = FloatRandomSampling()
    # instances = [file for file in os.listdir(DATA_PATH) if file[:2] == 'ta' and int(file[3]) <= 2]
    # Test auf einer einzelnen Instanz
    # instances = [functions.f5, functions.f6, functions.f7, functions.f8, functions.f9, functions.f20, functions.f21, functions.f23]
    instances = [functions.f5]
    # min_values = [int(f.__name__[1:]) * 100 for f in instances]
    min_values = [500]
    problem_getter = cecProblem
    n_objs = 1

    N_ITERS = 2 # Anzahl zu lösender Probleme
    STEPS_PER_INST = 5 # Anzahl Training Epochen
    TRAIN_EVERY_INST = 3 # Anpassung der Netzwerkgewichte nach n Episoden
    SAVE_PATH = '/results/AdaptationTest2'
    POP_SIZE = 100
    N_EVALS = 6000
    RDM_SEED = 123
    PROBLEM = 'CEC'
    DIMS = 30 # Nur für CEC wichtig
    param = {
        'problem_getter': problem_getter,
        'alg_name': 'GA', # Siehe algorithm_factory.py
        'crossover': crossover, # Standardmethode des Frameworks
        'mutation': mutation, # Standardmethode des Frameworks
        'sampling': sampling, # Standardmethode des Frameworks
        'adaptive_rate_mode': 'none', # none | icdm (linear increasing crossover, decr. mutation) | imdc (incr. mutation, decr. crossover)
        'obj_weights': [1],
        'n_evals': N_EVALS, # Max. Anz. der Evaluationen (Stopping-Kriterium anstelle von max. Anz. Generationen)
        'parallelization_method': 'multi_threading', # multi_threading | multi_processing | dask
        'n_parallelizations': 4, # Anzahl paralleler Prozesse oder Threads
        'eliminate_duplicates': False, # Eliminiere Duplikate (redundante Individuen)
        'seed': RDM_SEED, # Starte mit Zufallszahl
        'print_results': False, # Endergebnis ausgeben
        'print_iterations': False, # Zwischenergebnis ausgeben je Generation/Iteration
        'plot_solution': False, # Diagramm ausgeben
        'pop_size': POP_SIZE,
        'function': instances,
        'min_values': min_values,
        'dimensions': DIMS, # Funktionen sind für verschiedene Eingangsraumgrößen definiert
        'xl': np.ones(DIMS) * -100, #Upper and Lower Limit für Eingänge
        'xu': np.ones(DIMS) * 100
        }

    done = False
    t_start = time.time()
    test_env = adaptation_env(param)
    t_end = time.time()
    t = t_end - t_start
    print("Time for initialization: %.2f" % t)
    check_env(test_env)

    t_start_reset = time.time()
    test_env.reset()
    t_end_reset = time.time()
    reset_time = t_end_reset - t_start_reset
    print("Time for reset %.2f" %reset_time)
    action = np.array([1, 1, 1, 1])
    steps = 0
    t_start_opt = time.time()
    while not done:
        observation, reward, done, _, info = test_env.step(action)
        steps += 1
    t_end_opt = time.time()
    opt_time = t_end_opt - t_start_opt
    print("Time for optimization: %.2f" % opt_time)
    print("Steps: %.2f" %steps)
    print('finished')
