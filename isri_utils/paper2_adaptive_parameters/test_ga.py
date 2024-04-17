from cec_optimization_default import cecProblem
from cec2017 import functions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from custom_ga_operators import JobOrderCrossoverByKendallDistance, SwapMultipleJobsMutationByLogarithm
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from custom_ga_operators import FlipMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from permutation_flow_shop_default import PFSSP

import numpy as np
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pymoo_utils.algorithm_runner import run_alg, run_alg_parallel
from ray.rllib.policy.policy import Policy
import time

N_TRIES = 100
RDM_SEED = 123
ADAPTATION_MODES = ['none', 'imdc', 'icdm']
SAVE_FILE = '/isri_utils/paper2_adaptive_parameters/results/PFSSP_Results_GA.csv'
PROBLEM = 'pfssp'
#f9, f10, f24, f27
INSTANCES = ['ta033', 'ta034', 'ta041']
AGENT_PATH = "Y:/Documents/SUPPORT_AdaptiveHyperparameter/project_support/src/executable/isri_utils/paper2_adaptive_parameters/results/Trained_on_multiple/trial_57a2b_00001/checkpoint_000074/policies/default_policy/"

if __name__ == '__main__':
    experiment_results = {'AdaptationMode': [],
                          'Problem': [],
                          'Instance': [],
                          'TryNr': [],
                          'RDMSeed': [],
                          'MinValue': []
                          }

    if PROBLEM == 'pfssp':
        crossover = UniformCrossover()
        mutation = FlipMutation(0.4)
        sampling = PermutationRandomSampling()
        instances = ['ta033']
        problem_getter = PFSSP
        n_objs = 2
        POP_SIZE=50
        DIMS = None
        N_EVALS = 3000

        opt_params = {
            'ta033': {
                "pop_size": 50,
                "pressure": 2,
                'crossover': UniformCrossover(),
                'mutation': FlipMutation(0.6)
            },
            'ta034': {
                'pop_size': 50,
                'pressure': 10,
                'crossover': UniformCrossover(),
                'mutation': FlipMutation(0.05)
            }
        }

    elif PROBLEM == 'CEC':
        crossover = SimulatedBinaryCrossover()
        mutation = PolynomialMutation(1)
        sampling = FloatRandomSampling()
        # instances = [file for file in os.listdir(DATA_PATH) if file[:2] == 'ta' and int(file[3]) <= 2]
        # Test auf einer einzelnen Instanz
        problem_getter = cecProblem
        n_objs = 1

    
    param = {
            'problem_getter': problem_getter,
            'alg_name': 'GA', # Siehe algorithm_factory.py
            'crossover': crossover, # Standardmethode des Frameworks
            'mutation': mutation, # Standardmethode des Frameworks
            'sampling': sampling, # Standardmethode des Frameworks
            'adaptive_rate_mode': 'none', # none | icdm (linear increasing crossover, decr. mutation) | imdc (incr. mutation, decr. crossover)
            'obj_weights': [1],
            'n_evals': N_EVALS, # Max. Anz. der Evaluationen (Stopping-Kriterium anstelle von max. Anz. Generationen)
            'parallelization_method': 'multi_processing', # multi_threading | multi_processing | dask
            'n_parallelizations': 4, # Anzahl paralleler Prozesse oder Threads
            'eliminate_duplicates': False, # Eliminiere Duplikate (redundante Individuen)
            'seed': RDM_SEED, # Starte mit Zufallszahl
            'print_results': False, # Endergebnis ausgeben
            'print_iterations': False, # Zwischenergebnis ausgeben je Generation/Iteration
            'plot_solution': False, # Diagramm ausgeben
            'pop_size': POP_SIZE,
            'functions': INSTANCES,
            'dimensions': DIMS, # Funktionen sind für verschiedene Eingangsraumgrößen definiert
            'xl': np.ones(DIMS) * -100, #Upper and Lower Limit für Eingänge
            'xu': np.ones(DIMS) * 100,
            'pressure': 4
        }

    for instance in INSTANCES:
        param['functions'] = [instance]
        if instance in opt_params.keys():
            for k, v in opt_params.items():
                param[k] = v
        for adapt_mode in ADAPTATION_MODES:
            param['adaptive_rate_mode'] = adapt_mode
            for n in range(N_TRIES):
                t_start = time.process_time()
                RDM_SEED += 1
                param['seed'] = RDM_SEED
                result = run_alg_parallel(run_alg, param)
                min_value = result[-1]
                if isinstance(min_value, np.ndarray):
                    min_value = min_value[0]
                if isinstance(min_value, list):
                    min_values = min_value[0]
                experiment_results['AdaptationMode'].append(adapt_mode)
                experiment_results['Problem'].append(PROBLEM)
                experiment_results['Instance'].append(result[0].problem.function)
                experiment_results['TryNr'].append(n)
                experiment_results['RDMSeed'].append(RDM_SEED)
                experiment_results['MinValue'].append(min_value)
                t_end = time.process_time()
                t = t_end - t_start
                print(f"Iteration {n} with {adapt_mode}, duration: %.2f" %t, '|', min_value)

    df = pd.DataFrame(experiment_results)
    save_path = sys.path[-1] + SAVE_FILE
    if os.path.isfile(save_path):
        df_old = pd.read_csv(save_path, sep=";")
        df_new = pd.concat([df_old, df], axis=0, ignore_index=True)
        df_new.to_csv(save_path, sep=";")
    else:
        df.to_csv(save_path, sep=";")

