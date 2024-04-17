from ray import tune
from permutation_flow_shop_default import PFSSP
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from pymoo_utils.algorithm_runner import run_alg, run_alg_parallel
from pymoo.operators.crossover.pntx import SinglePointCrossover, TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from custom_ga_operators import JobOrderCrossoverByKendallDistance, FlipMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling

def objective(config):
    config['mutation'] = config['mutation'](config['mut_pb'])
    config['crossover'] = config['crossover'](prob=config['cx_pb'])
    result, best_schedule, makespan_flowtime = run_alg_parallel(run_alg, config)
    return makespan_flowtime

search_space = {
    "crossover": tune.choice([SinglePointCrossover, TwoPointCrossover, UniformCrossover]),
    "cx_pb" : tune.choice([0.05, 0.6, 0.95]),
    "mut_pb": tune.choice([0.05, 0.4, 0.6]),
    # 'crossover': SinglePointCrossover(),
    'problem_getter': PFSSP,
    'alg_name': 'GA', # Siehe algorithm_factory.py
    'mutation': FlipMutation, # Standardmethode des Frameworks
    'sampling': PermutationRandomSampling(), # Standardmethode des Frameworks
    'adaptive_rate_mode': 'none', # none | icdm (linear increasing crossover, decr. mutation) | imdc (incr. mutation, decr. crossover)
    'obj_weights': [1],
    'n_evals': 3000, # Max. Anz. der Evaluationen (Stopping-Kriterium anstelle von max. Anz. Generationen)
    'parallelization_method': 'multi_threading', # multi_threading | multi_processing | dask
    'n_parallelizations': 4, # Anzahl paralleler Prozesse oder Threads
    'eliminate_duplicates': False, # Eliminiere Duplikate (redundante Individuen)
    'seed': None, # Starte mit Zufallszahl
    'print_results': False, # Endergebnis ausgeben
    'print_iterations': False, # Zwischenergebnis ausgeben je Generation/Iteration
    'plot_solution': False, # Diagramm ausgeben
    'pop_size': tune.choice([10, 20, 50]),
    'pressure': tune.choice([2, 4, 10]),
    'functions': ['ta033']
}

tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=4**3 * 30)
)

results = tuner.fit()
df = results.get_dataframe()
df.to_csv('isri_utils/paper2_adaptive_parameters/HyperparameterGAOpt_with_pressure.csv', sep=";")



