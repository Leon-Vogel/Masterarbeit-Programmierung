from permutation_ga_env import PermutationGA
from pymoo.operators.crossover.pntx import SinglePointCrossover, TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from custom_ga_operators import JobOrderCrossoverByKendallDistance, FlipMutation
from pymoo.core.problem import LoopedElementwiseEvaluation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from permutation_flow_shop_default import PFSSP
import read_taillard_data as data_reader

def get_permutation_problem(type='pfssp', instances=None):
    """
    Definition der Trainingsprobleme mit allen Parametern
    """
    if type == 'pfssp':
        # Register Environment
        crossover = TwoPointCrossover()
        mutation = FlipMutation(0.01)
        sampling = PermutationRandomSampling()
        # Instanzen für Training hier definiert. Zum Test kann eine einzelne instanz übergeben werden
        if instances is None:
            instances = ['ta021', 'ta022', 'ta023', 'ta031', 'ta032', 'ta033']

        problem_getter = PFSSP
        n_objs = 1

        param = {
            'problem_getter': problem_getter,
            'alg_name': 'GA', # Siehe algorithm_factory.py
            'crossover': crossover, # Standardmethode des Frameworks
            'mutation': mutation, # Standardmethode des Frameworks
            'sampling': sampling, # Standardmethode des Frameworks
            'adaptive_rate_mode': 'none', # none | icdm (linear increasing crossover, decr. mutation) | imdc (incr. mutation, decr. crossover)
            'obj_weights': [1],
            'n_evals': 3000, # Max. Anz. der Evaluationen (Stopping-Kriterium anstelle von max. Anz. Generationen)
            'parallelization_method': 'none', # multi_threading | multi_processing | dask
            'n_parallelizations': 4, # Anzahl paralleler Prozesse oder Threads
            'eliminate_duplicates': False, # Eliminiere Duplikate (redundante Individuen)
            'seed': 134, # Starte mit Zufallszahl
            'print_results': False, # Endergebnis ausgeben
            'print_iterations': False, # Zwischenergebnis ausgeben je Generation/Iteration
            'plot_solution': False, # Diagramm ausgeben
            'pop_size': 100,
            'functions': instances
            }
        
        environment = PermutationGA(param)
        return environment
        
