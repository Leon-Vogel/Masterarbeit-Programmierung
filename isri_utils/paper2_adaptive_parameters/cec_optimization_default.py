import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pymoo.core.problem import ElementwiseEvaluationFunction, LoopedElementwiseEvaluation, Problem
from pymoo_utils.algorithm_runner import run_alg, run_alg_parallel
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.core.selection import Selection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.survival import Survival
from pymoo.core.population import Population
from multiprocessing import Manager
# from isri_utils.paper2_adaptive_parameters.custom_ga_operators import JobOrderCrossoverByKendallDistance, SwapMultipleJobsMutationByLogarithm
import math
import random as rnd
# from misc_utils.gantt_plotter_matplotlib import plot_gantt
import numpy as np
import cec2017.functions as functions


class cecProblem(Problem):
    def __init__(self, param, runner):
        super().__init__(n_var=param['dimensions'], n_obj=1, elementwise_evaluation=False)
        self.dims = param['dimensions']
        self.function = np.random.choice(param['function'], 1)[0]
        self.xl = param['xl']
        self.xu = param['xu']

    def _evaluate(self, x, out, *args, **kwargs):
        y = self.function(x)
        out['F'] = y


class FitnessProportionateSurvival(Survival):
    def __init__(self, filter_infeasible=False):
        super().__init__(filter_infeasible)
    
    def _do(self, problem, pop, n_survive=None, **kwargs):
        F = pop.get('F')
        rel_F = F / np.max(F)
        prob = 1 - rel_F
        new_pop = 0


if __name__ == '__main__':
    
    dims = 30
    param = {
        'problem_getter': cecProblem,
        'alg_name': 'GA', # Siehe algorithm_factory.py
        'crossover': SimulatedBinaryCrossover(0.95),
        'mutation': PolynomialMutation(0.05),
        'sampling': FloatRandomSampling(), # Standardmethode des Frameworks
        'adaptive_rate_mode': 'none', # none | icdm (linear increasing crossover, decr. mutation) | imdc (incr. mutation, decr. crossover)
        'obj_weights': [1],
        'n_evals': 10000, # Max. Anz. der Evaluationen (Stopping-Kriterium anstelle von max. Anz. Generationen)
        'parallel_computing': 'multi_processing', # multi_threading | multi_processing | dask
        'n_parallel_worker': 4, # Anzahl paralleler Prozesse oder Threads
        'eliminate_duplicates': False, # Eliminiere Duplikate (redundante Individuen)
        'seed': None, # Starte mit Zufallszahl
        'print_results': True, # Endergebnis ausgeben
        'print_iterations': True, # Zwischenergebnis ausgeben je Generation/Iteration
        'plot_solution': False, # Diagramm ausgeben
        'pop_size': 50,
        'function': functions.f5,
        'dimensions': dims, # Funktionen sind für verschiedene Eingangsraumgrößen definiert
        'xl': np.ones(dims) * -100, #Upper and Lower Limit für Eingänge
        'xu': np.ones(dims) * 100
    }
    run_alg_parallel(run_alg, param)
