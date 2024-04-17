from _dir_init import *


import multiprocessing
from multiprocessing.pool import ThreadPool

import numpy as np
from dask.distributed import Client
from pymoo.core.problem import (DaskParallelization,
                                LoopedElementwiseEvaluation,
                                StarmapParallelization)
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize
from pymoo_utils.algorithm_factory import get_alg
from pymoo_utils.iteration_callback import EachIterationCallback


def run_alg_parallel(run_alg_caller, param):
    if 'parallelization_method' not in param or param['parallelization_method'].lower() == 'none' or param['parallelization_method'] == None:
        runner = LoopedElementwiseEvaluation()
        param['parallelization_method'] = None
        param['n_parallelizations'] = 1
        return run_alg_caller(param, runner)
    elif param['parallelization_method'] == 'dask':
        with Client() as client:
            client.restart()
            return run_alg_caller(param, DaskParallelization(client))
    elif param['parallelization_method'] == 'multi_threading':
        with ThreadPool(param['n_parallelizations']) as pool:
            return run_alg_caller(param, StarmapParallelization(pool.starmap))
    elif param['parallelization_method'] == 'multi_processing':
        with multiprocessing.Pool(param['n_parallelizations']) as pool:
            return run_alg_caller(param, StarmapParallelization(pool.starmap))
    else:
        raise Exception(f'{param["parallelization_method"]} is not a valid parallelization method')


def run_alg(param, runner=None):
    # instantiate algorithm and problem to optimize
    probl = param['problem_getter'](param,runner)
    if 'pressure' in param.keys():
        pressure = param['pressure']
    else:
        pressure = 2
    alg = get_alg(param['alg_name'], n_objectives=len(param['obj_weights']), 
                    crossover=param['crossover'], mutation=param['mutation'], 
                    sampling=param['sampling'], duplicate_elimination=param['eliminate_duplicates'],
                    pressure=pressure) # TODO: hyperparameter tuning
    
    if param['pop_size'] and param['pop_size'] != None and alg.pop_size:
        alg.pop_size = param['pop_size']
        if alg.n_offsprings:
            alg.n_offsprings = param['pop_size']

    res = minimize(probl,
                    alg,
                    ('n_eval', param['n_evals']), # use max evals as stopping criterion
                    seed=param['seed'],
                    callback=EachIterationCallback(param['n_evals'], 
                                                    adaptive_rate_mode=param['adaptive_rate_mode'], # adjust mutation/crossover rates
                                                    print_iteration_stats=param['print_iterations'],
                                                    agent=param['agent'] if 'agent' in param else None,
                                                    save_stats=param['save_path_stats'] if 'save_path_stats' in param else None),
                    verbose=False, # print additional stats
                )

    # normalize results and select best by weights
    F_normalized = (res.F - res.F.min(axis=0)) / (res.F.max(axis=0) - res.F.min(axis=0))
    i_best = ASF().do(F_normalized, 1/np.array(param['obj_weights'])).argmin()
    
    if param['print_results']:
        print(f"\n{alg}\n=================")
        print("Best solution found: ", res.X[i_best])
        print("Objective value: ", res.F[i_best])
        print(f"Comp. time: {res.exec_time} s")
    
    if param['plot_solution']:
        probl.plot_solution(res.X[i_best])

    if 'final_processing_func' in param:
        param['final_processing_func'](probl, alg, param)

    return res, res.X[i_best], res.F[i_best]
