import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import path_definitions

from pymoo.core.problem import ElementwiseProblem
from pymoo_utils.algorithm_runner import run_alg, run_alg_parallel
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.crossover.ox import OrderCrossover
from multiprocessing import Manager
from isri_utils.paper2_adaptive_parameters.custom_ga_operators import JobOrderCrossoverByKendallDistance, SwapMultipleJobsMutationByLogarithm
from pymoo.operators.crossover.pntx import TwoPointCrossover
import math
import random as rnd
from misc_utils.gantt_plotter_matplotlib import plot_gantt
import numpy as np
# import numba
import isri_utils.paper2_adaptive_parameters.read_taillard_data as data_reader


def get_predecessors(job_assignments: list, machine_idx: int):
    result = []
    for j in job_assignments:
        if j['resource'] <= machine_idx:
            result.append(j)
    return result


def get_start_time(predecessors):
    result = []
    for p in predecessors:
        result.append(p['end_time'])
    if result:
        return max(result)
    else:
        return 0
    
def get_job_end_times(operations_on_last_machine: list):
    result = []
    for job in operations_on_last_machine:
        result.append(job['end_time'])
    return result


def generate_schedule(ind, n_machines, processing_times): # Übergeben wird ein Individuum für das PFSSP (Permutierung der einzulastenden Jobs), z.B. J1, J2, J4, J3
    # Baue den Schedule Stück für Stück auf. Verplane dabei jeden Job auf jeder Maschine.
    job_machine_assignments = []
    for job_idx in ind:
        for machine_idx in range(n_machines):
            # Extrahiere alle potenziellen Vorgängeroperationen
            predecessors = get_predecessors(job_machine_assignments, machine_idx)
            # Extrahiere aus diesen wiederrum den letzten End-Termin, welcher gleichzeitig der Starttermin des aktuellen Jobs ist
            start_time = get_start_time(predecessors)
            # Erzeuge den Eintrag für den Schedule
            job_machine_assignments.append({
                    'start_time': start_time, 
                    'end_time': start_time + processing_times[job_idx][machine_idx], 
                    'job': job_idx, 
                    'resource': machine_idx
                })
            
    # Gebe das Ergebnis (Start- und Endzeitpunkt je zugewiesenen Job) zurück, um z.B. ein Gantt-Diagramm zu erzeugen
    return job_machine_assignments

def generate_schedule_arr(ind, n_machines, processing_times):
    times = processing_times[ind]
    schedule = np.zeros([processing_times.shape[0], n_machines, 2])
    for machine_idx in range(n_machines):
        for job_idx in range(times.shape[0]):
            # Erster Job auf erster Maschine:
            if machine_idx == 0 and job_idx == 0:
                schedule[job_idx, machine_idx, 1] = times[job_idx, machine_idx]
            # Auf der ersten Maschine warten die Jobs nur auf den Vorgänger
            elif machine_idx == 0 and job_idx != 0:
                schedule[job_idx, machine_idx, 0] = schedule[job_idx -1, machine_idx, 1]
                schedule[job_idx, machine_idx, 1] = schedule[job_idx -1, machine_idx, 1] + times[job_idx, machine_idx]
            # der erste Job fängt auf allen Maschinen sofort an sobald auf M-1 fertig
            elif machine_idx != 0 and job_idx == 0:
                schedule[job_idx, machine_idx, 0] = schedule[job_idx, machine_idx - 1, 1]
                schedule[job_idx, machine_idx, 1] = schedule[job_idx, machine_idx - 1, 1] + times[job_idx, machine_idx]
            # alle anderen Jobs warten entweder auf den Vorgänger auf der Maschine oder eigene Vorgängeroperation auf vorheriger Maschine
            elif machine_idx != 0 and job_idx != 0:
                start = max(schedule[job_idx, machine_idx - 1, 1], schedule[job_idx - 1, machine_idx, 1])
                schedule[job_idx, machine_idx, 0] = start
                schedule[job_idx, machine_idx, 1] = start + times[job_idx, machine_idx]
    return schedule

def get_job_end_times_arr(schedule):
    return schedule[:, -1, 1]

def get_makespan(schedule):
    return float(np.max(schedule))


class PFSSP(ElementwiseProblem):
    def __init__(self, param, runner):
        rdm_idx = np.random.choice(range(len(param['functions'])), 1)[0]
        self.function = param['functions'][rdm_idx]
        instance = data_reader.get_taillard(param['functions'][rdm_idx])
        super().__init__(n_var=len(instance), n_obj=1, elementwise_evaluation=True, elementwise_runner=runner)
        self.n_machines = len(instance[0])
        self.processing_times = np.array(instance)

    def _evaluate(self, x, out, *args, **kwargs):
        schedule_info = generate_schedule_arr(x, self.n_machines, self.processing_times)
        job_end_times = get_job_end_times_arr(schedule_info)
        # Ermittle spätestes Job-Ende auf der letzten Maschine (=Zykluszeit)
        makespan = get_makespan(schedule_info)
        # Summe aller Job-Endzeitpunkte auf der letzten Maschine (=Flusszeit)
        flow_time = sum(job_end_times)
        # Setze Makespan und Flowtime als beide zu minimierenden Ziele
        out["F"] = makespan
    
    def _last_machine_operations(self, schedule_info):
        result = []
        for operation in schedule_info:
            if operation['resource'] == self.n_machines - 1:
                result.append(operation)
        return result
    
    def plot_solution(self, ind):
        # Diese Methode kann am Ende den Gantt-Chart ausgeben
        plot_gantt(generate_schedule(list(ind), self.n_machines, self.processing_times))



def optimize(param):
    if param['adaptive_rate_mode'] == 'rl':
        lr = 0.0001
        agent = PPO('MlpPolicy', adaptation_env(), lr, n_epochs=1, n_steps=32)
        tmp_path = "./logs/sb3_log/"
        logger = configure(tmp_path, ["stdout", "csv"])
        agent.set_logger(logger)
    else:
        agent = None

    result, best_schedule, makespan_flowtime = run_alg_parallel(run_alg, param)
    return makespan_flowtime, best_schedule

if __name__ == "__main__":
    instances = ['ta001', 'ta002', 'ta022', 'ta031']

    # Dieses Dictionary ist erforderlich um den Algorithmus nach den oben standardisierten Methoden auszuführen
    # Es können beliebige weitere Einträge hinzugefügt werden, die dann z.B. in der Problemklasse gespeichert werden können
    param = {
        'problem_getter': PFSSP,
        'n_jobs': 0,
        'n_machines': 0,
        'processing_times': [0],
        'alg_name': 'nsga2', # Siehe algorithm_factory.py
        'crossover': TwoPointCrossover(prob=0.8),
        'mutation': InversionMutation(prob=0.2),
        'sampling': PermutationRandomSampling(), # Standardmethode des Frameworks
        'adaptive_rate_mode': 'none', # none | icdm (linear increasing crossover, decr. mutation) | imdc (incr. mutation, decr. crossover)
        'obj_weights': [0.8, 0.2],
        'n_evals': 800, # Max. Anz. der Evaluationen (Stopping-Kriterium anstelle von max. Anz. Generationen)
        'parallel_computing': 'multi_processing', # multi_threading | multi_processing | dask
        'n_parallel_worker': 4, # Anzahl paralleler Prozesse oder Threads
        'eliminate_duplicates': False, # Eliminiere Duplikate (redundante Individuen)
        'seed': None, # Starte mit Zufallszahl
        'print_results': False, # Endergebnis ausgeben
        'print_iterations': True, # Zwischenergebnis ausgeben je Generation/Iteration
        'plot_solution': False, # Diagramm ausgeben
        'functions': instances,
        'pop_size': 50,
    }
    optimize(param)
    pass
