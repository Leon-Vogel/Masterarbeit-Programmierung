import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


from misc_utils.copy_helper import fast_deepcopy
import math
import random
import numpy as np
from isri_utils.paper2_shifting_jobs.evaluation_function import (
    FitnessCalculation,
    generate_schedule,
    get_observation_features,
)
from isri_utils.paper2_shifting_jobs.read_taillard_data import get_taillard_with_uncert_proc_time
from isri_utils.paper2_shifting_jobs.create_neighbor import create_neighbor_random, select_new_job_pos, SelectNextJobAction
import time
import pandas as pd

def perform_simulated_annealing(ind, evaluation_func, create_neighbor_func, n_iterations=50, do_print=True):
    best = fast_deepcopy(ind)
    curr = fast_deepcopy(ind)
    ev_start = evaluation_func(ind)
    ev_best = ev_start
    ev_curr = ev_start
    jobpos = len(ind)-1
     
    if do_print:
        print(f"curr fit score: {ev_curr['fitness_score']}")

    #attempts = 0
    for i in range(n_iterations):
        if do_print and i % 20 == 0:
            print(f"\titeration={i}/{n_iterations}")

        temperature = (n_iterations - i) / n_iterations

        selectnextjob = random.randint(0, max(value for name, value in vars(SelectNextJobAction).items() if not name.startswith("__")))
        jobpos = select_new_job_pos(selectnextjob, ev_curr["schedule"], ev_curr["df"], curr, jobpos)
        candidate = create_neighbor_func(fast_deepcopy(curr), jobpos)
        ev_candidate = evaluation_func(candidate)
        
        if ev_candidate["fitness_score"] < ev_best["fitness_score"]:
            if do_print:
                print(
                    f"iteration:{i}: improved from {ev_best['fitness_score']:.3f} -> {ev_candidate['fitness_score']:.3f})"
                )
            best = candidate
            curr = candidate
            ev_best = ev_candidate
            ev_curr = ev_candidate
            #attempts = 0
        elif ev_candidate["fitness_score"] < ev_curr["fitness_score"]:
            curr = candidate
            ev_curr = ev_candidate
        else:
            # attempts += 1
            # if attempts >= max_attempts:
            #     curr = best
            #     attempts = 0
            # else:
            #     diff = ev_cand - ev_curr
            #     if random.random() < math.exp(-diff / temperature):
            #         curr = candidate
            diff = ev_candidate["fitness_score"] - ev_curr["fitness_score"]
            if random.random() < math.exp(-diff / temperature):
                curr = candidate
                ev_curr = ev_candidate

    return best


if __name__ == "__main__":
    # verwende die Shortest Processing Time Priorirätsregel als Eröffnungsverfahren
    
    scores = []
    n_experiments_per_instance = 20
    n_annealing_iterations = 75
    instances = ["ta031", "ta032", "ta033", "ta034", "ta035", "ta038", "ta039", "ta040",]
    # instances = ["ta048", "ta049", "ta050"]
    fitness_features = ["flowtime_sum", "worker_load", "job_uncertainty_mean"]
    fitness_weights = [0.4, 0.4, 0.2]
    results = {}
    start = time.time()
    progress = 0
    for inst in instances:
        for exp in range(n_experiments_per_instance):
            ind = get_taillard_with_uncert_proc_time(inst)
            ind = sorted(ind, key=lambda x: x["operations"][0]["expected_duration"])
            fit_calc = FitnessCalculation(fitness_features, fitness_weights)
            def eval(ind):
                schedule, df = generate_schedule(ind)
                obs = get_observation_features(schedule, df, ind, 0)
                return fit_calc.set_fitness_score_and_log_schedule(ind, schedule, obs, df)
            winner = perform_simulated_annealing(ind, eval, create_neighbor_random, n_iterations=n_annealing_iterations, do_print=False)
            schedule, df = generate_schedule(winner)
            obs = get_observation_features(schedule, df, winner, 0)
            res = fit_calc.set_fitness_score_and_log_schedule(winner, schedule, obs, df)
            key = f"{inst}--n_iter={n_annealing_iterations}--n_exp={n_experiments_per_instance}"
            if key not in results:
                results[key] = []
            results[key].append({
                "linear_score": res["fitness_score"],
                "flowtime_sum": res["fitness_metrics"]["flowtime_sum"],
                "worker_load": res["fitness_metrics"]["worker_load"],
                "job_uncertainty_mean": res["fitness_metrics"]["job_uncertainty_mean"],
            })
            progress += 1
            status = f"{int(time.time()-start)}s ({int(progress/(n_experiments_per_instance*len(instances))*100)} %);"            
            status += f" INSTANCE {instances.index(inst)+1}/{len(instances)} ({inst});"            
            status += f" EXPERIMENT {exp+1}/{n_experiments_per_instance}"            
            print(f"{status} - fitness score: {res['fitness_score']}")
    
    results_dir = os.path.join(os.path.dirname(__file__), "experiment_results", "SA")
    for key, val in results.items():
        df = pd.DataFrame(val)
        file = os.path.join(results_dir, f"{key}.csv")
        df.to_csv(file, sep=';', encoding='utf-8', index=False, decimal=',')
        stats = df.describe()
        file = os.path.join(results_dir, f"{key}_STATS.csv")
        stats.to_csv(file, sep=';', encoding='utf-8', index=False, decimal=',')