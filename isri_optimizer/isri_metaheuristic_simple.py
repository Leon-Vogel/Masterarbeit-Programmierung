from _dir_init import *
import pickle
import random as rnd
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from isri_simulation.isri_simulation import Log, Product, Simulation, Siminfo
from misc_utils.copy_helper import fast_deepcopy

# from misc_utils.schedule_to_graph_converter import schedule_data_to_graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo_utils.algorithm_runner import run_alg, run_alg_parallel
from scipy.spatial.distance import cdist

try:
    from rl_agent import adaptation_evn, linear_schedule, reward_func
    from stable_baselines3.common.logger import configure
    from stable_baselines3.ppo import PPO
except ImportError:
    warnings.warn(
        "Stable baselines not available. Make sure no RL functionality is used."
    )


def _moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def _diff_n(a, n=1):
    # a = a.reshape(-1)
    return a[n:] - a[:-n]


def get_dummy_test_data(n_jobs, jobs_per_line, n_machines, time_range):
    assert jobs_per_line % 12 == 0, "jobs_per_line muss ein vielfaches von 12 sein!"
    assert n_jobs / 2 >= jobs_per_line, "zu wenig jobs insgesamt für zwei linien!"
    due_date_range = (
        int(round(time_range[1] * 0.7 * n_jobs * n_machines)),
        int(round(time_range[1] * n_jobs * n_machines)),
    )
    return {
        f"job{i}": {
            "times": [
                rnd.randint(time_range[0], time_range[1]) for _ in range(n_machines)
            ],
            "due_date": rnd.randint(due_date_range[0], due_date_range[1]),
        }
        for i in range(n_jobs)
    }


def isinfeasable(ind: np.ndarray, jobs: dict, jpl: int, n_lines: int):
    """
    Prüft ob ein Plan feasable ist. Gibt Feasability, INtersection zurück
    """
    if isinstance(ind, np.ndarray):
        assert len(set(ind)) == ind.shape[0], "Individuum enthält duplizierte Produktnummern"
    elif isinstance(ind, list):
        assert len(set(ind)) == len(ind), "Individuum enthält duplizierte Produktnummern"
    
    for product in ind:
        assert product in jobs.keys(), "Produkt im Individuum ohne Daten"
            
    plans = split_jobs_on_lines(ind, jpl=jpl, lines=n_lines)
    groups_planned = set(
        [jobs[job_nr]["group"] for job_nr in flatten_chromsome(plans[:-1])]
    )
    groups_unplanned = set([jobs[job_nr]["group"] for job_nr in plans[-1]])
    intersection = groups_planned.intersection(groups_unplanned)
    return len(intersection) > 0, intersection


def plan_by_due_date(jobs: dict, jpl: int, n_lines=2):
    """
    Einfache Planungsheuristik, die Produkte nach due date einsortiert
    """
    sort_by_due_date = list(
        dict(sorted(jobs.items(), key=lambda item: item[1]["due_date"])).keys()
    )
    plan = []
    for line in range(n_lines):
        line_plan = sort_by_due_date[line : jpl * 2 + line][::n_lines]
        plan = plan + line_plan
    backlog = sort_by_due_date[jpl * n_lines :]
    plan = plan + backlog
    return plan


def plan_by_due_date_save(jobs: dict, jpl: int, n_lines=2):
    """
    Einfache Planungsheuristik, die Produkte nach due date einsortiert
    """
    sort_by_due_date = list(
        dict(sorted(jobs.items(), key=lambda item: item[1]["due_date"])).keys()
    )
    last_element_planned = sort_by_due_date[jpl * n_lines - 1]
    first_element_backlog = sort_by_due_date[jpl * n_lines]
    last_element_group = jobs[last_element_planned]["group"]
    if last_element_group == jobs[first_element_backlog]["group"]:
        group_elements = [
            idx
            for idx, job_id in enumerate(jobs.keys())
            if jobs[job_id]["group"] == last_element_group
        ]
        group_start = min(group_elements)
        group_end = max(group_elements)
        group_size = group_end - group_start + 1
        elements_in_backlog = group_end - (jpl * n_lines) + 1
        pull_in_front_prob = np.random.uniform(0, 1, 1)[0]
        # Ziehe Elemente der Gruppe aus dem Backlog nach vorne
        if pull_in_front_prob > 0.5:
            possible_movements = [
                job_id
                for job_id in sort_by_due_date[: jpl * n_lines - 1]
                if len(
                    [
                        job
                        for job in sort_by_due_date[: jpl * n_lines - 1]
                        if jobs[job_id]["group"] == jobs[job]["group"]
                    ]
                )
                == elements_in_backlog
            ]  # Suche nach allen Jobs, deren Gruppengröße gleich der gewünschten Tauschgröße ist

    plan = []
    for line in range(n_lines):
        line_plan = sort_by_due_date[line : jpl * 2 + line][::n_lines]
        plan = plan + line_plan
    backlog = sort_by_due_date[jpl * n_lines :]
    plan = plan + backlog
    return plan


def find_next_jobs_by_due_date(jpl: int, n_lines: int, jobs: dict) -> list:
    """
    Finde alle jobs die Verplant werden sollen
    (strikt nach Due Date bis auf Konflikte mit Gruppengröße am Ende)
    """
    sort_by_due_date = list(
        dict(sorted(jobs.items(), key=lambda item: item[1]["due_date"])).keys()

    )
    to_be_planned = []
    remaining_places = jpl * n_lines
    idx = 0
    while remaining_places > 0 and idx < len(jobs):
        next_job = jobs[sort_by_due_date[idx]]
        next_job_group = next_job['group']
        all_jobs_from_group = [job_id for job_id, values in jobs.items() if values['group'] == next_job_group]
        if len(all_jobs_from_group) <= remaining_places:
            to_be_planned = to_be_planned + all_jobs_from_group
            remaining_places -= len(all_jobs_from_group)
        idx += len(all_jobs_from_group)
    return to_be_planned

def plan_line_heuristic(job_ids: list, jobs: dict, next_n: int = 4, dist_measure: str = 'euclidean', max_skips: int = 6):
    """
    Assumes job_ids is sorted by deadline and jobs contains in the 'time' value a list of times. Plans
    by starting with the first job. Afterwards from the next_n jobs by deadline the one with the maximum
    dist_measure to the last planned job is chosen.
    """
    plan = [job_ids.pop(0)]
    skips = {job: 0 for job in job_ids}
    while len(job_ids) > 1:
        current_time_vec = np.array(jobs[plan[-1]]['times'])
        plan_horizon = min((next_n, len(job_ids) - 1))

        # Jobs dürfen nicht ewig "vor sich her geschoben werden"
        plan_horizon_skips = [skips[job_id] for job_id in job_ids[:plan_horizon]]
        if max(plan_horizon_skips) >= max_skips:
            next_job = np.argmax(plan_horizon_skips)
        
        # Sonst nach Distanz
        else:
            next_jobs_time_vecs = [np.array(jobs[job]['times']) for job in job_ids[:plan_horizon]]
            distances = cdist([current_time_vec], next_jobs_time_vecs, metric=dist_measure)
            next_job = np.argmax(distances)

        job = job_ids.pop(next_job)
        plan.append(job)    
        for job in job_ids[:plan_horizon - 1]:
            skips[job] += 1
        
    plan.append(job_ids.pop(0))
    return plan


def heuristic_chromosome(jpl: int, n_lines: int, jobs: dict, next_n: int = 2, dist_measure: str = 'euclidean'):
    jobs_to_be_planned = find_next_jobs_by_due_date(jpl, n_lines, jobs)
    # Jobs werden Zufällig auf die Linien zugeordnet
    line_idx = []
    for line in range(n_lines):
        line_idx = line_idx + [line] * jpl
    rnd.shuffle(line_idx)

    # Make plan
    plan = []
    for line in range(n_lines):
        line_jobs = [job for idx, job in enumerate(jobs_to_be_planned) if line_idx[idx] == line]
        line_plan = plan_line_heuristic(line_jobs, jobs, next_n)
        plan = plan + line_plan
    
    # Remaining Jobs to Backlog
    backlog = [job_id for job_id in jobs.keys() if job_id not in plan]
    plan = plan + backlog
    return plan

def get_random_chromosome(jpl, n_lines, jobs, n_random_switches=5):
    """jpl: jobs_per_line, jobs: all jobs from problem"""
    job_numbers = np.copy(list(jobs.keys()))
    assert len(job_numbers) == len(set(job_numbers)), "dublet job numbers not allowed!"
    rnd.shuffle(job_numbers)
    return job_numbers


def get_due_date_random_chromosome(jpl, n_lines, jobs, n_random_switches=3):
    """jpl: jobs_per_line, jobs: all jobs from problem"""
    plan = np.array(plan_by_due_date(jobs, jpl))
    idxs = np.arange(0, jpl * 2, 1, dtype=int)
    rdm_switch_from = np.random.choice(idxs, n_random_switches)
    rdm_switch_to = np.random.choice(idxs, n_random_switches)
    for i in range(n_random_switches):
        plan[[rdm_switch_to[i], rdm_switch_from[i]]] = plan[
            [rdm_switch_from[i], rdm_switch_to[i]]
        ]
    return plan


def get_due_date_chromosome_safe_by_design(
    jpl, n_lines, jobs, n_random_mutations=(5, 8)
):
    curr_jobs_assigned_on_lines = [[] for _ in range(n_lines)]
    backlog = []

    def assign_jobs_on_lines():
        def get_group_list_with_jobs_by_duedate():
            for k in jobs.keys():
                jobs[k]["job_id"] = k
            groups = set([v["group"] for v in jobs.values()])
            groups = [
                {
                    "group": g,
                    "jobs": [j_val for j_val in jobs.values() if j_val["group"] == g],
                }
                for g in groups
            ]
            for g in groups:
                g["n_jobs"] = len(g["jobs"])
                g["due_date"] = g["jobs"][0]["due_date"]
            return sorted(groups, key=lambda g: g["due_date"])

        def enough_slots_avail_on_all_lines(n_jobs_in_curr_group):
            n_jobs_already_assigned = sum(
                [
                    len(jobs_on_single_line)
                    for jobs_on_single_line in curr_jobs_assigned_on_lines
                ]
            )
            n_slots_free = jpl * n_lines - n_jobs_already_assigned
            return n_jobs_in_curr_group <= n_slots_free


        def distribute_jobs_of_a_group_on_lines_equally(jobs_to_put_on_lines, lines):
            while jobs_to_put_on_lines:
                lines = sorted(lines, key=lambda l: len(l))
                lines[0].append(jobs_to_put_on_lines.pop())
                assert len(lines[0]) <= jpl

        for g in get_group_list_with_jobs_by_duedate():
            if not enough_slots_avail_on_all_lines(g["n_jobs"]):
                backlog.extend(g["jobs"])
                continue
            distribute_jobs_of_a_group_on_lines_equally(
                g["jobs"], curr_jobs_assigned_on_lines
            )

    def concat_chromosome_with_job_ids():
        chrom = curr_jobs_assigned_on_lines
        chrom.append(backlog)
        chrom = flatten_chromsome(chrom)
        chrom = [gene["job_id"] for gene in chrom]
        return chrom
       
    def perform_random_mutations(chrom_flat):
        n_mutations = rnd.randint(n_random_mutations[0], n_random_mutations[1])
        for _ in range(n_mutations):
            chrom_flat = chrom_flat
            chrom_flat = group_save_mutation(chrom_flat, jobs, jpl, n_lines)
        return chrom_flat

    assign_jobs_on_lines()   
    chrom_flat = concat_chromosome_with_job_ids()

    return perform_random_mutations(chrom_flat)



class ISRISampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # test = [
        #     get_due_date_random_chromosome(
        #         problem.jobs_per_line, problem.n_lines, problem.jobs
        #     )
        #     for _ in range(n_samples)
        # ]
        return [
            get_due_date_chromosome_safe_by_design(
                problem.jobs_per_line, problem.n_lines, problem.jobs
            )
            for _ in range(n_samples)
        ]


def read_ISRI_Instance(
    instance: pd.DataFrame,
    date_time,
    mjd,
    stunden_nach_rohbau=4,
    stunden_nach_lack=1.5,
    stunden_nach_reo=0.5,
    min_per_lud=3,
    next_n=200,
):

    instance = instance[instance.tr_zsbtyp == "ES        "]
    instance["freigabe"] = pd.to_datetime(instance.freigabe)
    instance["edi399"].replace("0001.01.01 00:00:00", np.nan, inplace=True)
    instance["edi399"] = pd.to_datetime(instance.edi399)
    instance["zielwerk"] = instance.zielwerk.astype(int)

    mjd.columns = ["material", "op", "mtm_time"]
    mjd.mtm_time = mjd.mtm_time * 60
    mjd["material_stamm"] = mjd.material.apply(lambda mat: mat.split("-")[0])
    min_seq_nr = min(instance[instance.zielwerk == 37].seqnr)
    ops = [11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 101, 102]
    jobs = {}
    min_rohbau = instance[instance.zielwerk == 65].freigabe.min()
    min_lack = instance[instance.zielwerk == 65].edi399.min()
    rohbau_lag = date_time - min_rohbau
    lack_lag = date_time - min_lack
    for job in instance.iterrows():
        job_id = job[1].auidnr
        if job[1].zielwerk == 37:
            place_in_queue = job[1].seqnr - min_seq_nr
            deadline = (
                date_time
                + pd.to_timedelta(min_per_lud, "minute") * place_in_queue
                + pd.to_timedelta(1, "hours")
            )

        if job[1].zielwerk == 65:
            if pd.isna(job[1].edi399):
                deadline = (
                    job[1].freigabe
                    + pd.to_timedelta(stunden_nach_rohbau, "hours")
                    + rohbau_lag
                )

            else:
                deadline = (
                    job[1].edi399
                    + pd.to_timedelta(stunden_nach_lack, "hours")
                    + lack_lag
                )

        if job[1].auftragsart == "REO":
            deadline = date_time + pd.to_timedelta(stunden_nach_reo, "hours")

        material_list = [m.split("/")[0] for m in job[1].variant.split("|")[::2]]
        material_list = [mat.split("-")[0] for mat in material_list]
        times = []
        for op in ops:
            work = mjd[
                (mjd.op == op) & (mjd.material_stamm.isin(material_list))
            ].mtm_time.sum()
            times.append(work)

        due_date = (deadline - date_time).total_seconds()
        jobs[job_id] = {
            "times": times,
            "due_date": due_date,
            "group": job[1].produktionsnummer,
        }

    if next_n is not None:
        # Sortiere nach Deadline
        deadline_sorted = list(
            dict(sorted(jobs.items(), key=lambda item: item[1]["due_date"])).keys()
        )
        short_jobs = {}
        for i in range(next_n):
            short_jobs[deadline_sorted[i]] = jobs[deadline_sorted[i]]
        return short_jobs
    else:
        return jobs


def split_jobs_on_lines(ind, jpl, lines):
    """
    Überführt ein Chromosom (ind) ind die Belegungspläne für die Linien.
    Bsp. ind = [1,2,3,4,5,6,7], jpl = 3
    Rückgabe = [[1,2,3],[4,5,6],[7]] (2 Lines a 3 Jobs, Job 7 in der Restekiste)
    """
    splitted = []
    for line in range(lines):
        splitted.append(list(ind[line * jpl : line * jpl + jpl]))
    splitted.append(list(ind[lines * jpl :]))
    return splitted


def flatten_chromsome(jobs_on_lines):
    return [item for sublist in jobs_on_lines for item in sublist]


def get_fitness(
    x,
    jobs,
    jpl,
    conv_speed: int,
    n_machines: int,
    ma_window_size=3,
    n_lines=2,
    return_value="minmax",
    use_sim: bool = True,
    sim: Siminfo = None,
    simstart: datetime = None,
):
    """berechne den fitnesswert für einen lösungskandidaten x ohne simulationsmodell.
    rückgabe zweier fitnesswerte: geglättete belastung, kleinster due date buffer. Entspricht MinMax Ziel im Paper
    """
    if not use_sim:
        diffsum, tardiness = fast_sim_diffsum(
            x, jobs, jpl, conv_speed, n_machines, n_lines=n_lines
        )
        schedule_info = {"human_workload_indicator": diffsum, "deadline_gap": tardiness}
    else:
        assert sim is not None, "Simulation is not defined"
        simcopy = Simulation(sim)
        prod_plan = split_jobs_on_lines(x, jpl, lines=n_lines)
        linie_A = [
            Product(
                p,
                jobs[p]["times"],
                simstart + timedelta(seconds=jobs[p]["due_date"]),
                sim=simcopy,
            )
            for p in prod_plan[0]
        ]
        linie_B = [
            Product(
                p,
                jobs[p]["times"],
                simstart + timedelta(seconds=jobs[p]["due_date"]),
                sim=simcopy,
            )
            for p in prod_plan[1]
        ]
        sim_plan = {0: linie_A, 1: linie_B}
        # with open('test_plan.pkl', 'wb') as outfile:
        #     pickle.dump(x, outfile)
        simcopy.run(sim_plan, simstart)
        deadline_gaps = simcopy.log.deadline_goal()
        simend = max(simcopy.log.eventlog["ReachedEnd"])
        deadline_not_produced = [simend - jobs[p]["due_date"] for p in prod_plan[2]]
        deadline_gaps = deadline_gaps + deadline_not_produced
        schedule_info = {}
        # deadline_gap_exp = np.sum(np.array(deadline_gaps) / 3600)
        deadline_gap_exp = sum([np.exp(gap / 3600) for gap in deadline_gaps if gap > 0])

        # print('Prep: %.2f | Sim: %.2f, | MA: %.2f | Results: %.2f' % (preptime, simtime, ma_time, restime))
        if return_value == "minmax":
            workload = simcopy.log.workload_goal()
            workload_ma = [
                _moving_average(workload[row, :], ma_window_size)
                for row in range(workload.shape[0])
            ]
            workload = np.array(workload_ma)
            schedule_info["human_workload_indicator"] = max(workload.flatten())
            schedule_info["deadline_gap"] = deadline_gap_exp
        elif return_value == "minavg":
            workload = simcopy.log.workload_goal()
            workload_ma = [
                _moving_average(workload[row, :], ma_window_size)
                for row in range(workload.shape[0])
            ]
            workload = np.array(workload_ma)
            schedule_info["human_workload_indicator"] = np.mean(workload.flatten())
            schedule_info["deadline_gap"] = deadline_gap_exp
        elif return_value == "minvar":
            workload = simcopy.log.workload_goal()
            workload_ma = [
                _moving_average(workload[row, :], ma_window_size)
                for row in range(workload.shape[0])
            ]
            workload = np.array(workload_ma)
            schedule_info["human_workload_indicator"] = np.var(
                np.array(workload).flatten()
            )
            schedule_info["deadline_gap"] = deadline_gap_exp
        elif return_value == "full_workload":
            workload = simcopy.log.workload_goal()
            workload_ma = [
                _moving_average(workload[row, :], ma_window_size)
                for row in range(workload.shape[0])
            ]
            workload = np.array(workload_ma)
            schedule_info["human_workload_indicator"] = np.array(workload)
            schedule_info["deadline_gap"] = deadline_gaps
        elif return_value == "diff_sum":
            n_lines = len(simcopy.stations)
            plan = split_jobs_on_lines(x, jpl, lines=n_lines)
            diffsums = 0
            for line in range(n_lines):
                jobs_on_line = plan[line]
                workload_per_op = np.array([jobs[j]["times"] for j in jobs_on_line])
                for op in range(workload_per_op.shape[1]):
                    diffsum = (
                        sum(
                            [
                                np.abs(_diff_n(workload_per_op[:, op], n) / 3600).sum()
                                * 1
                                / n
                                for n in range(1, ma_window_size)
                            ]
                        )
                        * -1
                    )
                    diffsums += diffsum
            
            schedule_info['human_workload_indicator'] = diffsums
            schedule_info['deadline_gap'] = deadline_gap_exp
        
    # infeasable, group_violations = isinfeasable(x, jobs, jpl, n_lines)

    return schedule_info["human_workload_indicator"], schedule_info["deadline_gap"]# , len(group_violations)


def get_schedule_kpis(
    ind,
    jobs,
    jpl,
    conv_speed: int,
    n_machines: int,
    ma_window_size=3,
    return_value="minmax",
):
    """DEPRECATED"""
    splitted_jobs = split_jobs_on_lines(ind, jpl)
    schedule_info = {"operations": []}
    ma_values = []
    full_ma_matrix = []
    deadline_gaps = []

    for line_idx, line in enumerate(splitted_jobs[:-1]):
        deadline_gap = [
            (idx * conv_speed + conv_speed * n_machines) - jobs[job]["due_date"]
            for idx, job in enumerate(line)
        ]
        deadline_gaps = deadline_gaps + deadline_gap
        for machine in range(n_machines):
            spare_time = [
                (conv_speed - jobs[job]["times"][machine]) * -1 for job in line
            ]
            mean_val = np.mean(spare_time)
            spare_time = (
                [mean_val] * ma_window_size + spare_time + [mean_val] * ma_window_size
            )
            ma = [
                np.mean(spare_time[i - ma_window_size : i + ma_window_size])
                for i in range(ma_window_size, len(spare_time) - ma_window_size)
            ]
            if return_value == "minmax":
                ma_values.append(np.max(ma))
            elif return_value == "minavg":
                ma_values.append(np.mean(ma))
            full_ma_matrix.append(ma)

            curr_start_time = 0
            for job in line:
                schedule_info["operations"].append(
                    {
                        "start_time": curr_start_time,
                        "end_time": curr_start_time + conv_speed,
                        "job": list(ind).index(job),
                        "resource": machine
                        + 100
                        * line_idx,  # line 1 machines: 0,1,2,... ; line 2 machines: 100,101,...
                        "line": line_idx,
                    }
                )
                curr_start_time += conv_speed

    # set stress indicators
    deadline_gaps = deadline_gaps + [
        (conv_speed * jpl) + 3600 - jobs[job]["due_date"] for job in splitted_jobs[-1]
    ]
    deadline_gap_exp = np.array(deadline_gaps) / 3600
    schedule_info["deadline_gap"] = np.sum(deadline_gap_exp)
    if return_value == "minmax":
        schedule_info["human_workload_indicator"] = max(ma_values)
    elif return_value == "minavg":
        schedule_info["human_workload_indicator"] = np.mean(ma_values)
    elif return_value == "minvar":
        schedule_info["human_workload_indicator"] = np.var(
            np.array(full_ma_matrix).flatten()
        )
    elif return_value == "full_workload":
        schedule_info["human_workload_indicator"] = np.array(full_ma_matrix)
        schedule_info["deadline_gap"] = deadline_gaps

    # set main kpis
    schedule_info["wip"] = sum(
        list(
            map(lambda o: o["end_time"] - o["start_time"], schedule_info["operations"])
        )
    )
    schedule_info["makespan"] = sum(
        list(
            map(
                lambda o: o["end_time"] - o["start_time"],
                filter(lambda x: x["line"] == 0, schedule_info["operations"]),
            )
        )
    )
    """
    schedule_info["flow_time"] = sum(list(map(lambda o: o["end_time"], schedule_info["operations"])))
    schedule_info["tardiness"] = schedule_info["deadline_gap"]
    schedule_info["wip_per_resource_mean"] = np.mean(schedule_info["wip"] / (n_machines * len(splitted_jobs[:-1])))
    schedule_info["wip_per_resource_stdev"] = np.std(schedule_info["wip"] / (n_machines * len(splitted_jobs[:-1])))
    schedule_info["n_operations"] = len(schedule_info["operations"])
    schedule_info["tardiness_mean"] = np.mean(deadline_gap_exp)
    schedule_info["tardiness_stdev"] = np.std(deadline_gap_exp)
    schedule_info["proc_time_per_resource_mean"] = conv_speed
    schedule_info["proc_time_per_resource_stdev"] = 0
    schedule_info["slack_time_per_resource_mean"] = 0
    schedule_info["slack_time_per_resource_stdev"] = 0
    # graph, edge_attr = schedule_data_to_graph(schedule_info)
    # schedule_info["graph"] = graph
    """
    return schedule_info


def fast_sim_diffsum(
    ind, jobs, jpl, conv_speed: int, n_machines: int, n_lines=2, window_size=4
):
    """Schnelle Version der Simulation mit Diff-Sum und Deadline als Ziel"""
    plans = split_jobs_on_lines(ind, jpl, lines=n_lines)

    diffsum = 0
    tardiness_sum = 0
    for plan in plans[:-1]:  # Backlog does not Count
        # Diffsum Goal
        times_array = [jobs[job]["times"] for job in plan]
        times_array = np.array(times_array)
        for n in range(1, window_size):
            differences = _diff_n(times_array, n)
            n_difference = np.sum(np.abs(differences)) * 1 / n
            diffsum -= n_difference

        # Deadline
        deadlines = [jobs[job]["due_date"] for job in plan]
        deadlines = np.array(deadlines)
        start_times = np.arange(len(plan)) * conv_speed
        finish_times = start_times + (conv_speed * n_machines)
        tardiness = (finish_times - deadlines) / 3600
        tardiness_sum += np.sum(np.exp(tardiness))

    deadlines_backlog = np.array([jobs[job]["due_date"] for job in plans[-1]])
    # End time for backlog jobs = jpl * conv_speed + (conv_speed * n_machines) * 2
    finish_time_backlog = jpl * conv_speed + (conv_speed * n_machines) * 2
    tardiness_sum_backlog = np.sum(np.exp((finish_time_backlog - deadlines_backlog) / 3600))
    tardiness_sum += tardiness_sum_backlog
        # Alte Version mit Clipping: 
        # tardiness = np.clip(
        #     (finish_times - deadlines) / 3600, 0, 2
        # )  # 3600 Sekunden = 1 Stunde
        # tardiness_exp = np.sum(np.exp(tardiness[tardiness > 0]))
        # tardiness_sum += tardiness_exp
    return diffsum, tardiness_sum


class ISRIDuplicateElimination(DuplicateElimination):
    def __init__(self, func=None) -> None:
        super().__init__(func)

    def _do(self, pop, other, is_duplicate):
        if other is None:
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    same = pop[i] == pop[j]
                    if same:
                        is_duplicate[i] = True
                        break

        else:
            for i in range(len(pop)):
                for j in range(len(other)):
                    same = pop[i] == other[j]
                    if same:
                        is_duplicate[i] = True
                        break
        return is_duplicate


class ISRIProblem(ElementwiseProblem):
    def __init__(self, param, runner):
        """standardmäßig 2 objectives: Belastung + maximiere den kleinsten due date puffer. Für Belastung gibt es drei Optionen: 'minmax' 'minavg' und 'minvar'"""
        super().__init__(
            n_var=param["n_jobs"],
            n_obj=len(param["obj_weights"]),
            n_constr=0,
            xl=0,
            xu=param["n_jobs"] - 1,
            elementwise_evaluation=True,
            elementwise_runner=runner,
        )
        self.jobs_per_line = param["jobs_per_line"]
        self.jobs = param["jobs"]
        self.n_lines = param["n_lines"]
        self.n_machines = param["n_machines"]
        self.conveyor_speed = param["conveyor_speed"]
        self.ma_window_size = param["ma_window_size"]
        self.criterion = param["criterion"]
        self.simstart = param["simstart"]
        self.simulation = param["simulation"]
        self.use_sim = param["use_sim"]

    def _evaluate(self, x, out, *args, **kwargs):
        obj_smoothed_worker_load, obj_min_due_date_buffer = (
            get_fitness(
                x,
                self.jobs,
                jpl=self.jobs_per_line,
                conv_speed=self.conveyor_speed,
                n_machines=self.n_machines,
                n_lines=self.n_lines,
                ma_window_size=self.ma_window_size,
                return_value=self.criterion,
                use_sim=self.use_sim,
                sim=self.simulation,
                simstart=self.simstart,
            )
        )  # rufe oben definierte funktion auf

        out["F"] = [
            obj_smoothed_worker_load,
            obj_min_due_date_buffer
        ]  # fitnesswerte müssen hier noch nicht normalisiert werden.

    def plot_solution(self, x, savepath=None):
        worker_load, min_due_date_buffer = get_fitness(
            x,
            self.jobs,
            jpl=self.jobs_per_line,
            conv_speed=self.conveyor_speed,
            n_machines=self.n_machines,
            ma_window_size=self.ma_window_size,
            return_value="full_workload",
            use_sim=self.use_sim,
            sim=self.simulation,
            simstart=self.simstart,
        )
        worker_load *= -1
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        fig, ax = plt.subplots(figsize=(3, 4))
        im = ax.imshow(worker_load, cmap="RdYlGn_r", aspect="auto")
        im.set_clim(vmin=-self.conveyor_speed, vmax=0)
        ax.set_ylabel("Werker", fontname="Arial", fontsize=10)
        ax.set_xlabel("Produkt Nr", fontname="Arial", fontsize=10)
        ax.set_yticks(np.arange(0, worker_load.shape[0]))
        ax.set_title("Belastung der Werker", fontname="Arial", fontsize=10)
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.6, pack_start=True)
        fig.add_axes(cax)
        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            ticks=[
                np.min(worker_load) + 1,
                np.median(worker_load),
                np.max(worker_load),
            ],
        )
        cb.set_ticklabels(
            [
                "Min (%.2f)" % np.min(worker_load),
                "Median (%.2f)" % np.median(worker_load),
                "Max (%.2f)" % np.max(worker_load),
            ],
            fontname="Arial",
            fontsize=10,
        )
        fig.tight_layout()
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath + ".svg", format="svg")

    def plot_deadline_gap(self, x, savepath=None):
        worker_load, min_due_date_buffer = get_fitness(
            x,
            self.jobs,
            jpl=self.jobs_per_line,
            conv_speed=self.conveyor_speed,
            n_machines=self.n_machines,
            ma_window_size=self.ma_window_size,
            return_value="full_workload",
            use_sim=self.use_sim,
            sim=self.simulation,
            simstart=self.simstart,
        )

        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        fig = plt.figure(figsize=[3, 3])
        plt.hist(
            np.array(min_due_date_buffer) / 3600,
            density=False,
            histtype="step",
            color="dimgray",
            linewidth=3,
        )
        plt.xlabel("Deadline Lücke in Stunden", fontname="Arial", fontsize=10)
        plt.ylabel("Anzahl Produkte", fontname="Arial", fontsize=10)
        plt.title("Verteilung der Deadline Lücke", fontname="Arial", fontsize=10)
        plt.tight_layout()
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath + ".svg", format="svg", bbox_inches="tight")

    def plot_line_plan(self, plan: list, savepath=None):
        workload = np.array([self.jobs[job]["times"] for job in plan])
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        fig, ax = plt.subplots(figsize=(3, 4))
        im = ax.imshow(workload, cmap="RdYlGn_r", aspect="auto")
        im.set_clim(vmin=0, vmax=np.max(workload))
        ax.set_ylabel("Produkt", fontname="Arial", fontsize=10)
        ax.set_xlabel("Op", fontname="Arial", fontsize=10)
        ax.set_yticks(np.arange(0, workload.shape[0]))
        ax.set_title("MTM-Zeit Pro OP", fontname="Arial", fontsize=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="vertical",
            ticks=[np.min(workload) + 1, np.median(workload), np.max(workload)],
        )
        cb.set_ticklabels(
            [
                "Min (%.2f)" % np.min(workload),
                "Median (%.2f)" % np.median(workload),
                "Max (%.2f)" % np.max(workload),
            ],
            fontname="Arial",
            fontsize=10,
        )
        fig.tight_layout()
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath + ".svg", format="svg", bbox_inches="tight")


def flip_positions_mutation(ind, all_jobs, jpl, nlines):
    genes_to_flip = rnd.sample(list(ind[: jpl * nlines]), k=2)
    pos1 = ind.index(genes_to_flip[0])
    pos2 = ind.index(genes_to_flip[1])
    ind[pos1], ind[pos2] = ind[pos2], ind[pos1]
    return ind


def flip_based_on_workload_mutation(ind, all_jobs):
    # Prob of being selected based on Workload
    workload_mat = np.array([all_jobs[job]["times"] for job in ind])
    normalised_workload_mat = workload_mat / np.sum(workload_mat, axis=0)
    workload_weight = np.sum(normalised_workload_mat, axis=1)
    workload_weight = workload_weight / np.sum(workload_weight)

    # Prob of being selected based on due Date
    # dt = np.array([all_jobs[job]['due_date'] for job in ind])
    # dt_scaled = dt / np.sum(dt)
    # p = (workload_weight + dt_scaled) / np.sum(workload_weight + dt_scaled)
    job_1 = np.random.choice(ind, p=workload_weight, size=1)[0]
    job_2 = np.random.choice(ind, p=workload_weight, size=1)[0]
    pos1 = ind.index(job_1)
    pos2 = ind.index(job_2)
    ind[pos1], ind[pos2] = ind[pos2], ind[pos1]
    return ind

def organize_by_group(individual, job_dict):
    # Initialize a dictionary to hold groups and their members
    groups = {}
    
    # Go through each job in 'individual' list
    for job in individual:
        if job in job_dict:
            group_id = job_dict[job]['group']
            # Check if the group_id is already in the groups dictionary
            if group_id not in groups:
                groups[group_id] = {'group_id': group_id, 'n_members': 0, 'group_members': []}
            
            # Add the job to the corresponding group and increment the member count
            groups[group_id]['group_members'].append(job)
            groups[group_id]['n_members'] += 1
    
    # Convert the groups dictionary to a list of dictionaries
    return list(groups.values())

def rearrange_elements(lst, original_idxs, new_idxs):
    # Create a copy of the original list to work with, so we don't modify the original list in-place
    temp_list = lst[:]
    
    for original_idx, new_idx in zip(original_idxs, new_idxs):
        # Move element from original index to the new index in the temporary list
        temp_list[new_idx] = lst[original_idx]
    
    return temp_list

def group_save_mutation(ind: list, jobs: dict, jpl: int, n_lines: int, **kwargs):
    ind_copy = fast_deepcopy(ind)
    first_gene = np.random.choice(ind, size=1)[0]
    group_of_first = jobs[first_gene]['group']
    group_jobs = [job for job in ind if jobs[job]['group'] == group_of_first]
    selected_group_in_backlog = ind.index(group_jobs[0]) > jpl * n_lines
    group_list = organize_by_group(ind, jobs)
    groups_with_same_length = [(group['group_id'], group['group_members']) for group in group_list if (group['n_members'] == len(group_jobs)) & (group['group_id'] != group_of_first)]
    if len(groups_with_same_length) == 0:
        return ind
    swap_group_idx = np.random.choice(np.arange(0, len(groups_with_same_length)), 1)[0]
    first_member_swap_group = groups_with_same_length[swap_group_idx][1][0]
    swap_group_in_backlog = ind.index(first_member_swap_group) > jpl * n_lines
    swap_group = groups_with_same_length[swap_group_idx][1]
    if swap_group_in_backlog != selected_group_in_backlog:
        for i in range(len(group_jobs)):
            # Swap elements between group1 and group2 at index i
            idx_group = ind.index(group_jobs[i])
            idx_swap = ind.index(swap_group[i])
            ind[idx_group], ind[idx_swap] = ind[idx_swap], ind[idx_group]
    else:
        # Randomly shuffle all members of these groups
        original_idxs = [ind.index(job) for job in group_jobs + swap_group]
        new_idxs = original_idxs[:] # creates copy
        rnd.shuffle(new_idxs)
        ind = rearrange_elements(ind, original_idxs, new_idxs)

    infeasable, errors = isinfeasable(ind, jobs, jpl, n_lines)
    if infeasable:
        ind = ind_copy
    return ind

class ISRIMutation(Mutation):
    """Tauscht zwei zufällige Jobs aus. Z.B. Job 5 auf Linie 1 mit Job 11 auf Line 2.
    Berücksichtigt auch die 'Restekiste'. Siehe PowerPoint..."""

    def _do(self, problem: ISRIProblem, X, **kwargs):
        """
        X := parent population (list of chromosomes)
        """
        return np.array(
            [
                group_save_mutation(
                    list(y), problem.jobs, problem.jobs_per_line, problem.n_lines
                )
                for y in np.copy(X)
            ]
        )
def find_closest_neg_inf(array, index):
    # Find indices of all -np.inf values in the array
    neg_inf_indices = np.where(array == -np.inf)[0]
    
    # Compute the absolute differences between these indices and the given index
    differences = np.abs(neg_inf_indices - index)
    
    # Find the index of the smallest difference
    closest_index = neg_inf_indices[np.argmin(differences)]
    
    return closest_index

def crossover_save(par1: list, par2: list, jobs: dict, jpl: int, n_lines: int) -> list:
    produced_jobs_split = jpl * n_lines
    child = np.ones(len(par1)) * -np.inf # Sicherstellen dass diese Zahl nie in den Produktnummern vorkommt
    scheduled_jobs = list(par1[:produced_jobs_split])
    rnd.shuffle(scheduled_jobs)
    for product in scheduled_jobs:
        pos_p1 = par1.index(product)
        pos_p2 = par2.index(product)
        if pos_p2 < produced_jobs_split:
            p1_prob = np.random.uniform(0, 1, 1)[0]
            if p1_prob > 0.5:
                child_position = pos_p1
            else:
                child_position = pos_p2
        else:
            child_position = pos_p1
        
        # Ersetzen
        if child[child_position] == -np.inf:
            child[child_position] = product
        else:
            child_position = find_closest_neg_inf(child[:produced_jobs_split], child_position)
            child[child_position] = product

    # Backlog
    child[produced_jobs_split:] = par1[produced_jobs_split:]
    if np.any(child == -np.inf):
        raise RuntimeError("Infeasable Child by Crossover")
    
    # infeasable, errors = isinfeasable(child, jobs, jpl, n_lines)
    # if infeasable:
    #     raise RuntimeError("Infeasable Child by Crossover")
    return list(child)


class ISRICrossover(Crossover):
    """Implementiert den Crossover -> Details siehe PowerPoint. Die Hälfte der Linien wird von P1 übernommen, die andere Hälfte entsprechend der Folge in P2 aufgefüllt, es sei denn der Job ist nicht mehr verfügbar. In diesem Fall wird ein zufälliger Job ausgewählt"""

    def __init__(self, prob):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)

    def _do(self, problem, X, **kwargs):
        """
        X := parent population (list of chromosomes)
        """

        # das folgende ist exemplarischer alt-code für ein job order crossover
        def get_one_child(par1, par2):
            n_lines = len(par1) - 1
            child = []
            par_1_copy = np.random.choice(
                range(n_lines), int(n_lines / 2), replace=False
            )
            all_jobs = set([job for line_plan in par1 for job in line_plan])

            # Bevorzuge Jobs die in p2 schon verplant sind -> durch große Restekiste sonst hohe Wkeit dass nicht dringende Jobs gezogen werden
            job_pool_planned_par2 = set()
            job_pool_fixed_par1 = set()
            for line in par_1_copy:
                job_pool_planned_par2 = job_pool_planned_par2 | set(par2[line])
                job_pool_fixed_par1 = job_pool_fixed_par1 | set(par1[line])

            other_jobs = all_jobs - job_pool_fixed_par1 - job_pool_planned_par2
            job_pool_planned_par2 = job_pool_planned_par2 - job_pool_fixed_par1
            for line in range(n_lines):
                if line in par_1_copy:
                    child.append(par1[line])
                else:
                    child.append([])
                    for job in par2[line]:
                        if job in job_pool_fixed_par1:
                            if len(job_pool_planned_par2) > 0:
                                job = np.random.choice(list(job_pool_planned_par2), 1)[
                                    0
                                ]
                                job_pool_planned_par2 = job_pool_planned_par2 - {job}
                            else:
                                job = np.random.choice(list(other_jobs), 1)[0]
                        child[line].append(job)
                        other_jobs = other_jobs - {job}
                        job_pool_planned_par2 = job_pool_planned_par2 - {job}

            child.append(list(other_jobs | job_pool_planned_par2))
            flat_child = flatten_chromsome(child)
            return flat_child

        n_matings = len(X[0])
        Y = [[], []]
        for i in range(n_matings):
            a = split_jobs_on_lines(
                X[0][i], jpl=problem.jobs_per_line, lines=problem.n_lines
            )  # extract both mating candidated a,b
            b = split_jobs_on_lines(
                X[1][i], jpl=problem.jobs_per_line, lines=problem.n_lines
            )
            # Y[0].append(get_one_child(a, b))
            # Y[1].append(get_one_child(b, a))
            Y[0].append(crossover_save(list(X[0][i]), list(X[1][i]), problem.jobs, problem.jobs_per_line, problem.n_lines))
            Y[1].append(crossover_save(list(X[1][i]), list(X[0][i]), problem.jobs, problem.jobs_per_line, problem.n_lines))
            
        return np.array(Y)


class TestDataHost:
    LUKAS = {
        "path": "C:/Users/lukas/Documents/SUPPORT/AzureDevOps/project_sup_support/src/executable/isri_optimizer/isri_instance",
        "driver": "SQL Server",
        "server": "DESKTOP-0Q4SQIV\SQLEXPRESS",
        "database": "IsriTest",
        "instance_path": "C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/BatchExport/",
        "path_mjd": "C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/mtm_info.csv",
    }
    # FELIX = {"path": "C:/Repo/project_sup_support/src/executable/isri_test_results/test_instances_isri",
    #          "driver": "",
    #          "server": "",
    #          "database": ""
    #         }


if __name__ == "__main__":
    config = TestDataHost.LUKAS

    load_new_instance = False
    if load_new_instance:
        files = os.listdir(config["instance_path"])
        random_file = files[int(np.random.choice(len(files), 1)[0])]
        instance = pd.read_csv(
            config["instance_path"] + random_file, config["path_mjd"], sep=",", header=0
        )
        date_time = pd.to_datetime(
            random_file.split("_")[-1], format="%Y%m%d-%H%M%S.csv"
        )
        mjd = pd.read_csv(config["path_mjd"], sep=";", header=0)
        problem_jobs = read_ISRI_Instance(
            instance,
            date_time,
            mjd,
            stunden_nach_rohbau=6,
            stunden_nach_lack=3,
            stunden_nach_reo=1,
            min_per_lud=4,
            next_n=300,
        )
        with open(TestDataHost.LUKAS["path"].split(".")[0], "wb") as outfile:
            pickle.dump(problem_jobs, outfile)
    else:
        test_instance = pickle.load(open(config["path"], "rb"))
        problem_jobs = test_instance

    n_jobs = len(problem_jobs)
    jobs_per_line = 48
    n_machines = 10
    time_range = (2, 7)
    adaptive_rate_mode = "none"

    if adaptive_rate_mode == "rl":
        lr = 0.0001
        batch_size = 16
        agent = PPO("MlpPolicy", adaptation_evn(), lr, n_epochs=1, n_steps=32)
        tmp_path = "./logs/sb3_log/"
        logger = configure(tmp_path, ["stdout", "csv"])
        agent.set_logger(logger)
    else:
        agent = None
        rollout_buffer = None
        reward_func = None

    use_sim = True
    if use_sim:
        simstart = datetime(2023, 5, 17, 15, 0, 0)
        conn_string = f'Driver={TestDataHost.LUKAS["driver"]};Server={TestDataHost.LUKAS["server"]};Database={TestDataHost.LUKAS["database"]};Trusted_Connection=yes;'
        simulation = Simulation(
            conn_str=conn_string,
            print_subjects=[],
            print_options={"General": False, "Production": False},
        )
    else:
        simstart = None
        simulation = None

    # test_jobs = get_dummy_test_data(n_jobs,jobs_per_line,n_machines,time_range)

    due_date_sorting = plan_by_due_date(problem_jobs, jobs_per_line)
    param = {
        "jobs": problem_jobs,
        "problem_getter": ISRIProblem,
        "crossover": ISRICrossover(0.9),
        "mutation": ISRIMutation(0.05),
        "sampling": ISRISampling(),
        "eliminate_duplicates": ISRIDuplicateElimination(),
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "pop_size": 1000,
        "time_range": time_range,
        "jobs_per_line": jobs_per_line,
        "alg_name": "nsga2",  # nsga2 | moead | rvea | ctaea | unsga3
        "obj_weights": [
            0.8,
            0.2,
        ],  # 1: min geglättete belastung, max minimalen due date puffer
        "conveyor_speed": 208,
        "ma_window_size": 2,
        "n_evals": 2000,  # number of fitness calculations,
        "seed": 25,
        "parallelization_method": "multi_threading",  # multi_threading | multi_processing | dask
        "n_parallelizations": 3,  # number threads or processes,
        "print_results": True,
        "print_iterations": True,  # number threads or processes,
        "adaptive_rate_mode": adaptive_rate_mode,  # none | icdm | imdc | rl
        "criterion": "minmax",  # 'minmax' | 'minavg' | 'minvar',
        "plot_solution": True,
        "use_sim": use_sim,
        "simstart": simstart,
        "simulation": simulation,
        "agent": agent,
        "train_agent": True,
        "reward_func": reward_func,
    }

    problem = ISRIProblem(param, runner=None)
    problem.plot_solution(due_date_sorting)
    problem.plot_deadline_gap(due_date_sorting)
    res, best_chromosome, best_fitness = run_alg_parallel(run_alg, param)
    # pprint(split_jobs_on_lines(best_chromosome, param['jobs_per_line']))
    # pprint(best_fitness)
