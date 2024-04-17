# from line_profiler_pycharm import profile
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import path_definitions
import random
from read_taillard_data import get_taillard_with_uncert_proc_time
from numba import njit
import math
import numpy as np
from scipy.stats import gamma
import pandas as pd
import Levenshtein  # pip install python-Levenshtein
from misc_utils.gantt_plotter_matplotlib import plot_gantt


def get_flowtime(schedule: list, job_id=None):
    if job_id != None:
        job_ids = [job_id]
    else:
        job_ids = set([operation["job"] for operation in schedule])
    scheduled_operations_per_job = {j: [o for o in schedule if o["job"] == j] for j in job_ids}
    job_end_times = [operations[-1]["time_window_end"] for operations in scheduled_operations_per_job.values()]
    return sum(job_end_times)


def get_worker_workload(schedule, ma_window_size=3, goal="min_var"):
    def moving_average(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    resources = {j["resource"] for j in schedule}
    stress_values = []
    for res in resources:
        stress = []
        ops = [o for o in schedule if o["resource"] == res]
        for o in ops:
            window_size = o["time_window_end"] - o["end_time"]
            stress.append(window_size)
        stress_values = stress_values + list(moving_average(np.array(stress), ma_window_size))

    if goal == "min_var":
        return np.var(stress_values)
    elif goal == "min_max":
        return np.max(stress_values)
    elif goal == "min_avg":
        return np.mean(stress_values)
    else:
        raise Exception(f"Goal {goal} is not defined")


@njit(nogil=True, fastmath=True, cache=True)
def f(x, shape, scale):
    # hart gecoded, da numba keine scipy funktionen verwenden kann
    # berechnung der pdf mit alpha (shape) und beta (rate)
    # rate = kehrwert der scale
    rate = 1 / scale
    x_density = (rate**shape / math.gamma(shape)) * x ** (shape - 1) * np.exp(-rate * x)
    return x_density


@njit(nogil=True, fastmath=True, cache=True)
def compute_overlap(critical_x, worst_case_x, shape, scale):
    step = 0.05  # adjust this based on the required precision
    x_values = np.arange(critical_x, worst_case_x, step)
    y_values = f(x_values, shape, scale)
    overlap = np.trapz(y_values, x_values)
    return overlap


def _get_uncertainty_integralsum(schedule, ind):
    overlap_integrals = []
    for operation in schedule:
        ind_job = next(filter(lambda x: x["job"] == operation["job"], ind))
        ind_operation = ind_job["operations"][operation["resource"]]
        # critical x value (inside the uncertainty distribution), when the expected processing time would be too long
        # and collides with the planned start time of the next job or operation (stop signal must be fired in that case!)
        critical_x = operation["time_window_end"] - operation["start_time"] - ind_operation["min_duration"]
        worst_case_x = ind_operation["dist_max"]

        if worst_case_x <= critical_x:
            # continue, when there is enough buffer time between the distribution end and the start time of the next operation
            continue
        integr = compute_overlap(critical_x, worst_case_x, ind_operation["shape"], ind_operation["scale"])
        overlap_integrals.append(integr)
    integr_sum = np.sum(overlap_integrals)
    return integr_sum


def _get_distance_pattern(distances):
    def find_sequences(arr):
        sequences = []
        current_sequence = []
        for i in range(len(arr)):
            if i == 0 or arr[i] == arr[i - 1]:
                current_sequence.append(arr[i])
            else:
                sequences.append(current_sequence)
                current_sequence = [arr[i]]
        sequences.append(current_sequence)
        return sequences

    def calculate_sequence_stats(arr):
        sequences = find_sequences(arr)
        num_sequences = len(sequences)
        total_length = sum(len(seq) for seq in sequences)
        average_length = total_length / num_sequences
        return num_sequences, average_length

    return calculate_sequence_stats(distances)


def get_slack_time_to_next_job(this_pos, ind, df):
    if this_pos == len(ind)-1: # last job of schedule has no slack to the follower
        return 0,0
    df_job = df[df.job == ind[this_pos]["job"]]
    slack_mean = df_job.slack_next.mean()
    slack_std = df_job.slack_next.std()
    return slack_mean, slack_std


# @profile
def get_observation_features(schedule, df, ind, curr_job_pos=0):
    obs = {}
    job_ids = [j["job"] for j in ind]

    obs["n_jobs"] = len(ind)
    obs["n_machines"] = len(ind[0]["operations"])
    obs["worker_load"] = get_worker_workload(schedule, ma_window_size=3)
    obs["flowtime_sum"] = get_flowtime(schedule)
    obs["flowtime_mean"] = np.mean([get_flowtime(schedule, j) for j in job_ids])
    obs["flowtime_std"] = np.std([get_flowtime(schedule, j) for j in job_ids])
    obs["job_uncertainty_mean"] = _get_uncertainty_integralsum(schedule, ind) / len(job_ids)
    makespan = max([operation["end_time"] for operation in schedule])
    obs["makespan"] = makespan
    obs["distance_pattern_levenshtein"] = Levenshtein.distance(
        "".join(map(str, [j["distance"] for j in ind])), "".join(map(str, [1 for _ in job_ids]))
    )
    num_seq, len_seq = _get_distance_pattern([j["distance"] for j in ind])

    # obs["distance_pattern_num_sequences"] = num_seq/len(ind)
    obs["distance_pattern_mean_len_sequences"] = len_seq / len(ind)

    sum_processing_times = df.expected_duration.sum()
    # obs["utilization_rate"] = sum_processing_times / makespan
    obs["processing_sum"] = sum_processing_times
    obs["processing_mean"] = df.expected_duration.mean()
    obs["processing_std"] = df.expected_duration.std()

    # Remove the first machine operation (it has no predecessor operation slack time)
    mean_slack_machines = df[df["slack"] >= 0].groupby("resource")["slack"].mean()
    obs["slack_std_machines"] = np.std(mean_slack_machines)

    # skip the first scheduled job (it has no slack to a predecessor job)
    mean_slack_jobs = df[df["job"] != schedule[0]["job"]].groupby("job")["slack"].mean()
    obs["slack_std_jobs"] = np.std(mean_slack_jobs)
    # print(f"test_val {test_val}, obs {obs['slack_std_jobs']}")

    # obs["proc_std_jobs"] = np.std([df[df.job == job_idx].expected_duration.mean() for job_idx in range(len(ind["jobs"])) if job_idx != schedule[0]["job"]])
    df_same_machine = df[df.resource == df.last_res].copy()
    obs["slack_mean"] = df_same_machine.slack.mean()
    obs["slack_sum"] = df_same_machine.slack.sum()
    # obs["slack_std"] = df_same_machine.slack.std()
    df_same_machine["quantile"] = pd.qcut(df_same_machine.end_time, 4, labels=np.arange(0, 4))
    obs.update(
        {
            f"slack_sum_q{q+1}": slack
            for q, slack in enumerate(df_same_machine.groupby("quantile").agg("sum").slack.tolist())
        }
    )
    obs.update(
        {
            f"slack_std_q{q+1}": slack
            for q, slack in enumerate(df_same_machine.groupby("quantile").agg("std").slack.tolist())
        }
    )

    obs["has_pre_job"] = 1 if curr_job_pos > 0 else 0
    obs["has_post_job"] = 1 if curr_job_pos < len(ind) - 1 else 0
    for job_type, p in [("curr", curr_job_pos), ("pre", curr_job_pos - 1), ("post", curr_job_pos + 1)]:
        if p < 0 or p == len(ind):
            obs[f"{job_type}_high_distance"] = -1
            obs[f"{job_type}_uncertainty"] = -1
            obs[f"{job_type}_processing_mean"] = -1
            obs[f"{job_type}_processing_std"] = -1
            obs[f"{job_type}_slack_mean"] = -1
            obs[f"{job_type}_slack_std"] = -1
            # obs[f"{job_type}_high_slack_mean"] = 0
            # obs[f"{job_type}_high_slack_std"] = 0
            # obs[f"{job_type}_high_processing_mean"] = 0
            # obs[f"{job_type}_high_processing_std"] = 0
            # UNCFEAT obs[f"{job_type}_high_uncertainty"] = 0
            continue
        j = next(filter(lambda x: x["job"] == ind[p]["job"], ind))
        df_j = df[df.job == j["job"]]
        operations = [op for op in schedule if op["job"] == j["job"]]
        obs[f"{job_type}_processing_mean"] = df_j.expected_duration.mean()
        obs[f"{job_type}_processing_std"] = df_j.expected_duration.std()
        obs[f"{job_type}_high_distance"] = 1 if j["distance"] > 1 else 0
        obs[f"{job_type}_uncertainty"] = _get_uncertainty_integralsum(operations, ind)
        obs[f"{job_type}_slack_mean"], obs[f"{job_type}_slack_std"] = get_slack_time_to_next_job(p, ind, df)

        # obs[f"{job_type}_high_slack_mean"] = 1 if obs[f"{job_type}_slack_mean"] > obs["slack_mean"] else 0
        # obs[f"{job_type}_high_slack_std"] = 1 if obs[f"{job_type}_slack_std"] > obs["slack_std_jobs"] else 0
        # obs[f"{job_type}_high_processing_mean"] = (
        #     1 if obs[f"{job_type}_processing_mean"] > obs["processing_mean"] else 0
        # )
        # obs[f"{job_type}_high_processing_std"] = 1 if obs[f"{job_type}_processing_std"] > obs["processing_std"] else 0
        # UNCFEAT obs[f"{job_type}_high_uncertainty"] = 1 if obs[f"{job_type}_uncertainty"] > obs["job_uncertainty_mean"] else 0

        if job_type == "curr":
            obs["curr_pos_rel"] = (curr_job_pos + 1) / len(ind)
            obs["curr_flowtime"] = get_flowtime(schedule, job_id=j["job"])

    return obs

def _get_df_from_schedule(schedule):
    df = pd.DataFrame(schedule).sort_values(by=["resource", "end_time"])
    df["last_end"] = df.end_time.shift(1)
    df["next_start"] = df.start_time.shift(-1)
    df["last_res"] = df.resource.shift(1)
    df["next_res"] = df.resource.shift(-1)
    df["slack"] = df.start_time - df.last_end
    df["slack_next"] = df.next_start - df.end_time
    return df

def generate_schedule(ind):
    def get_start_time_for_curr_operation(curr_schedule, curr_machine):
        pre_operations = []
        for j in curr_schedule:
            pre_machine = curr_machine - 1
            if j["resource"] in (pre_machine, curr_machine):
                pre_operations.append(j)
        if not pre_operations:
            return 0
        end_times_of_pre_operations = [p["time_window_end"] for p in pre_operations]
        return max(end_times_of_pre_operations)

    def get_time_windows(ind):
        time_windows_per_job = {}
        # all jobs have the same window
        time_window_overall = max([op["expected_duration"] for job in ind for op in job["operations"]])
        for job in ind:
            time_windows_per_job[job["job"]] = time_window_overall * job["distance"]
        return time_windows_per_job
        # every job has its own window
        # for job in ind:
        #     max_processing_time = max([get_expected_proc_time(operation) for operation in job["operations"]])
        #     time_windows_per_job[job["job"]] = max_processing_time * job["distance"]
        # return time_windows_per_job

    time_window_per_job = get_time_windows(ind)
    schedule = []
    for job in ind:
        for machine_idx, operation_info in enumerate(job["operations"]):
            start_time = get_start_time_for_curr_operation(schedule, machine_idx)
            time_window_end = start_time + time_window_per_job[job["job"]]
            schedule.append(
                {
                    "start_time": start_time,
                    "end_time": start_time + operation_info["expected_duration"],
                    "time_window_end": time_window_end,
                    "job": job["job"],
                    "resource": machine_idx,
                    "expected_duration": operation_info["expected_duration"],
                }
            )
    return schedule, _get_df_from_schedule(schedule)


class FitnessCalculation:
    def __init__(
        self, fitness_features=["flowtime_sum", "worker_load", "job_uncertainty_mean"], fitness_weights=[0.4, 0.3, 0.3]
    ):
        self.fitness_features = fitness_features
        self.fitness_weights = fitness_weights
        self.log_of_all_schedules_evaluated = []
        self.initial_fitness_metrics = {}

    def set_fitness_score_and_log_schedule(self, ind, schedule, obs, df):
        if not self.initial_fitness_metrics:
            for feature in self.fitness_features:
                self.initial_fitness_metrics[feature] = obs[feature]
            self.log_of_all_schedules_evaluated.append(
                {"ind": ind, "schedule": schedule, "fitness_score": 1, "fitness_metrics": self.initial_fitness_metrics, "df": df}
            )
        else:
            fit_score = 0
            fit_metrics = {}
            for i, feature in enumerate(self.fitness_features):
                fit_score += (obs[feature] / self.initial_fitness_metrics[feature]) * self.fitness_weights[i]
                fit_metrics[feature] = obs[feature]
            self.log_of_all_schedules_evaluated.append(
                {"ind": ind, "schedule": schedule, "fitness_score": fit_score, "fitness_metrics": fit_metrics, "df": df}
            )
        return self.log_of_all_schedules_evaluated[-1]


if __name__ == "__main__":
    ind = get_taillard_with_uncert_proc_time("ta001")
    for _ in range(1000):
        random.shuffle(ind)
        schedule, df = generate_schedule(ind)
        # plot_gantt(schedule)
        obs = get_observation_features(schedule, df, ind, curr_job_pos=0)
        # print(obs)
    pass
