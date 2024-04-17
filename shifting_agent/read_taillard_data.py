import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import path_definitions
import random
import pickle
import re
from scipy.stats import gamma


def get_taillard(dataset):
    with open(os.path.join(path_definitions.TAILLARD_DATA_SHIFTING_AGENT_SHIFTING_JOBS, dataset), 'r') as file:
        contents = file.read()
    rows = contents.split("\n")
    n_jobs = len(rows[0].split())
    n_machines = len(rows)
    jobs = []
    for j in range(n_jobs):
        curr_job = []
        for m in range(n_machines):
            operation_proc_time = int(rows[m].split()[j])
            curr_job.append(operation_proc_time)
        jobs.append(curr_job)
    return jobs

def get_taillard_with_uncert_proc_time(dataset="ta001"):
    filename = os.path.join(path_definitions.TAILLARD_DATA_SHIFTING_AGENT_SHIFTING_JOBS, f"{dataset}_uncert.pkl")
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def init_taillard_testdata_with_uncertain_proc_time(dataset):
    data_determ = get_taillard(dataset)
    uncertainty_scenarios = [ # check here: https://homepage.divms.uiowa.edu/~mbognar/applets/gamma.html
        {
            "shape": 1, # minimum uncertainty
            "scale": 8
        },
        {
            "shape": 1.5, # medium
            "scale": 10
        },
        {
            "shape": 2, # high
            "scale": 14
        }
    ]

    for sc in uncertainty_scenarios:
        sc["dist_max"] = gamma.ppf(0.9999, sc["shape"], scale=sc["scale"])

    jobs_new = []
    for i, job in enumerate(data_determ):
        curr_job = {"job": i, "operations": [], "distance": 1}
        for operation_duration in job:
            scenario = random.choices(uncertainty_scenarios, weights=[0.5, 0.35, 0.15], k=1)[0]
            expected_duration = operation_duration + scenario["shape"] * scenario["scale"]
            curr_job["operations"].append({
                'min_duration': operation_duration,
                'shape': scenario["shape"],
                'scale': scenario["scale"],
                'dist_max': scenario["dist_max"],
                'expected_duration': expected_duration
            })
        jobs_new.append(curr_job)

    filename = os.path.join(path_definitions.TAILLARD_DATA_SHIFTING_AGENT_SHIFTING_JOBS, f"{dataset}_uncert.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(jobs_new, file)

if __name__ == "__main__":
    default_instances = os.listdir(path_definitions.TAILLARD_DATA_SHIFTING_AGENT_SHIFTING_JOBS)
    default_instances = [inst for inst in default_instances if re.match(r"^ta\d{3}$", inst)]
    default_instances = ["ta035", "ta036", "ta037", "ta038", "ta039", "ta040"]
    for inst in default_instances:
        test = init_taillard_testdata_with_uncertain_proc_time(inst)
    test = get_taillard_with_uncert_proc_time("ta001")
    pass
