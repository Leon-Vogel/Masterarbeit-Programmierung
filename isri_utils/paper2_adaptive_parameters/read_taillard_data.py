import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import path_definitions

def get_taillard(dataset):
    with open(os.path.join(path_definitions.TAILLARD_DATA_ISRI_PAPER2_ADAPT_PARAM, dataset), 'r') as file:
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

if __name__ == "__main__":
    test = get_taillard('ta011')
