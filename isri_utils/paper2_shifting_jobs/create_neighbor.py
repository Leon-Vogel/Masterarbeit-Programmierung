import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import math
from isri_utils.paper2_shifting_jobs.evaluation_function import get_flowtime, get_worker_workload, get_slack_time_to_next_job
from misc_utils.copy_helper import fast_deepcopy
import numpy as np
import random 

class ShiftCurrentJobAction:
    """1. action im multidiskreten raum: soll der aktuelle job
    in der produktionsreihenfolge verschoben werden?"""

    DO_NOTHING = 0
    SHIFT_RIGHT = 1
    SHIFT_LEFT = 2


class ChangeCurrentJobDistanceAction:
    """2. action im multidiskreten raum: soll der bockabstand
    am hängeförderer verändert werden?"""

    DO_NOTHING = 0
    SWITCH_DISTANCE = 1


class SelectNextJobAction:
    """3. action im multidiskreten raum: soll der job für die nächste action
    beibehalten werden oder soll ein neuer zu shiftender job
    gem. einer heuristik ausgewählt werden?"""

    DO_NOTHING = 0
    HIGH_FLOWTIME = 1
    HIGH_SLACK = 2
    LOW_FLOWTIME = 3
    LOW_SLACK = 4
    MIN_SLACK = 5
    MAX_SLACK = 6
    FIRST_JOB = 7
    LAST_JOB = 8


DISTANCES = [1, 1.02]

def create_neighbor(ind, jobpos, shift, changedistance):
    ind = fast_deepcopy(ind)
    if changedistance == ChangeCurrentJobDistanceAction.SWITCH_DISTANCE:
        assert len(DISTANCES)==2, "currently, only two distances are selectable"
        job = ind[jobpos]
        switch = list(set(DISTANCES).difference([job["distance"]]))
        job["distance"] = switch[0]
    if shift == ShiftCurrentJobAction.DO_NOTHING:
        return ind
    if shift == ShiftCurrentJobAction.SHIFT_LEFT:
        neighbor_pos = jobpos - 1
    elif shift == ShiftCurrentJobAction.SHIFT_RIGHT:
        neighbor_pos = jobpos + 1 if jobpos < len(ind)-1 else 0
    ind[jobpos], ind[neighbor_pos] = ind[neighbor_pos], ind[jobpos]
    return ind


def create_neighbor_random(ind, jobpos):
    changedistance = random.randint(0, max(value for name, value in vars(ChangeCurrentJobDistanceAction).items() if not name.startswith("__")))
    shift = random.randint(0, max(value for name, value in vars(ShiftCurrentJobAction).items() if not name.startswith("__")))
    #jobpos = random.randint(0, len(ind)-1)
    return create_neighbor(ind, jobpos, shift, changedistance)


def select_new_job_pos(selectjob, schedule, df, ind, curr_pos):
    if selectjob == SelectNextJobAction.DO_NOTHING:
        return curr_pos
    def get_total_proctime(j):
        job = next(filter(lambda x: x["job"]==j, ind))
        return sum([o["expected_duration"] for o in job["operations"]])
    job_ids = [j["job"] for j in ind]
    FT = {j:get_flowtime(schedule, j) for j in job_ids}
    #PT = {j:get_total_proctime(j) for j in job_ids}
    job_ids_except_last_scheduled_job = job_ids[:-1]
    ST = {j:get_slack_time_to_next_job(i, ind, df)[0] for i,j in enumerate(job_ids_except_last_scheduled_job)}
    #ST_std = {j:get_slack_time_to_next_job(i, ind, df)[1] for i,j in enumerate(job_ids)}

    if selectjob == SelectNextJobAction.HIGH_FLOWTIME:
        J = [(j,ft) for j, ft in FT.items() if ft >= np.percentile(list(FT.values()), 70)]
    elif selectjob == SelectNextJobAction.LOW_FLOWTIME:
        J = [(j,-ft) for j, ft in FT.items() if ft <= np.percentile(list(FT.values()), 30)]
    elif selectjob == SelectNextJobAction.HIGH_SLACK:
        J = [(j,pt) for j, pt in ST.items() if pt >= np.percentile(list(ST.values()), 70)]
    elif selectjob == SelectNextJobAction.LOW_SLACK:
        J = [(j,-pt) for j, pt in ST.items() if pt <= np.percentile(list(ST.values()), 30)]
    elif selectjob == SelectNextJobAction.MAX_SLACK:
        J = [(max(ST, key=ST.get),0)]
    elif selectjob == SelectNextJobAction.MIN_SLACK:
        J = [(min(ST, key=ST.get),0)]
    elif selectjob == SelectNextJobAction.FIRST_JOB:
        J = [(job_ids[0],0)]
    elif selectjob == SelectNextJobAction.LAST_JOB:
        J = [(job_ids[-1],0)]
    elif selectjob != SelectNextJobAction.DO_NOTHING:
        raise Exception("action not implemented")
    J = [j for j,_ in sorted(J, key=lambda x: x[1])]
    weights = [i**0.5 for i in range(1, len(J) + 1)]
    next_job = random.choices(J, weights=weights, k=1)[0]
    for pos, job in enumerate(ind):
        if job["job"] == next_job:
            return pos