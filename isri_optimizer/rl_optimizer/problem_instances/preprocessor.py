import numpy as np
from isri_optimizer.isri_metaheuristic_simple import read_ISRI_Instance
import os


class Instance:
    """
    Jobdata example dict of entries: 23070621922: {'times': [111.06,
   101.22,
    ...
   88.07999999999998,
   158.09999999999997],
  'due_date': 24300.0},
    """
    def __int__(self, jobdata, jpl, n_stations=2):
        self.jobdata = jobdata
        self.jobs_per_line = jpl
        self.n_stations = n_stations


def load_instance(path):
    with open(path, 'rb') as infile:
        instance = pickle.load(infile)
    return instance
