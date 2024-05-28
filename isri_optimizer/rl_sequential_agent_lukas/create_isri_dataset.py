from _dir_init import *
import numpy as np
import pickle
from pymoo_utils.algorithm_runner import run_alg_parallel, run_alg
from misc_utils.copy_helper import fast_deepcopy
from isri_optimizer.isri_metaheuristic_simple import *
import os
from datetime import datetime
import multiprocessing
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Any

RAW_DATA_PATH = "C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/BatchExport"
MJD_PATH = "C:/Users/lukas/Documents/SUPPORT/Isringhausen/mjd.csv"
MULTIPROCESSING = True
N_JOBS = 20
PREPROCESSED_FILES_PATH = f"./isri_optimizer/instances/"



GA_PARAMS = {
            "problem_getter": ISRIProblem,
            "crossover": ISRICrossover(0.8),
            "mutation": ISRIMutation(0.8),
            "sampling": ISRISampling(),
            "eliminate_duplicates": ISRIDuplicateElimination(),
            "n_jobs": N_JOBS,
            "n_machines": 12,
            "n_lines": 1,
            "pop_size": 50,
            "jobs_per_line": N_JOBS,
            "alg_name": 'nsga2',  # nsga2 | moead | rvea | ctaea | unsga3
            "obj_weights": [0.8, 0.2],
            "conveyor_speed": 208,
            "ma_window_size": 4,
            "n_evals": 10000,  # number of fitness calculations,
            "seed": 123,
            "parallelization_method": 'multi_processing',  # multi_threading | multi_processing | dask
            "n_parallelizations": 6,  # number threads or processes,
            "print_results": False,
            "print_iterations": False, 
            "adaptive_rate_mode": 'none',  # none | icdm | imdc | rl
            "criterion": 'minmax',  # 'minmax' | 'minavg' | 'minvar',
            'plot_solution': False,
            'use_sim': False,
            'simstart': 'NOW',
            'simulation': None,
            'agent': None, 
            'train_agent': False,
            'reward_func': None,
            'save_path_stats': None
        }


class IsriDataset(Dataset):
    """
    Isri Dataset
    """

    def __init__(self, data_size, seq_len, solve=True):
        self.data_size = 0
        self.seq_len = seq_len
        self.solve = solve
        self.data = {'Jobdata': [], 'GASolutions': [], 'Files': [], 'GAChromosome': [], 'GAFitness': []}

    def __len__(self):
        return self.data_size
    
    def add_item(self, jobdata: np.ndarray, file: str, best_chromosome: np.ndarray, best_fitness):
        self.data['Jobdata'].append(jobdata)
        self.data['Files'].append(file)
        self.data['GAChromosome'].append(best_chromosome)
        self.data['GAFitness'].append(best_fitness)
        self.data_size += 1


def preprocess_data(params):
    """Read Raw Data Files and get random samples of the jobs to be on one line"""
    file = params[0]
    mjd = params[1]
    data = pd.read_csv(RAW_DATA_PATH + "/" + file, sep=",", header=0)
    timestamp_str = file.split("_")[1].split(".")[0]
    timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
    jobdata = read_ISRI_Instance(data, timestamp, mjd)
    jobs = find_next_jobs_by_due_date(len(jobdata), 1, jobdata)
    jobs_a = jobs[::2]
    jobs_b = jobs[1::2]
    jobdata_a = {job: jobdata[job] for job in jobs_a}
    jobdata_b = {job: jobdata[job] for job in jobs_b}

    with open(PREPROCESSED_FILES_PATH + file.split(".")[0] + "_a.pkl", "wb") as outfile:
        pickle.dump(jobdata_a, outfile)

    with open(PREPROCESSED_FILES_PATH + file.split(".")[0] + "_b.pkl", "wb") as outfile:
        pickle.dump(jobdata_b, outfile)


def plot_barplots(chromosome, jobdata):

    def plot_werkerlast_bar(data: np.ndarray, deadlines: list, title: str, ax):
        colors = map_to_rgb(deadlines)
        ax.bar(np.arange(0, data.shape[0]), height=data, width=0.5, color=colors, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('Produktreihenfolge')
        ax.set_ylabel('MTM Zeit pro Produkt')
    
    fig, axs = plt.subplots(6, 2)
    for idx, ax in enumerate(list(axs.flatten())):
        data = np.array([jobdata[job_id]['times'][idx] for job_id in chromosome])
        deadlines = np.array([jobdata[job_id]['due_date'] for job_id in chromosome])
        plot_werkerlast_bar(data, deadlines, f"AP {idx}", ax)
    
    plt.show()


def map_to_rgb(array: np.ndarray):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    cmap = plt.colormaps.get_cmap('RdYlGn')
    colors = cmap(normalized_array)
    return colors


def to_one_hot(sequence_idx, max_idx):
    one_hot = np.zeros(max_idx)
    one_hot[sequence_idx] = 1
    return one_hot


if __name__ == '__main__':

    preprocessed_files = os.listdir(PREPROCESSED_FILES_PATH)
    
    data = IsriDataset(len(preprocessed_files), seq_len=GA_PARAMS['n_jobs'], solve=False)

    for file in tqdm.tqdm(preprocessed_files, desc="Processing files"):
        try:
            with open(PREPROCESSED_FILES_PATH + file, "rb") as infile:
                jobdata = pickle.load(infile)
                sorted_jobdata = list(
                        dict(sorted(jobdata.items(), key=lambda item: item[1]["due_date"])).keys()
                    )
                jobdata = {key: value for key, value in jobdata.items() if key in sorted_jobdata[:N_JOBS]}
            GA_PARAMS['jobs'] = jobdata
            res, best_chromosome, best_fitness = run_alg_parallel(run_alg, GA_PARAMS)
            job_list = [job for job in jobdata.keys()] # Damit Reihenfolge klar ist
            # plot_barplots(best_chromosome, jobdata)
            assert np.all([job_id in jobdata.keys() for job_id in best_chromosome]), 'Chromosome passt nicht zum Problem?!'
            data.add_item(jobdata, file, best_chromosome, best_fitness)
        except Exception as E:
            print(f"Error in file {file}: {E}")
    
    with open(f"IsriDataset_{N_JOBS}_jobs.pkl", "wb") as outfile:
        pickle.dump(data, outfile)

