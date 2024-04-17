from _dir_init import *
import numpy as np
import pandas as pd
import pickle
import warnings
import configparser
from datetime import timedelta, datetime
from isri_simulation.isri_simulation import Siminfo, Simulation, Product
from pymoo_utils.algorithm_runner import run_alg_parallel, run_alg
warnings.simplefilter(action='ignore', category=FutureWarning)
from isri_metaheuristic_simple import ISRIProblem, ISRIMutation, ISRICrossover, ISRIDuplicateElimination, ISRISampling, get_random_chromosome, split_jobs_on_lines, read_ISRI_Instance
pd.set_option('mode.chained_assignment', None)

path = "./isri_optimizer/"
# path = "./"
if __name__ == '__main__':
    config_path = path + 'config.ini'
    """
    with open(path + 'instances/' + "instance_45", 'rb') as jb_file:
        jobdata = pickle.load(jb_file)
        time = jobdata[1]
        jobdata = jobdata[0]
        jobdata = {int(job_id): jobdata[job_id] for job_id in jobdata.keys()}
        jobdata = {job_id: {'times': [round(t, 4) for t in jobdata[job_id]['times']], 'due_date': jobdata[job_id]['due_date']} for job_id in jobdata.keys()}
    """
    instance = pd.read_csv("C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/BatchExport/NemetrisExportBatch_20230223-154500.csv", sep=",")
    #instance = pd.read_csv("C:/Users/fgrumbach1/Desktop/isri_testdata/NemetrisExportBatch_20230217-143003.csv", sep=",")
    mjd = pd.read_csv("C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/mtm_info.csv", sep=";")
    #mjd = pd.read_csv("C:/Users/fgrumbach1/Desktop/isri_testdata/mtm_info.csv", sep=";")
    mjd.columns = ["material", "op", "mtm_time"]
    mjd['mtm_time'] = mjd['mtm_time'].str.replace(',', '.').astype(float)
    dt = datetime(2023, 2, 23, 15, 45)
    jobdata = read_ISRI_Instance(instance=instance, date_time=dt, mjd=mjd)

    with open(path + 'siminfo_test.pkl', 'rb') as si_file:
        siminfo = pickle.load(si_file)
    
    # Test Optimisation
    simulation = siminfo
    sim = Simulation(simulation)
    config = configparser.ConfigParser()
    config.read(config_path)
    n_jobs = len(jobdata)
    n_machines = siminfo.n_machines
    param = {
            "jobs": jobdata,
            "problem_getter": ISRIProblem,
            "crossover": ISRICrossover(config['GA'].getfloat('cx_rate')),
            "mutation": ISRIMutation(config['GA'].getfloat('mut_rate')),
            "sampling": ISRISampling(),
            "eliminate_duplicates": ISRIDuplicateElimination(),
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "n_lines": config['GA'].getint('n_lines'), # config['GA'].getint('n_lines'),
            "pop_size": config['GA'].getint('pop_size'),
            "jobs_per_line": config['GA'].getint('jobs_per_line'),
            "alg_name": config['GA']['alg_name'],  # nsga2 | moead | rvea | ctaea | unsga3
            "obj_weights": [config['GA'].getfloat('weight_workload'), config['GA'].getfloat('weight_deadline')],  # 1: min geglättete belastung, max minimalen due date puffer, , config['GA'].getfloat('weight_group_constraint')]
            "conveyor_speed": config['GA'].getint('conveyor_speed'),
            "ma_window_size": config['GA'].getint('ma_window_size'),
            "n_evals": config['GA'].getint('n_evals'), # number of fitness calculations,
            "seed": config['GA'].getint('seed'),
            "parallelization_method": 'multi_processing', # multi_threading | multi_processing | dask
            "n_parallelizations": 6, # number threads or processes,
            "print_results": config['GA'].getboolean('print_results'),
            "print_iterations": config['GA'].getboolean('print_iterations'), # number threads or processes,
            "adaptive_rate_mode": config['GA']['adaptive_rate_mode'], # none | icdm | imdc | rl
            "criterion": config['GA']['criterion'], # 'minmax' | 'minavg' | 'minvar',
            'plot_solution': config['GA'].getboolean('plot_solutions'),
            'use_sim': config['GA'].getboolean('use_sim'),
            'simstart': siminfo.start_time,
            'simulation': simulation,
            'agent': None, 
            'train_agent': config['GA'].getboolean('train_agent'),
            'reward_func': None
        }

    # Optimierung
    problem = ISRIProblem(param, runner=None)
    res, best_chromosome, best_fitness = run_alg_parallel(run_alg, param)
    split_plan = split_jobs_on_lines(best_chromosome, config['GA'].getint('jobs_per_line'), config['GA'].getint('n_lines'))
    table_a = [jobdata[nr]["times"] + [jobdata[nr]['due_date']] + [nr] for nr in split_plan[0]]
    table_b = [jobdata[nr]["times"] + [jobdata[nr]['due_date']] + [nr] for nr in split_plan[1]]
    df_a = pd.DataFrame(table_a)
    df_b = pd.DataFrame(table_b)
    columns_plan = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOL Mech', 'EOL_elektr.', 'Deadline', 'Produktionsnummer']
    df_a.columns = columns_plan
    df_b.columns = columns_plan
    # instance_data_rel = instance_data[['produktionsnummer', 'seqtermin', 'seqnr', 'edi399', 'tr_zsbtyp', 'auidnr', 'zielwerk', 'auftragsart']]
    # df_a = df_a.merge(instance_data_rel, left_on='Produktionsnummer', right_on='auidnr', how='left')
    # df_b = df_b.merge(instance_data_rel, left_on='Produktionsnummer', right_on='auidnr', how='left')
    # df_a.drop(columns=['produktionsnummer'], inplace=True)
    # df_b.drop(columns=['produktionsnummer'], inplace=True)
    df_a.to_json(path + 'PlanA.json')
    df_b.to_json(path + 'PlanB.json')
    # df_a = df_a.to_json()
    # df_b = df_b.to_json()
