from _dir_init import *
import configparser
from datetime import datetime
import numpy as np
from multiprocessing import freeze_support
import pandas as pd
import psycopg2
import pyodbc
from isri_metaheuristic_simple import *
from pymoo_utils.algorithm_runner import run_alg, run_alg_parallel
import warnings
from isri_simulation.isri_simulation import Siminfo
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def save_results(data, line_name, workload_matrix, db_connection):
    pass

def isri_optimize(config):
    freeze_support()

    # Aufträge aus Nemetris auslesen
    nm_con = psycopg2.connect(
        host=config['CONNECTION_NEMETRIS']['host'],
        database=config['CONNECTION_NEMETRIS']['database'],
        user=config['CONNECTION_NEMETRIS']['user'],
        password=config['CONNECTION_NEMETRIS']['password'])

    conn_string = f"Driver={config['CONNECTION_DB']['Driver']};\
        Server={config['CONNECTION_DB']['Server']};\
        Database={config['CONNECTION_DB']['Database']};\
        Trusted_Connection=no;\
        UID={config['CONNECTION_DB']['Username']};\
        PWD={config['CONNECTION_DB']['Password']}"
    isri_con = pyodbc.connect(conn_string)


    sqlSelect = \
        "select to_char( to_timestamp ( wautab.istbegduz ,'YYYYMMDDHH24MISS' ),'YYYY.MM.DD HH24:MI:SS' ) as Freigabe, wautab.tr_vin as Produktionsnummer, \
        to_char( to_timestamp ( wautab.tr_istseqdatuz ,'YYYYMMDDHH24MISS' ),'YYYY.MM.DD HH24:MI:SS' ) as SeqTermin, \
        wautab.tr_istseqno as SeqNr,to_char( to_timestamp ( wautab.tr_steukz11 ,'YYYYMMDDHH24MISS' ),'YYYY.MM.DD HH24:MI:SS' ) as EDI399, \
        wautab.tr_autotyp as Autotyp,wautab.tr_zsbtyp,wautab.tr_cvinbelegnr, wautab.auidnr,wautab.tr_plantloc as zielwerk, wautab.auart as Auftragsart, \
        trvarchecktab.variant as Variant \
        from wautab \
        inner join trvarchecktab \
        on trvarchecktab.variantid = wautab.variantid \
        where wautab.finr = 100 \
        and wautab.werk = '0001' \
        and wautab.status in ('10')  \
        and wautab.tr_zsbtyp in ('DN','ES') \
        and wautab.tr_usageloc in ('037','037') \
        order by to_char( to_timestamp ( wautab.tr_steukz11 ,'YYYYMMDDHH24MISS' ),'YYYY.MM.DD HH24:MI:SS' ),to_char( to_timestamp ( wautab.istbegduz ,'YYYYMMDDHH24MISS' ),'YYYY.MM.DD HH24:MI:SS' ),  wautab.tr_vin "  

    try:
        instance_data = pd.read_sql(sqlSelect, nm_con)
        # instance_data = pd.read_csv("C:/1_Services/0001_KI_Support/data/NemetrisExportBatch_20230220-230000.csv", sep=",", header=0)
        # instance_data.produktionsnummer = instance_data.produktionsnummer.str.replace(" ", "")
        instance_data.to_csv(config['GA']['savepath_nemetris_data']) #savepath_nemetris_data
        nm_con.close()
    except psycopg2.DatabaseError as e:
        print("Data export unsuccessful.")
        print(e)
        quit()
    finally:
        nm_con.close() 

    mjd = pd.read_sql("""
        SELECT
            materialnumber,
            operationnumber,
            sum(mtmtgTotal) as total_time
        FROM materialjobdata
        GROUP BY materialnumber, operationnumber""", isri_con)

    time_now = datetime.now()
    # Formatierung der Ergebnisse
    instance_data['produktionsnummer'] = instance_data['produktionsnummer'].str.strip().astype(int)
    instance_data['auidnr'] = instance_data['auidnr'].str.strip().astype('Int64')

    jobdata = read_ISRI_Instance(instance_data, time_now, mjd,
        config['DEADLINES'].getfloat('stunden_nach_rohbau'), 
        config['DEADLINES'].getfloat('stunden_nach_lack'),
        config['DEADLINES'].getfloat('stunden_nach_reo'),
        config['DEADLINES'].getfloat('min_per_lud'),
        config['DEADLINES'].getint('next_n'))

    # Vorbereitung Simulation und Metaheuristik
    # jobdata = {int(job_id.replace(" ", "")): jobdata[job_id] for job_id in jobdata.keys()}
    simulation = Siminfo(conn_str=conn_string, print_subjects=[], start_time=time_now,
        print_options={"General": False, "Production": False})

    n_jobs = len(jobdata)
    n_machines = simulation.n_machines
    param = {
            "jobs": jobdata,
            "problem_getter": ISRIProblem,
            "crossover": ISRICrossover(config['GA'].getfloat('cx_rate')),
            "mutation": ISRIMutation(config['GA'].getfloat('mut_rate')),
            "sampling": ISRISampling(),
            "eliminate_duplicates": ISRIDuplicateElimination(),
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "n_lines": config['GA'].getint('n_lines'),
            "pop_size": config['GA'].getint('pop_size'),
            "jobs_per_line": config['GA'].getint('jobs_per_line'),
            "alg_name": config['GA']['alg_name'],  # nsga2 | moead | rvea | ctaea | unsga3
            "obj_weights": [config['GA'].getfloat('weight_workload'), config['GA'].getfloat('weight_deadline')],  # 1: min geglättete belastung, max minimalen due date puffer
            "conveyor_speed": config['GA'].getint('conveyor_speed'),
            "ma_window_size": config['GA'].getint('ma_window_size'),
            "n_evals": config['GA'].getint('n_evals'),  # number of fitness calculations,
            "seed": config['GA'].getint('seed'),
            "parallelization_method": config['GA']['parallelization_method'],  # multi_threading | multi_processing | dask
            "n_parallelizations": config['GA'].getint('n_parallelizations'),  # number threads or processes,
            "print_results": config['GA'].getboolean('print_results'),
            "print_iterations": config['GA'].getboolean('print_iterations'),  # number threads or processes,
            "adaptive_rate_mode": config['GA']['adaptive_rate_mode'],  # none | icdm | imdc | rl
            "criterion": config['GA']['criterion'],  # 'minmax' | 'minavg' | 'minvar',
            'plot_solution': config['GA'].getboolean('plot_solutions'),
            'use_sim': config['GA'].getboolean('use_sim'),
            'simstart': time_now,
            'simulation': simulation,
            'agent': None, 
            'train_agent': config['GA'].getboolean('train_agent'),
            'reward_func': None,
            'save_path_stats': config['GA']['save_path_stats']
        }

    # Optimierung
    res, best_chromosome, best_fitness = run_alg_parallel(run_alg, param)

    # Speichern der Ergebnisse
    split_plan = split_jobs_on_lines(best_chromosome, config['GA'].getint('jobs_per_line'), config['GA'].getint('n_lines'))
    # workload_matrix, tardiness, constraint_violations = get_fitness(best_chromosome, jobdata, conv_speed=208,
        # n_machines=config['GA'].getint('n_lines'), jpl=config['GA'].getint('jobs_per_line'),
        # use_sim=True, sim=simulation, simstart=datetime.now(),
        # return_value='full_workload')

    for line in range(config['GA'].getint('n_lines')):
        table = [jobdata[nr]["times"] + [jobdata[nr]['due_date']] + [nr] for nr in split_plan[line]]
        df = pd.DataFrame(table)
        columns_plan = ['OP11', 'OP12', 'OP13', 'OP14', 'OP15', 'OP31', 'OP32', 'OP33', 'OP34', 'OP35', 'EOLMech', 'EOL_elektr.', 'Deadline', 'auidnr']
        # columns_workload = ['workload_OP11', 'workload_OP12', 'workload_OP13', 'workload_OP14', 'workload_OP15', 'workload_OP31', 'workload_OP32',
        # 'workload_OP33', 'workload_OP34', 'workload_OP35', 'workload_EOLMech', 'workload_EOL_elektr.']
        # workload_start = line * 12
        # workload_end = (line + 1) * 12
        # workload = pd.DataFrame(workload_matrix[workload_start:workload_end, :].T, columns=columns_workload)
        df.columns = columns_plan
        instance_data_rel = instance_data[['produktionsnummer', 'seqtermin', 'seqnr', 'edi399', 'tr_zsbtyp', 'auidnr', 'zielwerk', 'auftragsart', 'autotyp']]
        df = df.merge(instance_data_rel, left_on='auidnr', right_on='auidnr', how='left')
        # df = pd.concat((df, workload), axis=1)
        name = ALPHABET[line]
        df['linie'] = name
        df.to_json(f"{config['GA']['savepath']}_{name}")
        df = df.to_xml()
        with isri_con.cursor() as cur:
            cur.execute("insert into Result (Datum, Result, Description) VALUES (?, ?, ?)",
                [pd.Timestamp.now(), df, f'Results {name}'])
            cur.commit()
    
    isri_con.close()
    return best_fitness


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("isri_optimizer/config.ini")
    isri_optimize(config)
