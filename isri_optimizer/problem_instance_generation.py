from _dir_init import *
from isri_metaheuristic_simple import read_ISRI_Instance
import numpy as np
import pandas as pd
import pyodbc
from datetime import datetime, timedelta
import configparser
import pickle
from tqdm import tqdm
pd.options.mode.chained_assignment = None

PATH = "C:/1_Services/0001_KI_Support/data/"
N_SAMPLES = 500
CONFIG_PATH = './config.ini'

instance_files = os.listdir(PATH)
random_choices = np.random.choice(instance_files, N_SAMPLES)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
conn_string = f"Driver={config['CONNECTION_DB']['Driver']};\
    Server={config['CONNECTION_DB']['Server']};\
    Database={config['CONNECTION_DB']['Database']};\
    Trusted_Connection=no;\
    UID={config['CONNECTION_DB']['Username']};\
    PWD={config['CONNECTION_DB']['Password']}"
isri_con = pyodbc.connect(conn_string)
mjd = pd.read_sql("""
    SELECT
        materialnumber,
        operationnumber,
        sum(mtmtgTotal) as total_time
    FROM materialjobdata
    GROUP BY materialnumber, operationnumber""", isri_con)

for idx, file in tqdm(enumerate(random_choices)):
    data = pd.read_csv(PATH + file)
    time_str = file.split("_")[1]
    year = int(time_str[:4])
    month = int(time_str[4:6])
    day = int(time_str[6:8])
    hour = int(time_str[9:11])
    minute = int(time_str[11:13])
    # print(year, month, day, hour, minute)
    date = datetime(year, month, day, hour, minute)
    try:
        jobdata = read_ISRI_Instance(data, date, mjd.copy(deep=True))
    except:
        pass
    with open(f'instances/instance_{idx}', 'wb') as outfile:
        pickle.dump([jobdata, date], outfile)
