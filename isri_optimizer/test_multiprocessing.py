from _dir_init import *
import pickle
import configparser
import pyodbc
from isri_metaheuristic_simple import *
import datetime
from isri_simulation.isri_simulation import Simulation
from copy import deepcopy

config_path = "./config.ini"
config = configparser.ConfigParser()
config.read(config_path)

conn_string = f"Driver={config['CONNECTION_DB']['Driver']};\
    Server={config['CONNECTION_DB']['Server']};\
    Database={config['CONNECTION_DB']['Database']};\
    Trusted_Connection=no;\
    UID={config['CONNECTION_DB']['Username']};\
    PWD={config['CONNECTION_DB']['Password']}"
isri_con = pyodbc.connect(conn_string)
time_now = datetime.datetime.now()
simulation = Simulation(conn_str=conn_string, print_subjects=[], start_time=time_now,
    print_options={"General": False, "Production": False})

simcopy = deepcopy(simulation)
print('deepcopy')
# Test Pickle und Entpickle
with open('test_simulation_pickle', 'wb') as sim_file:
    pickle.dump(simulation, sim_file)

with open('test_simulation_pickle', 'rb') as sim_file:
    sim_loaded = pickle.load(sim_file)