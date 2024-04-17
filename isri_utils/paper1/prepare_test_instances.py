import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from isri_optimizer.isri_metaheuristic_simple import read_ISRI_Instance
import pickle
import os


if __name__ == '__main__':
    # Parameter
    path_mjd = 'C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/mtm_info.csv'
    instance_path = 'C:/Users/lukas/Documents/SUPPORT/Isringhausen/Daten/BatchExport'
    stunden_nach_rohbau = 4 # Deadline für Produkte mit Rohbausignal ist Jetzt + Abstand zum neuest Rohbausignal + stunden_nach_rohbau
    stunden_nach_lack = 1.5 # Deadline für Produkte mit Lacksignal ist Jetzt + Abstand zum neuest Lacksignal + stunden_nach_lack
    stunden_nach_reo = 0.8 # Deadline für Reo ist Jeztz + stunden_nach_reo
    min_per_lud = 3 # Für LUD Produkte gibt es kein Rohbau oder Lacksignal. Alle min_per_lud Minuten ist die Deadline für ein LUD Produkt, sortiert nach Plansequenz
    next_n = 400 # Schneidet die next_n Produkte ab -> gleich große Instanzen und reduziert unnötig großen Backlog
    save_path = 'C:/Users/lukas/Documents/SUPPORT/AzureDevOps/project_sup_support/src/executable/isri_optimizer/isri_instances'
    every_n_instance = 1


    files = os.listdir(instance_path)[::every_n_instance]
    instances = []
    for file in files:
        print(f'Reading: {file}')
        jobs = read_ISRI_Instance(instance_path + "/" + file, path_mjd, stunden_nach_rohbau, stunden_nach_lack,
                                   stunden_nach_reo, min_per_lud, next_n)
        jobnew = {}
        for i, key in enumerate(jobs):
            jobnew[f"Job {i}"] = jobs[key]
            instances.append(jobnew)

    with open(save_path, 'wb') as outfile:
        pickle.dump(instances, outfile)
