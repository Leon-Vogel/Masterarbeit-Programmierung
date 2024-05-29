import pickle
from data_preprocessing import IsriDataset

"""
Hier werden die Isri Daten als Liste gespeichert. Stable Baselines3 Multiprocessing führt die sb3_training.py datei nochmal aus - aber in einem anderen Namespace
Dadurch sind alle importierten Klassen nicht mehr richtig zugeordnet ... Python Klassen wie listen gehen aber
"""

DATASET_PATH = "isri_optimizer/rl_sequential_agent/data/IsriDataset_100_jobs.pkl"
DICT_PATH = "isri_optimizer/rl_sequential_agent/data/IsriDataDict_100_jobs.pkl"

isri_dataset = pickle.load(open(DATASET_PATH, 'rb'))
data = isri_dataset.data
with open(DICT_PATH, "wb") as outfile:
    pickle.dump(data, outfile)