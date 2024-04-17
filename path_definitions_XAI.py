import os

ROOT_DIR = os.path.dirname(__file__)
# ISRI_TENSORBOARD = os.path.join(ROOT_DIR, "isri_test_results", "tensorboard_log")
GTG_DATA_LEGACY = os.path.join(ROOT_DIR, "gtg_simulation", "flexsim_dummy", "data_for_metaheuristic")
GTG_DATA = os.path.join(ROOT_DIR, "miele_secret") #changed to miele_secret
GTG_CONFIGS = os.path.join(ROOT_DIR, "gtg_optimizer", "rl", "configs")
GTG_SIMULATION = os.path.join(ROOT_DIR, "gtg_simulation")
GTG_PAPER1 = os.path.join(ROOT_DIR, "gtg_utils", "paper")
TAILLARD_DATA_ISRI_PAPER2 = os.path.join(ROOT_DIR, "isri_utils", "paper2", "taillard_data")