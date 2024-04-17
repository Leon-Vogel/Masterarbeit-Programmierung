import os

ROOT_DIR = os.path.dirname(__file__)
# ISRI_TENSORBOARD = os.path.join(ROOT_DIR, "isri_test_results", "tensorboard_log")
GTG_DATA_LEGACY = os.path.join(ROOT_DIR, "gtg_simulation", "flexsim_dummy", "data_for_metaheuristic")
GTG_DATA_FLEXSIM = os.path.join(ROOT_DIR, "miele_secret", "flexsim_dummy")
GTG_DATA = os.path.join(ROOT_DIR, "miele_secret", "simpy_model_data")
GTG_SIMULATION = os.path.join(ROOT_DIR, "gtg_simulation")
GTG_PAPER1 = os.path.join(ROOT_DIR, "gtg_utils", "paper")
TAILLARD_DATA_ISRI_PAPER2_ADAPT_PARAM = os.path.join(ROOT_DIR, "isri_utils", "paper2_adaptive_parameters", "taillard_data")
TAILLARD_DATA_ISRI_PAPER2_SHIFTING_JOBS = os.path.join(ROOT_DIR, "isri_utils", "paper2_shifting_jobs", "taillard_data")
TAILLARD_DATA_SHIFTING_AGENT_SHIFTING_JOBS = os.path.join(ROOT_DIR, "shifting_agent", "taillard_data")

# test