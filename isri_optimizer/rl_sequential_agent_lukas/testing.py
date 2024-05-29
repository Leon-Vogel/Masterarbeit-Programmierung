from _dir_init import *
from sb3_contrib.ppo_mask import MaskablePPO
from isri_optimizer.rl_sequential_agent.environment import IsriEnv
from data_preprocessing import IsriDataset
from isri_optimizer.rl_sequential_agent.sb3_training import env_config, JOBDATA_DIR, N_TRAINING_INSTANCES, GA_SOLUTIONS_PATH
from isri_optimizer.isri_metaheuristic_simple import fast_sim_diffsum
from pymoo_utils.algorithm_runner import run_alg_parallel, run_alg
from isri_optimizer.isri_metaheuristic_simple import *
import pickle
import random
import pandas as pd
import time
from isri_optimizer.rl_sequential_agent.create_isri_dataset import GA_PARAMS


MODEL_FILE = "isri_optimizer/rl_sequential_agent/savefiles/mlp_model_350_instances_with_csv_500000_1.zip"
N_TESTFILES = 50
DATASET_PATH = 'isri_optimizer/rl_sequential_agent/data/IsriDataDict_50_jobs.pkl'

def play_episode(model, isri_dataset, params):
    params['isri_dataset'] = isri_dataset
    env = IsriEnv(params)
    obs, info = env.reset()
    done = False
    while not done:
        action_mask = env.action_masks()
        action, _states = model.predict(obs.reshape(1, obs.shape[0]), deterministic=True, action_masks=action_mask)
        obs, rewards, done, truncated, info = env.step(action)
    return env.genome

def test_episode_render(model, dataset: IsriDataset, params: dict, instance: int = None):
    dummy_dataset = IsriDataset(data_size=1, seq_len=None)
    if instance is None:
        instance = random.choice(list(range(N_TRAINING_INSTANCES, dataset.data_size)))
    dummy_dataset.add_item(dataset.data['Jobdata'][instance], dataset.data['Files'][instance],
                           dataset.data['GAChromosome'][instance], dataset.data['GAFitness'][instance])
    params['isri_dataset'] = dummy_dataset
    env = IsriEnv(params)
    obs, info = env.reset()
    done = False
    while not done:
        action_mask = env.action_masks()
        action, _states = model.predict(obs, deterministic=True, action_masks=action_mask)
        obs, rewards, done, truncated, info = env.step(action)
    rl_plan = env.genome
    ga_plan = dummy_dataset.data['GAChromosome'][0]
    assert len(set(rl_plan).intersection(set(ga_plan))) == len(rl_plan), 'Unterschiedliche Jobs verplant!'
    rl_fitness = fast_sim_diffsum(rl_plan, env.jobdata, env.jpl, env.conv_speed, env.n_machines, n_lines=1, window_size=env.window_size)
    ga_fitness = fast_sim_diffsum(ga_plan, env.jobdata, env.jpl, env.conv_speed, env.n_machines, n_lines=1, window_size=env.window_size)
    print(f"RL Ergebnisse: {rl_fitness}")
    print(f"GA Ergebnisse: {ga_fitness}")
    env.render()

if __name__ == '__main__':

    # Load the Data
    isri_dataset = pickle.load(open(DATASET_PATH, 'rb'))
    model = MaskablePPO.load(MODEL_FILE)
    test_ids = np.random.choice(list(range(len(isri_dataset['Jobdata']))), N_TESTFILES)
    test_dataset = {
        'Jobdata': [],
        'Files': [],
        'GAChromosome': [],
        'GAFitness': []
    }
    for test_id in test_ids:
        test_dataset['Jobdata'].append(isri_dataset['Jobdata'][test_id])
        test_dataset['Files'].append(isri_dataset['Files'][test_id])
        test_dataset['GAChromosome'].append(isri_dataset['GAChromosome'][test_id])
        test_dataset['GAFitness'].append(isri_dataset['GAFitness'][test_id])

    # Test RL Agent
    results = {
        "instance_id": [],
        "file": [],
        "ga_tardiness": [],
        "rl_tardiness": [],
        "ga_diffsum": [],
        "rl_diffsum": [],
        "time_in_seconds": []
    }

    for test_id in range(len(test_dataset['Jobdata'])):
        data_copy = {
            'Jobdata': [test_dataset['Jobdata'][test_id]],
            'Files': [test_dataset['Files'][test_id]],
            'GAChromosome': [test_dataset['GAChromosome'][test_id]],
            'GAFitness': [test_dataset['GAFitness'][test_id]]
        }
        t_start = time.time()
        genome = play_episode(model, data_copy, env_config)
        t_end = time.time()
        time_in_seconds = t_end - t_start
        assert len(set(genome).intersection(set(data_copy['GAChromosome'][0]))) == len(genome), 'Unterschiedliche Jobs verplant!'
        rl_fitness = fast_sim_diffsum(genome, data_copy['Jobdata'][0], len(genome),
                                      env_config["conv_speed"], env_config["n_machines"],
                                      n_lines=1, window_size=env_config["window_size"])
        ga_fitness = fast_sim_diffsum(data_copy['GAChromosome'][0], data_copy['Jobdata'][0], len(genome),
                                      env_config["conv_speed"], env_config["n_machines"], n_lines=1,
                                      window_size=env_config["window_size"])
        results['instance_id'].append(test_id)
        results['file'].append(data_copy['Files'][0])
        results['ga_tardiness'].append(ga_fitness[1])
        results['rl_tardiness'].append(rl_fitness[1])
        results['ga_diffsum'].append(ga_fitness[0])
        results['rl_diffsum'].append(rl_fitness[0])
        results['time_in_seconds'].append(time_in_seconds)
    
    df = pd.DataFrame(results)
    df.to_csv("./isri_optimizer/rl_sequential_agent/Test_Results_RL_50.csv", sep=";")

    # Run GA for Statistics, Time Comparison
    times = []
    for test_id in range(len(test_dataset['Jobdata'])):
        n_jobs = len(test_dataset['Jobdata'][test_id])
        GA_PARAMS['jobs'] = test_dataset['Jobdata'][test_id]
        GA_PARAMS['save_path_stats'] = f'isri_optimizer/rl_sequential_agent/savefiles/ga_stats/ga_{test_id}_{n_jobs}_jobs.csv'
        GA_PARAMS['n_jobs'] = n_jobs
        GA_PARAMS['jobs_per_line'] = n_jobs
        # GA_PARAMS['print_iterations'] = True
        t_start = time.time()
        res, best_chromosome, best_fitness = run_alg_parallel(run_alg, GA_PARAMS)
        t_end = time.time()
        time_in_seconds = t_end - t_start
        times.append(time_in_seconds)
    
    with open(f'isri_optimizer/rl_sequential_agent/savefiles/ga_stats/ga_times_{n_jobs}.pkl', "wb") as outfile:
        pickle.dump(times, outfile)


