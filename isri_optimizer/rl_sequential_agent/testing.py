from _dir_init import *
from sb3_contrib.ppo_mask import MaskablePPO
from isri_optimizer.rl_sequential_agent.environment import IsriEnv
from data_preprocessing import IsriDataset
from isri_optimizer.rl_sequential_agent.training import env_config, JOBDATA_DIR, N_TRAINING_INSTANCES, GA_SOLUTIONS_PATH
from isri_optimizer.isri_metaheuristic_simple import fast_sim_diffsum
import pickle
import random


MODEL_FILE = "./isri_optimizer/rl_sequential_agent/savefiles/ISRI_Agent_NextJobs/shorter_lookahead_1000000.zip"
N_TESTS = 5


def test_episode(model, dataset: IsriDataset, params: dict, instance: int = None):
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
    isri_dataset = pickle.load(open(GA_SOLUTIONS_PATH, 'rb'))
    model = MaskablePPO.load(MODEL_FILE)
    isri_dataset.data['Jobdata'] = isri_dataset.data['Jobdata'][N_TRAINING_INSTANCES:]
    isri_dataset.data['Files'] = isri_dataset.data['Files'][N_TRAINING_INSTANCES:]
    isri_dataset.data['GAChromosome'] = isri_dataset.data['GAChromosome'][N_TRAINING_INSTANCES:]
    isri_dataset.data['GAFitness'] = isri_dataset.data['GAFitness'][N_TRAINING_INSTANCES:]
    isri_dataset.data_size = len(isri_dataset.data['Files'])
    for test_idx in range(N_TESTS):
        test_episode(model, isri_dataset, env_config)

