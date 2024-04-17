
from problem_factory import get_permutation_problem
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pickle
from ray.rllib.policy.policy import Policy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGENT_PATH = r"Y:\Documents\SUPPORT_AdaptiveHyperparameter\project_support\src\executable\isri_utils\paper2_adaptive_parameters\results\Trained_on_ta21_to_ta33_new_obs_space\trial_4f14e_00000\checkpoint_000074\policies\default_policy"
OUTPUT_PATH = AGENT_PATH + "../../../"
N_TRIES = 70
PROBLEM = 'pfssp'
TEST = 'performance'
EXPERIMENT_PATH = 'src/executable/isri_utils/paper2_adaptive_parameters/results/PermutationResults.csv'
RESULT_PATH = 'src/executable/isri_utils/paper2_adaptive_parameters/results/PermutationResults.csv'

def run_optimization(policy, test_env):
    done = False
    test_env.reset()
    observation = test_env._get_obs()
    steps = 0
    observations = []
    actions = []
    
    while not done:
        action = policy.compute_single_action(observation)[0]
        observation, reward, done, _, info = test_env.step(action)
        steps += 1
        observations.append(observation)
        actions.append(action)
    
    result = min(test_env.algorithm.pop.get('F'))
    return observations, actions, result

if __name__ == '__main__':
    policy = Policy.from_checkpoint(AGENT_PATH)
    instances = ['ta033', 'ta034', 'ta041']
    experiment_results = {
        'AdaptationMode': [],
        'Problem': [],
        'Instance': [],
        'TryNr': [],
        'RDMSeed': [],
        'MinValue': []
    }
    for instance in instances:
        test_env = get_permutation_problem(PROBLEM, instances=[instance])
        if TEST == 'explain':
            obs, acts, result = run_optimization(policy, test_env)

            with open(OUTPUT_PATH + 'observations.pkl', 'wb') as outfile:
                pickle.dump(obs, outfile)
            
            with open(OUTPUT_PATH + 'actions.pkl', 'wb') as outfile:
                pickle.dump(acts, outfile)

            # Plot
            actions = np.array(acts)
            observations = np.array(obs)
            labels = ['CX Prob', 'Mut Prob', 'PopSize', 'Pressure', 'Crossover Operation']
            fig, axs = plt.subplots(len(labels), 1)
            for i in range(test_env.action_space.shape[0]):
                axs[i].plot(np.arange(0, actions.shape[0], 1), actions[:, i])
                axs[i].set_title(labels[i])
            plt.show()
        
        else:
            for i in range(N_TRIES):
                obs, acts, result = run_optimization(policy, test_env)
                experiment_results['AdaptationMode'].append('rl')
                experiment_results['Problem'].append(PROBLEM)
                experiment_results['Instance'].append(instance)
                experiment_results['TryNr'].append(i)
                experiment_results['RDMSeed'].append(None)
                experiment_results['MinValue'].append(result[0])

    df = pd.DataFrame(experiment_results)
    existing_results = pd.read_csv(EXPERIMENT_PATH, sep=";", index_col=0)
    all_results = pd.concat([df, existing_results], axis=0).reset_index()
    all_results.to_csv(RESULT_PATH)

