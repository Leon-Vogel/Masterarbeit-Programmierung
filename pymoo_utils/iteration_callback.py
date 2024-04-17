from _dir_init import *
import time
import warnings
from pprint import pprint
import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.variable import Real
try:
    from isri_utils.paper2_adaptive_parameters.rl_agent import get_observation, decode_action
except ModuleNotFoundError:
    # Passiert während des Trainings der Agenten weil die von RLLIB gespawnten Worker die Pfadänderungen nicht berücksichtigen
    pass


class EachIterationCallback(Callback):  # called in each generation/iteraion
    def __init__(self, max_evals, adaptive_rate_mode="none", print_iteration_stats=True,
                 agent=None, save_stats=None):
        self.max_evals = max_evals
        self.print_iteration_stats = print_iteration_stats
        self.adaptive_rate_mode = adaptive_rate_mode
        self.start = time.time()
        self.mean_hist = []
        self.min_hist = []
        self.agent = agent
        self.adaptation_steps = 0
        self.save_stats = save_stats
        self.iteration_calls = 0
        if adaptive_rate_mode == 'rl':
            assert agent is not None, "rl Adaptation is specified, but no RL-Agent provided"
            
        super().__init__()

    def update(self, algorithm):
        """
        icdm: (linearly) increasing crossover & decreasing mutation
        imdc: decr. mutation, incr. crossover
        """

        def get_val(var):
            if isinstance(var, Real):
                return var.value
            elif isinstance(var, float):
                return var
            elif isinstance(var, np.float32):
                return var

        def set_val(var, val):
            if isinstance(var, Real):
                var.value = val
                return var
            elif isinstance(var, float):
                return val
            elif isinstance(var, np.float32):
                return val

        if self.adaptive_rate_mode == "icdm":
            algorithm.mating.crossover.prob = set_val(
                algorithm.mating.crossover.prob, algorithm.evaluator.n_eval / self.max_evals
            )
            algorithm.mating.mutation.prob = set_val(
                algorithm.mating.mutation.prob, 1.0 - get_val(algorithm.mating.crossover.prob)
            )

        elif self.adaptive_rate_mode == "imdc":
            algorithm.mating.mutation.prob = set_val(
                algorithm.mating.mutation.prob, algorithm.evaluator.n_eval / self.max_evals
            )
            algorithm.mating.crossover.prob = set_val(
                algorithm.mating.crossover.prob, 1.0 - get_val(algorithm.mating.mutation.prob)
            )
        
        elif self.adaptive_rate_mode == "rl":
            # Observation
            observation = get_observation(algorithm, self.min_hist, self.mean_hist)
            
            # Action TODO 5 Steps per Parameter
            if self.adaptation_steps % 5 == 0:
                action = self.agent.compute_single_action(observation)[0]
                crossover, mutation, pressure = decode_action(action)
                    
                # Set Params accordingly
                algorithm.mating.crossover.prob = set_val(
                    algorithm.mating.crossover.prob, crossover
                )
                algorithm.mating.mutation.prob = set_val(
                    algorithm.mating.mutation.prob, mutation
                )
                # algorithm.pop_size = pop_size
                # algorithm.n_offsprings = pop_size

                algorithm.mating.selection.pressure = pressure
            
        elif self.adaptive_rate_mode != "none" and self.adaptive_rate_mode != "static" and self.adaptive_rate_mode:
            raise Exception(f"{self.adaptive_rate_mode} is not defined.")
        
            
        if self.print_iteration_stats:
            fitnesses_by_ind = list(algorithm.pop.get("F"))
            fitness_means = []
            fitness_mins = []
            fitness_std = []
            for i in range(len(fitnesses_by_ind[0])):
                all_fitness_values_of_a_objective = [row[i] for row in fitnesses_by_ind]
                fitness_means.append(np.mean(all_fitness_values_of_a_objective))
                fitness_mins.append(min(all_fitness_values_of_a_objective))
                fitness_std.append(np.std(all_fitness_values_of_a_objective))
            print(
                "%s s | EVALS: %s | CXPB: %s, MUTPB: %s | MEAN FITNESSES: %s | MIN FITNESSES: %s"
                % (
                    str(round(time.time() - self.start, 2)),
                    algorithm.evaluator.n_eval,
                    get_val(algorithm.mating.crossover.prob),
                    get_val(algorithm.mating.mutation.prob),
                    fitness_means,
                    fitness_mins,
                )
            )
        
        if self.save_stats is not None:
            fmean_str = [str(f) for f in fitness_means]
            fmin_str = [str(f) for f in fitness_mins]
            fstd_str = [str(f) for f in fitness_std]
            if self.iteration_calls > 0:
                with open(self.save_stats, 'a') as outfile:
                    outfile.write(",".join(fmean_str) + "," + ",".join(fmin_str) + "," + ",".join(fstd_str) + "\n")
            else:
                with open(self.save_stats, 'w') as outfile:
                    outfile.write(",".join(fmean_str) + "," + ",".join(fmin_str) + "," + ",".join(fstd_str) + "\n")
        
        self.iteration_calls += 1

