from _dir_init import *
import gym
from gym.spaces import Box, MultiDiscrete
import numpy as np
from random import choice
from isri_optimizer.isri_metaheuristic_simple import get_schedule_kpis, get_due_date_random_chromosome
from misc_utils import tensorboard_caller
import path_definitions

import ray



class ISRIGymEnvFelix01(gym.Env):
    def __init__(self, n_observation_features, n_actions, max_steps, train_data_full, eval_func):
        self.action_space = MultiDiscrete([n_actions[0],n_actions[1],n_actions[2],n_actions[3],n_actions[4]])
        self.n_observation_features = n_observation_features
        self.observation_space = Box(
            np.array([-sys.float_info.max] * n_observation_features),
            np.array([sys.float_info.max] * n_observation_features),
        )
        self.global_steps = 0
        self.train_data_full = train_data_full
        self.eval_func = eval_func
        self.tensorboard_called = False

    def reset(self):
        self.done = False
        self.episode_steps = 0
        self.max_ep_steps = 100
        self.action_log = []
        self.sample = choice(self.train_data_full)
        self.ind = get_due_date_random_chromosome(self.sample['jobs_per_line'], self.sample['job_list'], n_random_switches=0)


    def step(self, action):
        self.global_steps += 1
        self.episode_steps += 1

        act_select_job_by_metric = action[0] # select by proc. time | tardiness | slack time
        act_select_by_value = action[1] # high | low (metric value, e.g. high tardiness)
        act_select_by_start_time = action[2] # early | late (in the schedule)
        act_select_by_resource_util = action[3] # high utilized resource | low utilized resource
        act_shift_job = action[4] # do later | do earlier | assign to low utilized resource

        if self.episode_steps == self.max_ep_steps:
            self.done = True
            _print_episode_final_infos(0)
        else:
            pass

        return list(schedule_info.values()), reward, self.done, {}


    def _evaluate_and_get_state():
        schedule_info = get_schedule_kpis(self.ind, self.sample['job_list'], self.sample['jobs_per_line'], self.sample['conveyor_speed'], self.sample['n_machines'])
        return schedule_info
    

    def _call_tensorboard(self):
        if self.tensorboard_called == False:
            tensorboard_caller.open_newest(path_definitions.ISRI_TENSORBOARD)
            self.tensorboard_called = True


    def _print_episode_final_infos(self, end_rew):
        actions = list(map(lambda x: x[0], self.action_log))
        interim_rewards = sum(list(map(lambda x: x[1], self.action_log)))
        action_std = np.std(actions)
        percent = f"{self.global_steps / self.n_steps * 100:.2f}%"
        print(
            f"{percent}  |  Makespan={self.sim.get_kpis()[SimKpi.CMAX]}/{self.curr_sample.kpi_list[SimKpi.CMAX]}  |  TotalTardiness={self.sim.get_kpis()[SimKpi.TARDINESS]}/{self.curr_sample.kpi_list[SimKpi.TARDINESS]}  |  EndRew={end_rew:6.2f}  |  InterRew={interim_rewards:5.3f}  |  ActionStd={action_std:4.3f} ({actions})"
        )



"""
class TaskSequencingGymEnv(gym.Env):
    def __init__(self, samples_data, n_obs_features, n_actions, n_steps, w=0.5):
        #self.action_space = Discrete(n_actions)
        self.action_space = MultiDiscrete([n_actions[0],n_actions[1]])
        self.n_observation_features = n_obs_features
        self.n_steps = n_steps
        self.observation_space = Box(
            np.array([-sys.float_info.max] * n_obs_features),
            np.array([sys.float_info.max] * n_obs_features),
        )
        self.samples_data = samples_data
        self.global_steps = 0
        self.w = w  # weight for cmax. tmax weight = 1-w
        self.results = []
        self.tb_called = False
        self.best_solution_found_value = 0
        self.best_solution_found = [None,None]

    def __call_tensorboard(self):
        if self.tb_called == False:
            tbc.open_newest()
            self.tb_called = True

    def reset(self):
        self.done = False
        self.episode_steps = 0
        samples_collection = choice(self.samples_data)
        self.multi_objective: MultiObjective
        self.multi_objective = fast_deepcopy(samples_collection["multi_obj"])
        self.master_data: MasterDataContainer
        self.master_data = samples_collection["master_data"]
        self.curr_sample: IndividualWrapper
        self.curr_sample = fast_deepcopy(choice(samples_collection["samples"]))
        self.action_log = []
        self.throughput_last = 0

        # the agent is trained on the BOTTLENECK station per episode
        # utils = self.curr_sample.kpi_list["station_utils"]
        # n_tasks = self.curr_sample.kpi_list["station_n_tasks"]
        # stations = utils.keys()
        # def norm(all_data, val):
        #     return (val - min(all_data.values()))/(max(all_data.values())-min(all_data.values()))
        # bottleneck_station_indicators = {s : norm(utils, utils[s]) + norm(n_tasks, n_tasks[s]) for s in stations}
        # bottleneck_station_no = max(bottleneck_station_indicators.items(), key=operator.itemgetter(1))[0]
        # station_for_training = next(filter(lambda x: x.no == bottleneck_station_no, self.master_data.stations))
        # station_for_training = choice(self.master_data.stations) # TODO: not just a random station?

        stations_for_training = self.curr_sample.get_bottleneck_stations(self.master_data, 1.0)

        self.sim = Simulation(
            self.curr_sample.genome,
            self.master_data,
            disp_rule_per_station=self.curr_sample.dispatching_mode,
            print_sim=False,
            print_interval=1,
            slow_mode=False,
            rl_mode=ReinforcementLearningMode.LEARNING,
            rl_stations=stations_for_training,
        )
        self.sim_ctrl = SimGymController(self.sim, self, set_state)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self.sim_ctrl.start_sim)
        self.state = self.sim_ctrl.get_next_state_for_gym()
        self.last_state = self.state

        return list(self.state["statedict"].values())

    def step(self, action):
        self.__call_tensorboard()
        self.global_steps += 1
        self.episode_steps += 1

        dispatching_rule = self.sim.convert_action(action[0])
        resource_flip = int(action[1])

        if self.sim.all_tasks_processed:
            self.done = True
            reward = self._calc_end_rew()
            self._print_episode_final_infos(reward)
        else:
            # set the state for the next step
            self.sim_ctrl.set_input_from_gym(self.state["station"], dispatching_rule, resource_flip)
            self.last_state = fast_deepcopy(self.state)
            self.state = self.sim_ctrl.get_next_state_for_gym()
            reward = self._calc_inter_rew(action, dispatching_rule, resource_flip)

        return list(self.state["statedict"].values()), reward, self.done, {}

    def _calc_end_rew(self):
        old_kpis = self.curr_sample.kpi_list
        new_kpis = self.sim.get_kpis()
        old_obj_value = self.multi_objective.get_obj_value(old_kpis)
        new_obj_value = self.multi_objective.get_obj_value(new_kpis)

        if new_obj_value < self.best_solution_found_value:
            self.best_solution_found_value = new_obj_value
            self.best_solution_found = self.sim.get_kpis()

        diff = old_obj_value - new_obj_value
        rew = diff * self.episode_steps * self.episode_steps * 20

        return rew

    def _calc_inter_rew(self, action, disp_rule, flip):
        try:
            last = self.last_state["statedict"]
            curr = self.state["statedict"]

            ir = 0
            if flip == 2 and last['two_tasks_ready'] != 1:
                ir = -3
            elif flip == 2 and last['wip'] > last['wip_all_mean']:
                ir = 2
            elif flip == 1 and last['worker_wip_rel']/last['slots_rel'] > 1:
                ir = 1
            if disp_rule == DispatchingMode.SLACK and last['slack_mean_this'] < last['slack_mean_all']:
                ir += 1
            #if disp_rule == DispatchingMode.MTWR:
            if curr["throughput_mean_all"] > last["throughput_mean_all"]:
                ir += 3
            if curr["slack_min_this"] > 0:
                ir += 3
            self.action_log.append([(action[0], action[1]), ir])
            return ir
        finally:
            self.action_log.append([(action[0], action[1]), ir])

    def _print_episode_final_infos(self, end_rew):
        actions = list(map(lambda x: x[0], self.action_log))
        interim_rewards = sum(list(map(lambda x: x[1], self.action_log)))
        action_std = np.std(actions)
        percent = f"{self.global_steps / self.n_steps * 100:.2f}%"
        print(
            f"{percent}  |  Makespan={self.sim.get_kpis()[SimKpi.CMAX]}/{self.curr_sample.kpi_list[SimKpi.CMAX]}  |  TotalTardiness={self.sim.get_kpis()[SimKpi.TARDINESS]}/{self.curr_sample.kpi_list[SimKpi.TARDINESS]}  |  EndRew={end_rew:6.2f}  |  InterRew={interim_rewards:5.3f}  |  ActionStd={action_std:4.3f} ({actions})"
        )

"""