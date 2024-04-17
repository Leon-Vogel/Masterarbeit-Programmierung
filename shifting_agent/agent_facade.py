import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import math
from evaluation_function import (
    FitnessCalculation,
    generate_schedule,
    get_observation_features,
)
from read_taillard_data import get_taillard_with_uncert_proc_time
from create_neighbor import (
    SelectNextJobAction,
    ShiftCurrentJobAction,
    ChangeCurrentJobDistanceAction,
    create_neighbor,
    select_new_job_pos
)


# ZUR INFO: Verwendung der API wird unten im Skript anhand eines Beispiels gezeigt

PAIN_OF_LIVING_PER_STEP = 0.7
# # massnahme für das credit assignment problem:
# # wenn der agent den schedule aus dem letzten step verbessert, gibt es eine belohnung
REWARD_FOR_IMPROVING_THE_LAST_SCHEDULE = 1
# # ... wenn er sogar den ursprungsschedule verbessert, gibt es eine noch höhere belohnung
REWARD_FOR_IMPROVING_THE_INITIAL_SCHEDULE = 10
# # wenn der agent am ende einen besseren plan als den ursprungsplan erzeugt hat,
# # gibt es eine sehr hohe belohnung mittels dieses multiplikators
REWARD_FACTOR_FOR_FINAL_IMPROVEMENT = 5
# # wenn der agent den schedule insgesamt nicht verbessern konnte, gibt es eine hohe bestrafung
PUNISHMENT_FOR_FINAL_DETERIORATION = 10
REWARD_DEFINITIONS = {
    "reward_for_improving_the_last_schedule": REWARD_FOR_IMPROVING_THE_LAST_SCHEDULE,
    "reward_for_improving_the_initial_schedule": REWARD_FOR_IMPROVING_THE_INITIAL_SCHEDULE,
    "reward_factor_for_final_improvement": REWARD_FACTOR_FOR_FINAL_IMPROVEMENT,
    "punishment_for_final_deterioration": PUNISHMENT_FOR_FINAL_DETERIORATION,
    "pain_of_living_per_step": PAIN_OF_LIVING_PER_STEP
}

# DEPRECATED, ggf. zum testen:
# ab der hälfte der episode wird der agent sukzessive bestraft, wenn er weiter shiftet.
# dies soll ihn motivieren, vor allem am anfang zu explorieren und später nicht mehr gewagte optimierungsversuche vorzunehmen.
# (vgl. trajektionsbasierte suchen wie simulated annealing)
# dies ist zwar ein semi-markoviansches reward-design, soll aber helfen, rechenzeit (simulationszeit) einzusparen.
# idealerweise braucht der agent dann nicht alle potenziellen N schritte
SUM_PAIN_OF_LIVING = 5


class AgentFacade:
    def __init__(
        self,
        individual_to_optimize,
        fitness_features = ["flowtime_sum", "worker_load", "job_uncertainty_mean"],
        fitness_weights = [0.4, 0.3, 0.3],
        episode_len=200,
        reward_definitions=REWARD_DEFINITIONS,
    ):
        # es gibt N maximale tausschritte pro episode
        self.episode_len = episode_len
        # im ersten step wird immer mit dem letzten job in der sequenz gestartet
        self.curr_job_pos = len(individual_to_optimize) - 1
        self.curr_ind = individual_to_optimize
        self.step_counter = 0
        self.fit_calc = FitnessCalculation(fitness_features, fitness_weights)
        # massnahme für das credit assignment problem:
        # wenn der agent den schedule aus dem letzten step verbessert, gibt es eine belohnung
        self.reward_for_improving_the_last_schedule = reward_definitions["reward_for_improving_the_last_schedule"]
        # ... wenn er sogar den ursprungsschedule verbessert, gibt es eine noch höhere belohnung
        self.reward_for_improving_the_initial_schedule = reward_definitions["reward_for_improving_the_initial_schedule"]
        # wenn der agent am ende einen besseren plan als den ursprungsplan erzeugt hat,
        # gibt es eine sehr hohe belohnung mittels dieses multiplikators
        self.reward_factor_for_final_improvement = reward_definitions["reward_factor_for_final_improvement"]
        # wenn der agent den schedule insgesamt nicht verbessern konnte, gibt es eine hohe bestrafung
        self.punishment_for_final_deterioration = reward_definitions["punishment_for_final_deterioration"]
        self.pain_of_living_per_step = reward_definitions["pain_of_living_per_step"]

    def _get_reward(self, done=False, has_ind_manipulated=True):
        R = 0

        def POL(x, N, k):
            # so ist es full-markovian
            return self.pain_of_living_per_step/N
            # so ist es semi:
            if x < N / 2 or x < 0 or x > N:
                return 0
            else:
                return (x / N) * (k / N) * math.e

        pain_of_living = POL(self.step_counter, self.episode_len, SUM_PAIN_OF_LIVING) if has_ind_manipulated else 0
        R -= pain_of_living
        curr_fit_score = self.fit_calc.log_of_all_schedules_evaluated[-1]["fitness_score"]
        last_fit_score = self.fit_calc.log_of_all_schedules_evaluated[-2]["fitness_score"]
        if curr_fit_score < last_fit_score:
            R += self.reward_for_improving_the_last_schedule / self.episode_len
        if curr_fit_score < 1:
            R += self.reward_for_improving_the_initial_schedule / self.episode_len
        if done:
            final_fitness_delta = (1 - curr_fit_score) * 100
            if final_fitness_delta <= 0:
                R -= self.punishment_for_final_deterioration
            else:
                R += self.reward_factor_for_final_improvement * final_fitness_delta
        return R

    def perform_action_and_get_new_observation_and_reward(
        self,
        shift_action=ShiftCurrentJobAction.DO_NOTHING,
        changedistance_action=ChangeCurrentJobDistanceAction.DO_NOTHING,
        selectjob_action=SelectNextJobAction.DO_NOTHING,
    ):
        ind_manipulation = shift_action + changedistance_action > 0
        assert self.step_counter < self.episode_len
        self.step_counter += 1
        if ind_manipulation:
            self.curr_ind = create_neighbor(self.curr_ind, self.curr_job_pos, shift_action, changedistance_action)
            schedule, df = generate_schedule(self.curr_ind)
        else: # keine simulation erforderlich, wenn der schedule nicht manipuliert wurde
            last_log = self.fit_calc.log_of_all_schedules_evaluated[-1]
            schedule = last_log["schedule"]
            df = last_log["df"]
        self.curr_job_pos = select_new_job_pos(selectjob_action, schedule, df, self.curr_ind, self.curr_job_pos)
        obs = get_observation_features(schedule, df, self.curr_ind, self.curr_job_pos)
        self.fit_calc.set_fitness_score_and_log_schedule(self.curr_ind, schedule, obs, df)
        done = self.step_counter == self.episode_len
        reward = self._get_reward(done, ind_manipulation)
        info = self.fit_calc.log_of_all_schedules_evaluated[-1]["fitness_metrics"]
        info["fitness_score"] = self.fit_calc.log_of_all_schedules_evaluated[-1]["fitness_score"]
        return obs, reward, done, info

    def get_first_observation_and_episode_len(self, ind):
        schedule, df = generate_schedule(self.curr_ind)
        obs = get_observation_features(schedule, df, self.curr_ind, self.curr_job_pos)
        self.fit_calc.set_fitness_score_and_log_schedule(ind, schedule, obs, df)
        return obs, self.episode_len


if __name__ == "__main__":
    # verwende die Shortest Processing Time Priorirätsregel als Eröffnungsverfahren
    ind = get_taillard_with_uncert_proc_time("ta001")
    ind = sorted(ind, key=lambda x: x["operations"][0]["expected_duration"])

    # instanziiere die Fassade (API), die vom gym-Env. für alle operationen
    # verwendet werden kann
    af = AgentFacade(ind)

    # am anfang episode / nach jedem reset muss folgendes aufgerufen werden:
    obs, n_steps = af.get_first_observation_and_episode_len(ind)

    # in der step-function dann jeweils der nachfolgende aufruf
    # es müssen die actions aus dem multidiskration action space übergeben werden
    obs, reward, done, info = af.perform_action_and_get_new_observation_and_reward(
        ShiftCurrentJobAction.SHIFT_LEFT,
        ChangeCurrentJobDistanceAction.SWITCH_DISTANCE,
        SelectNextJobAction.LOW_FLOWTIME,
    )
    
    pass
