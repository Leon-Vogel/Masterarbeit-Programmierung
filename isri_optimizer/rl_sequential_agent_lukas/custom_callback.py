from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        self.is_tb_set = False
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.workload_hist = deque(maxlen=100)
        self.deadline_hist = deque(maxlen=100)
        self.balance_punishement_hist = deque(maxlen=100)
        self.reward_hist = deque(maxlen=100)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        dones = self.locals['dones']
        for idx in range(dones.shape[0]):
            if dones[idx]:
                if isinstance(self.locals['env'], SubprocVecEnv):
                    for reward in list(self.locals['rewards']):
                        self.reward_hist.append(reward)
                    self.logger.record("reward_mean", np.mean(self.reward_hist))
                else:
                    env = self.locals['env'].envs[idx].env
                    self.deadline_hist.append(env.deadline_gap)
                    self.workload_hist.append(env.workload_gap)
                    self.balance_punishement_hist.append(env.balance_punishement)
                    self.logger.record('deadline_gap', np.mean(self.deadline_hist))
                    self.logger.record('workload_gap', np.mean(self.workload_hist))
                    self.logger.record('balance_punishement', np.mean(self.balance_punishement_hist))

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Hier noch saven
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass