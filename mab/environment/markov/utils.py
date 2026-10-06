from env import Environment
import numpy as np


class Arm:
    def __init__(self, id, states, trans_matrix, st_dist, rewards, cur_state):
        """
        Args:
            states: States of the MC [len: d]
            trans_matrix: Trnasition matrix. [dim: dxd]
            st_dist: Stationary distribution of the MC [len d].
            rewards: Rewards corresponding to each state of the MC [len d].
            curr_state: Current state the MC is in.
        """
        self.id = id
        self.states = states
        self.trans_matrix = trans_matrix
        self.st_dist = st_dist
        self.rewards = rewards
        self.mean = None
        if cur_state is None:
            cur_state = np.random.choice(states)
        self.curr_state = cur_state
        self.n_states = len(states)

    def reward(self):
        """
        Reward corresponding to the arm.
        """
        raise NotImplementedError(
            f"Reward function for arm{self.id} is not implemented."
        )


class MarkovEnvironment(Environment):
    def __init__(self, arms, seed) -> None:
        mean_rewards = [arm.mean for arm in arms]
        super().__init__(mean_rewards, seed)

    def step(self, action: int) -> float:
        """
        Take an action and return the reward.

        Args:
            action: int
                The index of the arm to pull.

        Returns: int
            The reward obtained from pulling the arm 0 and 1.
        """
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def oracle(self):
        """
        Return the oracle information.
        """
        result = {
            "best_arm": self.best_arm,
            "best_mean": self.best_mean,
            "mean_rewards": self.mean_rewards,
        }
        return result
