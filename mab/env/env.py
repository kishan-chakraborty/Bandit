import sys
from pathlib import Path

current_dir = Path.cwd()
sys.path.append(str(current_dir.parent))

import numpy as np


class Environment:
    """
    Create a MAB environment.
    """

    def __init__(self, mean_rewards: list, seed: int, **kwargs):
        """
        Stochastic MAB environment:
            r ~ prob_distribution(mu_i)
        Args:
            mean_rewards: list of mean rewards for each arm.
            seed: the random seed for reproducibility.
        """
        self.n_arms = len(mean_rewards)
        self.mean_rewards = mean_rewards

        self.best_arm = np.argmax(mean_rewards)
        self.best_mean = mean_rewards[self.best_arm]
        self.rng = np.random.default_rng(seed)
        self.reset()

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

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.action_hist = []  # List to store the action history.
        self.reward_hist = []  # List to store the reward history.

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

    def cal_regret(self):
        """
        Compute the cumulative regret based on the action history and reward history.
        """
        raise NotImplementedError(
            "The compute_regret method must be implemented by subclasses."
        )
