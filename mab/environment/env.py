import numpy as np


class Agent:
    def __init__(self, algo):
        """
        Initialize the agent who will play using a MAB algorithm.

        Args:
            algo: The MAB algorithm that the agent will use to select actions.
        """
        self.algo = algo

        # Initialized during experimentation.
        self.reward_hist = None
        self.action_hist = None
        self.regret = None

    def select_action(self):
        """
        For iteration t, select an action, observe the reward and update the algo.

        Returns:
            action: The index of the arm to pull.
        """
        action = self.algo.select_action()
        return action

    def reset(self):
        """
        Reset the agent to the initial state.
        """
        self.algo.reset()


class Environment:
    """
    Create a iid MAB environment.
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
        self.rng = np.random.default_rng(seed)  # Local random number generator per env.

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
