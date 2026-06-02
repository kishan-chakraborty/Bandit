import matplotlib.pyplot as plt
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


def cal_stochastic_regret(mean_rewards: list, chosen_arms: list) -> np.ndarray:
    """
    Args:
        mean_rewards: [K] array of mean rewards for each arm.
        chosen_arms: [T] array of arms chosen by the algorithm at each time step.
    return:
        cumulative_regret: [T] array of cumulative regret at each time step.
    """
    max_mean = max(mean_rewards)  # Optimal mean reward
    per_step_regret = np.array(
        [max_mean - mean_rewards[arm] for arm in chosen_arms]
    )  # Per step regret
    cumulative_regret = np.cumsum(
        per_step_regret
    )  # Cumulative regret at each time step
    return cumulative_regret
