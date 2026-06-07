import matplotlib.pyplot as plt
import numpy as np


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
