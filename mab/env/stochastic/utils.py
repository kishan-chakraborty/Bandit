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


def plot_regret(regret_dict, marker_spacing=20):
    """
    Plot the cumulative regret for different algorithms.
    Parameters:
        regret_dict (dict): algo_name -> cumulative_regret
    """
    plt.figure(figsize=(10, 6))
    for algo, regret in regret_dict.items():
        plt.plot(regret, label=algo)

    # Plot marker at regular intervals
    marker_style = ["o", "*", "^", "s", "D"]
    for i, (algo, regret) in enumerate(regret_dict.items()):
        plt.plot(regret, label=algo)

    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret of Different Algorithms")
    plt.legend()
    return plt.gcf()
