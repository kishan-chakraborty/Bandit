import numpy as np
import matplotlib.pyplot as plt


class BanditEnvironment:
    def __init__(self, reward_distributions):
        self.reward_distributions = reward_distributions
        self.K = len(reward_distributions)

    def pull(self, arm):
        return self.reward_distributions[arm].pull()


def cal_regret(reward_distributions, hist):
    """
    Calculate the regret based on the optimal mean and the history of chosen arms.

    Parameters:
        opt_mean (float): The mean reward of the optimal arm.
        hist (list): A list of chosen arms.

    Returns:
        list: A list of cumulative regret values at each time step.
    """
    opt_mean = max([rd.mu for rd in reward_distributions])

    for t, arm in enumerate(hist):
        mean_reward = reward_distributions[arm].mu
        regret = opt_mean - mean_reward
        hist[t] = regret

    # Calculate cumulative regret
    cumulative_regret = np.cumsum(hist)
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
        plt.plot(
            np.arange(0, len(regret), marker_spacing),
            regret[::marker_spacing],
            marker=marker_style[i % len(marker_style)],
            linestyle="",
        )

    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret of Different Algorithms")
    plt.legend()
    # plt.show()
    return plt.gcf()
