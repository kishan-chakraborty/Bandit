import matplotlib.pyplot as plt


def plot_regret(regret_dict, n_markers=20):
    """
    Plot the cumulative regret for different algorithms.
    Parameters:
        regret_dict (dict): algo_name -> cumulative_regret
        n_markers (int): Number of markers to show on the plot for better visibility.
    """
    plt.figure(figsize=(10, 6))

    # Plot marker at regular intervals
    marker_style = ["o", "*", "^", "s", "D"]
    for i, (algo, regret) in enumerate(regret_dict.items()):
        marker_spacing = len(regret) // n_markers
        plt.plot(
            regret,
            label=algo,
            marker=marker_style[i % len(marker_style)],
            markevery=marker_spacing,
        )

    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret of Different Algorithms")
    plt.legend()
    plt.show()
    return plt.gcf()
