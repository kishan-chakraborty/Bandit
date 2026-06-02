import matplotlib.pyplot as plt

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