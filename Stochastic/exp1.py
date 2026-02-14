import numpy as np
from Algorithms.ucb import UCB
from Algorithms.kl_ucb import KLUCB
from Algorithms.epsilon_t_greedy import EpsilonTGreedy
from Rewards.gaussian import GaussianReward
from utils import cal_regret, plot_regret, BanditEnvironment

# Experiment to compare various stochastic bandit algorithms

# Define the experiment parameters
num_arms = 5
time_horizon = 1000
num_simulations = 100

normal_means = [1, 2, 3, 4, 5]  # True means for each arm
arms = [GaussianReward(mu=normal_means[i], sigma=1) for i in range(num_arms)]

env = BanditEnvironment(arms)

# Store mean rewards of number of simulations for each algorithm
ucb_rewards = [0] * time_horizon
kl_ucb_rewards = [0] * time_horizon
epsilon_t_greedy_rewards = [0] * time_horizon

ucb_arms = [0] * time_horizon
kl_ucb_arms = [0] * time_horizon
epsilon_t_greedy_arms = [0] * time_horizon

for i in range(num_simulations):
    ucb = UCB(num_arms)
    kl_ucb = KLUCB(num_arms, 1.0)
    epsilon_t_greedy = EpsilonTGreedy(num_arms)

    for t in range(time_horizon):
        chosen_arm = ucb.select_arm()
        reward = env.pull(chosen_arm)
        ucb.update(chosen_arm, reward)

        # Store the running mean reward for each time step
        ucb_rewards[t] = (ucb_rewards[t] * i + reward) / (i + 1)
        ucb_arms[t] = chosen_arm

        chosen_arm = kl_ucb.select_arm()
        reward = env.pull(chosen_arm)
        kl_ucb.update(chosen_arm, reward)

        kl_ucb_rewards[t] = (kl_ucb_rewards[t] * i + reward) / (i + 1)
        kl_ucb_arms[t] = chosen_arm

        chosen_arm = epsilon_t_greedy.select_arm()
        reward = env.pull(chosen_arm)
        epsilon_t_greedy.update(chosen_arm, reward)

        epsilon_t_greedy_rewards[t] = (epsilon_t_greedy_rewards[t] * i + reward) / (
            i + 1
        )
        epsilon_t_greedy_arms[t] = chosen_arm

# Calculate cumulative regret for each algorithm
ucb_regret = cal_regret(arms, ucb_arms)
kl_ucb_regret = cal_regret(arms, kl_ucb_arms)
epsilon_t_greedy_regret = cal_regret(arms, epsilon_t_greedy_arms)

# Plot the cumulative regret
regret_dict = {
    "UCB": ucb_regret,
    "KL-UCB": kl_ucb_regret,
    "Epsilon-t-Greedy": epsilon_t_greedy_regret,
}
figure = plot_regret(regret_dict, 50)
figure.savefig("Plots/regret_comparison.pdf", dpi=300, bbox_inches="tight")
