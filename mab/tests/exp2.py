"""
This is to compare UCB, EXP3 and EXP3++ algorithms in stochastic bandit environment.

Kishan Chakraborty, June 5, 2026.
"""

from mab.algorithms import epsilon_t_greedy, kl_ucb
from mab.algorithms.ucb import UCB
from mab.algorithms.exp3 import EXP3
from mab.algorithms.exp3_plus_plus import EXP3PlusPlus

from mab.environment.stationary.bernoulli import Bernoulli
from mab.environment.stationary.utils import Agent
from mab.environment.experiment import Experiment
from mab.environment.plots import plot_regret

# Experiment to compare various stochastic bandit algorithms

# Define the experiment parameters
n_arms = 5
time_horizon = 1000
num_simulations = 10

bernoulli_probs = [0.1, 0.3, 0.5, 0.7, 0.9]  # True probabilities for each arm
env = Bernoulli(mean_rewards=bernoulli_probs, seed=42)

# Initialize the algorithms
ucb = Agent(UCB(n_arms=n_arms))
exp3 = Agent(EXP3(n_arms=n_arms, seed=42))
exp3_plus_plus = Agent(EXP3PlusPlus(n_arms=n_arms, seed=42))

algorithms = [ucb, exp3, exp3_plus_plus]

# Run the experiment
experiment = Experiment(env, algorithms, time_horizon, num_simulations)
experiment.run()
regret_dict = experiment.compute_average_regret()
regret_plot = plot_regret(regret_dict)
