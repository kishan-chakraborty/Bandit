"""
This is to compare various MAB algorithms in Bernoulli environment.

Kishan Chakraborty, 23-05-2026
"""

from mab.algorithms.epsilon_t_greedy import EpsilonTGreedy
from mab.algorithms.kl_ucb import KLUCB
from mab.algorithms.ucb import UCB
from mab.environment.experiment import Experiment
from mab.environment.stationary.bernoulli import Bernoulli

# Experiment to compare various stochastic bandit algorithms

# Define the experiment parameters
n_arms = 5
time_horizon = 1000
num_simulations = 10

bernoulli_probs = [0.1, 0.3, 0.5, 0.7, 0.9]  # True probabilities for each arm
env = Bernoulli(mean_rewards=bernoulli_probs, seed=42)

# Initialize the algorithms
ucb = UCB(n_arms=n_arms)
kl_ucb = KLUCB(n_arms=n_arms)
epsilon_t_greedy = EpsilonTGreedy(n_arms=n_arms)

algorithms = [ucb, kl_ucb, epsilon_t_greedy]

# Run the experiment
experiment = Experiment(env, algorithms, time_horizon, num_simulations)
experiment.run()
