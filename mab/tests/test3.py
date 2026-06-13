"""
Experimenting congestion gamge.
"""
from mab.algorithms import exp3

from mab.environment.games.congestion_game import SingletonCongestionGame
from mab.environment.env import Agent
from mab.environment.games.experiment import Experiment
from mab.environment.games.utils import Resource, AffineResource
from mab.environment.plots import plot_regret

# Experiment to compare various stochastic bandit algorithms

# Define the experiment parameters
n_arms = 3
time_horizon = int(1e5)
num_simulations = 1

# Define resources
res1 = AffineResource(1, 0)
res2 = AffineResource(2, 0)
res3 = AffineResource(0.5, 5)

resources = [Resource(res1.cost), Resource(res2.cost), Resource(res3.cost)]

# Define environment.
env = SingletonCongestionGame(resources=resources)

# Initialize the algorithms
exp3_agent1 = Agent(exp3.EXP3(n_arms=n_arms, seed=41, gamma=0.01))
exp3_agent2 = Agent(exp3.EXP3(n_arms=n_arms, seed=42, gamma=0.01))
exp3_agent3 = Agent(exp3.EXP3(n_arms=n_arms, seed=43, gamma=0.01))

agents = [exp3_agent1, exp3_agent2, exp3_agent3]

# Run the Experiment
experiment = Experiment(env, agents, time_horizon, num_simulations)
experiment.run()
regret_dict = experiment.cal_avg_bandit_regret(0)