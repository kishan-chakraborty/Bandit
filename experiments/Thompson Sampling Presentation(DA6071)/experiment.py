from pathlib import Path

current_dir = Path(__file__).resolve().parent

import concurrent.futures
import pickle
from mab.algorithms.thompson_sampling import BernoulliTS
from mab.environment.stationary.bernoulli import Bernoulli
from mab.environment.env import Agent
from mab.environment.experiment import Experiment
from mab.environment.plots import plot_regret
import numpy as np


class BernoulliTS1(BernoulliTS):
    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms, **kwargs)

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.iters = 1
        self.mean_est = np.zeros(self.n_arms)
        self.save_alphas = []
        self.save_betas = []

        self.initial_exploration_order = self.initial_exploration()

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.save_alphas.append(self.alphas)
        self.save_betas.append(self.betas)

        self.iters += 1
        self.alphas[arm] = self.alphas[arm] + reward
        self.betas[arm] = self.betas[arm] + (1 - reward)


# Define the experiment parameters
n_arms = 3
time_horizon = 10_000
num_simulations = 5

# Prior parameters of beta distribution.
alphas = np.ones(n_arms)
betas = np.ones(n_arms)

bernoulli_probs = [0.9, 0.8, 0.7]  # True probabilities for each arm.

# Save data
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)

seeds = np.arange(1, 11, dtype=int)


# Setup the experiment.
def run_experiment(seed):
    env = Bernoulli(mean_rewards=bernoulli_probs, seed=seed)
    kwargs = {"alphas": alphas, "betas": betas, "seed": seed}
    ts_agent = Agent(BernoulliTS1(n_arms, **kwargs))

    algos = [ts_agent]
    kwargs = {"store_rewards": True}
    experiment = Experiment(env, algos, time_horizon, num_simulations, **kwargs)
    experiment.run()

    file_path = str(data_dir) + f"/seed{seed}_environment_data.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(experiment, f)

    return f"Seed {seed} done"


with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(run_experiment, seeds))

print(results)
