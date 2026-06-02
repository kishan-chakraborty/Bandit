"""
Script to run MAB experiments with different environments and algorithms.
"""

import numpy as np


class Experiment:
    def __init__(self, env, algos, horizon, n_rounds):
        """
        Run a MAB experiment.

        Args:
            env: Environment
                The MAB environment to run the experiment on.
            alg: Algorithms
                The MAB algorithms to use in the experiment.
            horizon: int
                The number of time steps to run the experiment for.
            n_rounds: int
                The number of rounds to average the results over.
        """
        self.env = env
        self.algos = algos
        self.horizon = horizon
        self.n_rounds = n_rounds

        self.n_algos = len(self.algos)
        self.regret_hist = np.zeros((self.n_algos, self.horizon))

    def run(self):
        """
        Run the experiment and compute the regret for each algorithm.
        """
        for round in range(self.n_rounds):
            self.env.reset()
            for t in range(self.horizon):
                for i, algo in enumerate(self.algos):
                    action = algo.select_action()
                    reward = self.env.step(action)
                    algo.update(action, reward)
                    self.regret_hist[i, t] += self.env.best_mean - reward

        # Average the regret over the rounds
        self.regret_hist /= self.n_rounds
