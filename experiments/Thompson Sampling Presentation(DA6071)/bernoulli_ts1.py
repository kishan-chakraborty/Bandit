import numpy as np
from mab.algorithms.thompson_sampling import BernoulliTS

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
        self.save_alphas.append(self.alphas.tolist())
        self.save_betas.append(self.betas.tolist())

        self.iters += 1
        self.alphas[arm] = self.alphas[arm] + reward
        self.betas[arm] = self.betas[arm] + (1 - reward)
