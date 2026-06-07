"""
Implementation of EXP3++ algorithm with gap estimation. This is based on two papers:
Source: One Practical Algorithm for Both Stochastic and Adversarial Bandits
An Improved Parametrization and Analysis of the EXP3++ Algorithm
for Stochastic and Adversarial Bandits

Author: Kishan Chakraborty
"""
import numpy as np
from mab.algorithms.base import AdversarialBasePolicy

class EXP3PlusPlus(AdversarialBasePolicy):
    name = 'exp3++'
    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.alpha = kwargs.get('alpha', 0.5) if 'alpha' in kwargs else 3
        self.beta = kwargs.get('beta', 0.5) if 'beta' in kwargs else 256

        self.rng = np.random.default_rng(kwargs.get('seed', 42))

        self.initialize_algorithm()

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.loss_unweighted = np.zeros(self.n_arms) # \hat{L}_{t}(a)
        self.loss_weighted = np.zeros(self.n_arms) # \tilde{L}_{t}(a)
        self.counts = np.zeros(self.n_arms) # N_{t}(a), counts action selections.
        self.iters = 1

        self.initial_exploration_order = self.initial_exploration()

        self.save_probs = []
    
    def cal_ucb(self):
        temp = (self.loss_unweighted / self.counts) + np.sqrt((self.alpha * np.log(self.iters * (self.n_arms ** (1/self.alpha))) / (2 * self.counts)))
        return np.minimum(1, temp) # Ensure UCB is in [0,1]

    def cal_lcb(self):
        temp = (self.loss_unweighted / self.counts) - np.sqrt((self.alpha * np.log(self.iters * (self.n_arms ** (1/self.alpha))) / (2 * self.counts)))
        return np.maximum(0, temp) # Ensure LCB is in [0,1]

    def estimate_gaps(self):
        """
        Estimate the gaps (Delta).
        """
        ucb = self.cal_ucb()
        lcb = self.cal_lcb()

        gaps = np.maximum(0, lcb - np.min(ucb)) # Gap is max(0, LCB - min UCB)
        return gaps
    
    def cal_xi(self):
        gaps = self.estimate_gaps() + 1e-10 # To avoid division by zero
        temp = (self.beta * np.log(self.iters)) / (self.iters * (gaps ** 2))
        return temp

    def cal_eps(self):
        self.xi = self.cal_xi()
        eps = np.minimum(0.5 * (1/self.n_arms), 0.5 * np.sqrt(np.log(self.n_arms) / (self.iters * self.n_arms)), self.xi)
        return eps
    
    def _compute_probs(self):
        "Compute the probability distribution over actions."

        eta = 0.5 * np.sqrt(self.n_arms / (self.iters * self.n_arms))

        # Calculate the probabilities over arms
        weights_normalized = self.loss_weighted - min(self.loss_weighted)
        exp_weights = np.exp(-eta * weights_normalized)
        probs = exp_weights / np.sum(exp_weights)

        return probs
    
    def cal_probs(self):
        """Calculate the probability distribution over arms."""
        self.eps = self.cal_eps()

        probs = self._compute_probs()
        probs = (1 - sum(self.eps)) * probs + self.eps

        # Ensure probs are not very small
        probs = np.maximum(probs, 1e-10)

        return probs

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iters += 1
        p = self.probs[action]
        loss = 1 - reward # Convert to loss
        loss_weighted = loss / p # Weighted loss estimate.

        self.loss_unweighted[action] = self.loss_unweighted[action] + loss
        self.counts[action] = self.counts[action] + 1

        self.loss_weighted[action] = self.loss_weighted[action] + loss_weighted

if __name__ == "__main__":
    n_arms = 4
    args = {}
    learner = EXP3PlusPlus(n_arms, **args)
    action = learner.select_action()
    learner.update(action, reward=1)