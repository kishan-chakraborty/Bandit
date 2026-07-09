"""
This implementation of EXP3 is based on "Regret Analysis of Stochastic and
Nonstochastic Multi-armed Bandit Problems" with a dynamic exploration rate (gamma).
This is a loss based implementation without any uniform exploration (No gamma/K).
"""

import numpy as np

from mab.algorithms.exp3 import EXP3


class EXP3Dynamic(EXP3):
    name = "exp3_dynamic"

    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms=n_arms, **kwargs)

        self.rng = np.random.default_rng(kwargs.get("seed", 42))

        self.initialize_algorithm()

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.iters = 1
        self.weights = np.zeros(self.n_arms)
        self.save_probs = []

        self.initial_exploration_order = self.initial_exploration()

    def cal_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        min_weight = np.min(self.weights)
        weights_normalized = self.weights - min_weight

        self.eta = np.sqrt(np.log(self.n_arms) / (self.iters * self.n_arms))
        weights = np.exp(-self.eta * weights_normalized)
        probs = weights / sum(weights)
        return probs

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iters += 1

        loss = 1 - reward
        p = self.probs[action]
        x_hat = loss / p
        self.weights[action] = self.weights[action] + x_hat


if __name__ == "__main__":
    args = {"gamma": 0.1, "seed": 42}
    learner = EXP3Dynamic(4, **args)
    action = learner.select_action()
    learner.update(action, reward=1)
    print(learner.name)
