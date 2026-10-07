import numpy as np
from mab.algorithms.base import AdversarialBasePolicy


class EXP3P(AdversarialBasePolicy):
    name = "exp3_p1"

    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms=n_arms, **kwargs)

        self.gamma = kwargs.get("gamma", 0.1)
        self.alpha = kwargs.get("alpha", 1.0)
        self.T = kwargs.get("T")

        if self.T is None:
            raise ValueError("EXP3P requires the horizon T.")

        self.rng = np.random.default_rng(kwargs.get("seed", 42))

        self.initialize_algorithm()

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.iters = 1
        self.save_probs = []

        initial_log_weight = (self.alpha * self.gamma / 3) * np.sqrt(
            self.T / self.n_arms
        )

        self.log_weights = np.full(self.n_arms, initial_log_weight, dtype=float)

        self.initial_exploration_order = self.initial_exploration()

    def cal_probs(self):
        """
        Compute the probability distribution over actions.
        """
        # Numerical stabilization for softmax.
        max_log_weight = np.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight

        weights = np.exp(log_weights_normalized)

        probs = (1 - self.gamma) * (weights / weights.sum()) + self.gamma / self.n_arms

        return probs

    def update(self, action: int, reward: float):
        """
        Update the weights based on the observed reward.

        reward is assumed to lie in [0, 1].
        """
        self.iters += 1

        # Probability of the selected arm.
        p = self.probs[action]
        x_hat = reward / p
        confidence_bonus = self.alpha / (self.probs * np.sqrt(self.n_arms * self.T))

        self.log_weights += self.gamma / (3 * self.n_arms) * confidence_bonus

        # The estimated reward is nonzero only for the selected arm.
        self.log_weights[action] += self.gamma / (3 * self.n_arms) * x_hat


class EXP3P1(AdversarialBasePolicy):
    name = "exp3_p1"

    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.gamma = kwargs["gamma"]
        self.beta = kwargs["beta"]
        self.eta = kwargs["eta"]

    def cal_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        max_log_weight = np.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight

        # mix with uniform
        weights = np.exp(self.eta * log_weights_normalized)
        probs = (1 - self.gamma) * (weights / weights.sum()) + (
            self.gamma / self.n_arms
        )
        return probs

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iters += 1
        p = self.probs[action]
        x_hat = reward / p
        self.log_weights = self.log_weights + (self.beta / self.probs)
        self.log_weights[action] = self.log_weights[action] + x_hat
