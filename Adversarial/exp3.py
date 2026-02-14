import numpy as np


class EXP3:
    def __init__(self, K: int, gamma: float = 0.1):
        self.K = K
        self.gamma = gamma
        self.weights = np.ones(K)
        self.probs = self._compute_probs()
        self.save_probs = []

    def _compute_probs(self):
        "Compute the probability distribution over actions."
        # Regularize the weights to avoid weights being too large.
        self.weights = self.weights / self.weights.sum()
        # mix with uniform
        probs = (1 - self.gamma) * self.weights + (self.gamma / self.K)
        return probs

    def get_action(self) -> int:
        "Sample an action according to the current probability distribution."
        self.probs = self._compute_probs()
        self.save_probs.append(self.probs)
        if np.isnan(self.probs).any():
            print("NaN detected")
        return np.random.choice(self.K, p=self.probs)

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        p = self.probs[action]
        x_hat = reward / p
        self.weights[action] = self.weights[action] * np.exp(
            (self.gamma * x_hat) / self.K
        )
        # small numeric safety
        self.weights = np.maximum(self.weights, 1e-12)
