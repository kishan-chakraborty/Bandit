import numpy as np

from .base import BasePolicy


class EXP3(BasePolicy):
    name = "exp3"

    def __init__(self, n_arms: int, seed=None, **kwargs):
        super().__init__(n_arms=n_arms, seed=seed, **kwargs)
        self.gamma = kwargs.get("gamma", 0.1)
        self.rng = np.random.default_rng(seed)
        self.log_weights = np.zeros(n_arms)
        self.probs = self._compute_probs()
        self.save_probs = []

    def _compute_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        max_log_weight = np.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight
        # mix with uniform
        weights = np.exp(log_weights_normalized)
        probs = (1 - self.gamma) * (weights / weights.sum()) + (
            self.gamma / self.n_arms
        )
        return probs

    def select_action(self) -> int:
        "Sample an action according to the current probability distribution."
        self.probs = self._compute_probs()
        self.save_probs.append(self.probs)
        if np.isnan(self.probs).any():
            print("NaN detected")
        return np.random.choice(self.n_arms, p=self.probs)

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        p = self.probs[action]
        x_hat = reward / p
        self.log_weights[action] = (
            self.log_weights[action] + (self.gamma * x_hat) / self.n_arms
        )

    def reset(self, **kwargs):
        "Reset the policy to the initial state."
        self.seed = kwargs.get("seed")
        self.log_weights = np.zeros(self.n_arms)
        self.probs = self._compute_probs()
        self.save_probs = []


if __name__ == "__main__":
    learner = EXP3(4, {"gamma": 0.1})
    action = learner.choose_action()
    learner.update(action, reward=1)
    print(learner.name)
