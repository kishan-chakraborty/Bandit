import numpy as np

from .base import BasePolicy


class KLUCB(BasePolicy):
    name = "kl_ucb"

    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.sigma = kwargs.get("sigma", 1.0)  # variance of rewards

    def select_action(self):
        if self.n_iters < self.n_arms:
            return int(self.n_iters)

        kl_ucb_values = self.values + np.sqrt(
            (
                2
                * self.sigma**2
                * (np.log(self.n_iters) + 3 * np.log(np.log(self.n_iters)))
            )
            / self.counts
        )
        return int(np.argmax(kl_ucb_values))

    def update(self, chosen_arm, reward):
        self.n_iters += 1
        self.counts[chosen_arm] += 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n

    def reset(self, **kwargs):
        self.n_iters = 0
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
