import numpy as np


class KLUCB:
    name = "kl_ucb"

    def __init__(self, num_arms, sigma=1.0):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_iters = 0
        self.sigma = sigma

    def select_arm(self):
        if self.total_iters < self.num_arms:
            return int(self.total_iters)

        kl_ucb_values = self.values + np.sqrt(
            (
                2
                * self.sigma**2
                * (np.log(self.total_iters) + 3 * np.log(np.log(self.total_iters)))
            )
            / self.counts
        )
        return int(np.argmax(kl_ucb_values))

    def update(self, chosen_arm, reward):
        self.total_iters += 1
        self.counts[chosen_arm] += 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n
