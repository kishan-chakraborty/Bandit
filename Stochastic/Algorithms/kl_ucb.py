import numpy as np


class KLUCB:
    def __init__(self, num_arms, sigma=1.0):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_counts = 0
        self.sigma = sigma

    def select_arm(self):
        if self.total_counts < self.num_arms:
            return self.total_counts

        t = self.total_counts
        kl_ucb_values = self.values + np.sqrt(
            (2 * self.sigma**2 * (np.log(t) + 3 * np.log(np.log(t)))) / self.counts
        )
        return np.argmax(kl_ucb_values)

    def update(self, chosen_arm, reward):
        self.total_counts += 1
        self.counts[chosen_arm] += 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n
