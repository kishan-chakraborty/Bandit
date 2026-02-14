import numpy as np
 
class EpsilonTGreedy:
    def __init__(self, num_arms, c=1.0):
        self.num_arms = num_arms
        self.c = c
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_counts = 0

    def select_arm(self):
        self.total_counts += 1
        eps_t = min(1.0, self.c * self.num_arms / self.total_counts)

        if np.random.rand() < eps_t:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n