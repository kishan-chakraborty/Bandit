import numpy as np


class UCB:
    """UCB1 Agent."""

    name = "ucb"

    def __init__(self, n_arms, args=None):
        self.n_arms = n_arms
        self.c = args["C"]  # exploration parameter
        self.counts = np.zeros(n_arms, dtype=int)  # times each arm pulled
        self.values = np.zeros(n_arms)  # estimated mean rewards
        self.t = 0

    def select_arm(self):
        """Select an arm using UCB rule."""
        self.t += 1
        # Pull each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
