import numpy as np

from .base import BasePolicy


class UCB(BasePolicy):
    """UCB1 Agent."""

    name = "ucb"

    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.c = (
            kwargs["C"] if kwargs and "C" in kwargs else 1.0
        )  # exploration parameter
        self.counts = np.zeros(n_arms, dtype=int)  # times each arm pulled
        self.mean_est = np.zeros(n_arms)  # estimated mean rewards

    def select_action(self):
        """Select an arm using UCB rule."""
        # Pull each arm at least once
        if self.n_iters < self.n_arms:
            return int(self.n_iters)

        ucb_values = self.mean_est + self.c * np.sqrt(
            np.log(self.n_iters) / self.counts
        )

        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.n_iters += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.mean_est[arm] += (reward - self.mean_est[arm]) / n

    def reset(self, **kwargs):
        """Reset the policy to the initial state."""
        self.n_iters = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.mean_est = np.zeros(self.n_arms)
