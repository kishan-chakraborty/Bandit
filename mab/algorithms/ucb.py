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

        self.rng = np.random.default_rng(kwargs.get("seed", 42))
        self.initial_exploration_order = self.initial_exploration()

    def initial_exploration(self):
        """
        To ensure that all arms are explored at least once.
        """
        # Randomize the order of arms selected.
        arms = np.arange(self.n_arms, dtype=int)
        self.rng.shuffle(arms)
        return arms

    def select_action(self):
        """Select an arm using UCB rule."""
        # Pull each arm at least once
        if self.iters <= self.n_arms:
            return int(self.initial_exploration_order[self.iters - 1])

        ucb_values = self.mean_est + self.c * np.sqrt(
            np.log(self.iters) / self.counts
        )

        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.iters += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.mean_est[arm] += (reward - self.mean_est[arm]) / n

    def reset(self, **kwargs):
        """Reset the policy to the initial state."""
        self.iters = 1
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.mean_est = np.zeros(self.n_arms)
