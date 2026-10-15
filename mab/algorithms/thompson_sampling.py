import numpy as np
from mab.algorithms.base import BasePolicy


class BernoulliTS(BasePolicy):
    name = "bernoulli_ts"

    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.alphas = kwargs["alphas"]
        self.betas = kwargs["betas"]
        self.rng = np.random.default_rng(kwargs.get("seed", 42))

        self.initialize_algorithm()

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.iters = 1
        self.mean_est = np.zeros(self.n_arms)

        self.initial_exploration_order = self.initial_exploration()

    def select_action(self):
        """Select an arm using UCB rule."""
        # Pull each arm at least once
        if self.iters <= self.n_arms:
            return self.initial_selection()

        self.mean_est = [
            np.random.beta(alpha, beta) for alpha, beta in zip(self.alphas, self.betas)
        ]

        return int(np.argmax(self.mean_est))

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.iters += 1
        self.alphas[arm] = self.alphas[arm] + reward
        self.betas[arm] = self.betas[arm] + (1 - reward)


if __name__ == "__main__":
    n_arms = 5
    args = {"alphas": np.ones(n_arms), "betas": np.ones(n_arms), "seed": 42}
    learner = BernoulliTS(4, **args)
    action = learner.select_action()
    learner.update(action, reward=1)
    print(learner.name)
