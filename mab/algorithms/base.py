class BasePolicy:
    """Base class for all MAB algorithms."""

    name = "base_policy"

    def __init__(self, n_arms, seed=None, **kwargs):
        self.n_arms = n_arms
        self.iters = 1

    def select_arm(self):
        """Select an arm using UCB rule."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.iters += 1
        raise NotImplementedError("This method must be implemented by subclasses.")

    def reset(self):
        """Reset the policy to the initial state."""
        self.iters = 1
