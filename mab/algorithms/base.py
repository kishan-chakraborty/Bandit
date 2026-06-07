import numpy as np

class BasePolicy:
    """
    Base class for all MAB algorithms. This should be overridden for adversarial setting.
    For stochastic regime, this can be used as a base policy.
    """

    name = "base_policy"

    def __init__(self, n_arms, **kwargs):
        self.n_arms = n_arms
        self.iters = 1
        self.rng = np.random.default_rng(kwargs.get('seed', 42))
        self.initial_exploration_order = self.initial_exploration()

    def initialize_algorithm(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def initial_exploration(self):
        """
        To ensure that all arms are explored at least once.
        """
        # Randomize the order of arms selected.
        arms = np.arange(self.n_arms, dtype=int)
        self.rng.shuffle(arms)
        return arms
    
    def initial_selection(self):
        """
        Select arms in the initial exploration phase.
        """
        return int(self.initial_exploration_order[self.iters - 1])

    def select_action(self):
        """Select an arm using some rule."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def update(self, action, reward):
        """Update the empirical mean of chosen arm."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def reset(self):
        """Reset the policy to the initial state."""
        self.initialize_algorithm()
        
class AdversarialBasePolicy(BasePolicy):
    """Base class for all adversarial MAB algorithms."""
    def __init__(self, n_arms, **kwargs):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(kwargs.get('seed', 42))

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.save_probs = []
        raise NotImplementedError("This method must be implemented by subclasses.")

    def initial_exploration(self):
        """
        To ensure that all arms are explored at least once.
        """
        # Randomize the order of arms selected.
        arms = np.arange(self.n_arms, dtype=int)
        self.rng.shuffle(arms)
        return arms

    def initial_selection(self):
        """
        Select arms in the initial exploration phase.
        """
        # Select arms in the order determined by initial_exploration.
        action = self.initial_exploration_order[self.iters - 1]
        self.probs = np.ones(self.n_arms) / self.n_arms
        self.save_probs.append(self.probs)
        return action
    
    def cal_probs(self):
        """
        Calculate action probabilities.
        """
        raise NotImplementedError('To be implemented by a subclass')
    
    def select_action(self):
        "Sample an action according to the current probability distribution."
        # Ensure all arms are explored at least once.
        if self.iters <= self.n_arms:
            return self.initial_selection()
        
        self.probs = self.cal_probs()
        self.save_probs.append(self.probs)
        if np.isnan(self.probs).any():
            print("NaN detected")

        return np.random.choice(self.n_arms, p=self.probs)
    
    def reset(self):
        "Reset the policy to the initial state."
        self.initialize_algorithm()
