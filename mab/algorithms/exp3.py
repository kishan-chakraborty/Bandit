import numpy as np

from mab.algorithms.base import AdversarialBasePolicy


class EXP3(AdversarialBasePolicy):
    name = "exp3"

    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms=n_arms, **kwargs)
        self.gamma = kwargs.get('gamma', 0.1)
            
        self.rng = np.random.default_rng(kwargs.get('seed', 42))

        self.initialize_algorithm()

    def initialize_algorithm(self):
        """
        Initialize the policy.
        This function should be called during reset.
        """
        self.iters = 1
        self.log_weights = np.zeros(self.n_arms)
        self.save_probs = []

        self.initial_exploration_order = self.initial_exploration()

    def cal_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        max_log_weight = np.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight
        
        # mix with uniform
        weights = np.exp(log_weights_normalized)
        probs = (1 - self.gamma) * (weights / weights.sum()) + (
            self.gamma / self.n_arms
        )
        return probs

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iters += 1
        p = self.probs[action]
        x_hat = reward / p
        self.log_weights[action] = (
            self.log_weights[action] + (self.gamma * x_hat) / self.n_arms
        )


if __name__ == "__main__":
    args = {"gamma": 0.1, "seed": 42}
    learner = EXP3(4, **args)
    action = learner.select_action()
    learner.update(action, reward=1)
    print(learner.name)
