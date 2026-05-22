import numpy as np


class EXP3:
    name = 'exp3'
    def __init__(self, K: int, args: dict = None):
        self.K = K
        self.gamma = args.get('gamma', 0.1) if args else 0.1
        self.log_weights = np.zeros(K)
        self.probs = self._compute_probs()
        self.save_probs = []

    def _compute_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        max_log_weight = np.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight
        # mix with uniform
        weights = np.exp(log_weights_normalized)
        probs = (1 - self.gamma) * (weights / weights.sum()) + (self.gamma / self.K)
        return probs

    def choose_action(self) -> int:
        "Sample an action according to the current probability distribution."
        self.probs = self._compute_probs()
        self.save_probs.append(self.probs)
        if np.isnan(self.probs).any():
            print("NaN detected")
        return np.random.choice(self.K, p=self.probs)

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        p = self.probs[action]
        x_hat = reward / p
        self.log_weights[action] = self.log_weights[action] + (self.gamma * x_hat) / self.K

if __name__ == "__main__":
    learner = EXP3(4, {'gamma': 0.1})
    action = learner.choose_action()
    learner.update(action, reward=1)
    print(learner.name)