"""
Paper:Explore no more: Improved high-probability regret bounds for non-stochastic bandits.

Adversarial Bandit algorithm based on EXP3 to reduce reward variance and
provide a high probability bound over the regret. They have introduced a biased
estimator (under estimate) and overcomes the high variance of the unbiased
estimator through implicit exploration. This also reduces computational complexity
of previous high probability algolrithms such as EXP3.P while maintainig similar result.
"""

from csv import Error
from logging import raiseExceptions

import numpy as np
from mab.algorithms.exp3 import EXP3


class EXP3_IX(EXP3):
    name = "exp3_ix"

    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms, **kwargs)

    def cal_probs(self):
        try:
            max_log_weight = np.max(self.log_weights)
        except:
            raise ValueError("log weights is empty")

        log_weights_normalized = self.log_weights - max_log_weight

        # mix with uniform
        weights = np.exp(log_weights_normalized)
        probs = weights / weights.sum()
        return probs

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iters += 1
        p = self.probs[action]
        x_hat = reward / (p + self.gamma)
        self.log_weights[action] = self.log_weights[action] + (2 * self.gamma * x_hat)


if __name__ == "__main__":
    args = {"gamma": 0.1, "seed": 42}
    learner = EXP3_IX(4, **args)
    action = learner.select_action()
    learner.update(action, reward=1)
    print(learner.name)
