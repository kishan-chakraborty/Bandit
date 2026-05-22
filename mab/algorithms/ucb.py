import matplotlib.pyplot as plt
import numpy as np

from Stochastic.Rewards.gaussian import GaussianReward
from Stochastic.utils import cal_regret


class UCB:
    """UCB1 Agent."""

    name = "ucb"

    def __init__(self, n_arms, args=None):
        self.n_arms = n_arms
        self.c = args["C"]  # exploration parameter
        self.total_iters = 0
        self.counts = np.zeros(n_arms, dtype=int)  # times each arm pulled
        self.mean_est = np.zeros(n_arms)  # estimated mean rewards

    def select_arm(self):
        """Select an arm using UCB rule."""
        # Pull each arm at least once
        if self.total_iters < self.n_arms:
            return int(self.total_iters)

        ucb_values = self.mean_est + self.c * np.sqrt(
            np.log(self.total_iters) / self.counts
        )

        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        """Update the empirical mean of chosen arm."""
        self.total_iters += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.mean_est[arm] += (reward - self.mean_est[arm]) / n


if __name__ == "__main__":
    learner = UCB(2, args={"C": 2.0})
    arm1 = GaussianReward(1, 1)
    arm2 = GaussianReward(2, 1)
    arms = [arm1, arm2]
    T = 100000

    hist = np.zeros(T, dtype=int)

    for t in range(T):
        arm = learner.select_arm()
        reward = arms[arm].pull()
        learner.update(arm, reward)
        hist[t] = arm

        # print(arm, reward)

    regret = cal_regret(reward_distributions=arms, hist=hist)

    plt.plot(regret)
    plt.show()
