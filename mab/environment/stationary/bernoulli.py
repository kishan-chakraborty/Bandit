import numpy as np

from mab.environment.env import Environment

from .utils import cal_stochastic_regret


class Bernoulli(Environment):
    """
    Create a MAB environment with Bernoulli rewards.
    """

    def __init__(self, mean_rewards: list, seed: int, **kwargs):
        super().__init__(mean_rewards, seed, **kwargs)
        if any([mu < 0 or mu > 1 for mu in mean_rewards]):
            raise ValueError(
                "Mean rewards must be in the range [0, 1] for Bernoulli distribution."
            )

    def step(self, action: int | np.int64) -> float:
        """
        Take an action and return the reward.

        Args:
            action: int or np.int64
                The index of the arm to pull.
        Returns: int
            The reward obtained from pulling the arm 0 and 1.
        """
        reward = self.rng.binomial(n=1, p=self.mean_rewards[action])
        self.action_hist.append(action)
        self.reward_hist.append(reward)

        return reward

    def cal_regret(self):
        """
        Compute the cumulative regret based on the action history and reward history.
        """
        cumulative_regret = cal_stochastic_regret(self.mean_rewards, self.action_hist)
        return cumulative_regret


if __name__ == "__main__":
    mean_rewards = [0.1, 0.5, 0.9]
    seed = 42
    env = Bernoulli(mean_rewards, seed)

    print("Oracle information:", env.oracle())
    for _ in range(10):
        action = env.rng.integers(0, env.n_arms)
        reward = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
