import numpy as np
import regret

class Environment:
    """
    Create a Bernoulli MAB environment.
    """
    def __init__(self, mean_rewards: list, seed: int, **kwargs):
        """
        Stochastic MAB environment:
            r ~ Bernoulli(mu_i)
        Args:
            mean_rewards: list of mean rewards for each arm.
            seed: the random seed for reproducibility.
        """
        self.n_arms = len(mean_rewards)
        self.mean_rewards = mean_rewards

        if any([mu < 0 or mu > 1 for mu in mean_rewards]):
            raise ValueError("Mean rewards must be in the range [0, 1] for Bernoulli distribution.")
        
        self.best_arm = np.argmax(mean_rewards)
        self.best_mean = mean_rewards[self.best_arm]
        self.rng = np.random.default_rng(seed)
        self.reset()

    def step(self, action: int) -> float:
        """
        Take an action and return the reward.

        Args:
            action: int
                The index of the arm to pull.

        Returns: int
            The reward obtained from pulling the arm 0 and 1.
        """
        reward = self.rng.binomial(n=1, p=self.mean_rewards[action])
        self.action_hist.append(action)
        self.reward_hist.append(reward)

        return reward
    
    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.action_hist = []   # List to store the action history.
        self.reward_hist = []   # List to store the reward history.

    def oracle(self):
        """
            Return the oracle information.
        """
        result = {
            "best_arm": self.best_arm,
            "best_mean": self.best_mean,
            "mean_rewards": self.mean_rewards
            }
        return result
    def compute_regret(self):
        """
        Compute the cumulative regret based on the action history and reward history.
        """
        optimal_reward = self.best_mean
        regret = [optimal_reward - self.mean_rewards[action] for action in self.action_hist]
        cumulative_regret = np.cumsum(regret)
        return cumulative_regret
    