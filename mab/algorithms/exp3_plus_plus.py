"""
Implementation of EXP3++ algorithm with gap estimation.
Source: An Improved Parametrization and Analysis of the EXP3++ Algorithm
for Stochastic and Adversarial Bandits

Author: Kishan Chakraborty
"""
import numpy as np

class EXP3PlusPlus:
    name = 'exp3++'
    def __init__(self, n_arms, args=None):
        self.n_arms = n_arms
        self.c = args['c'] if args and 'c' in args else 20

        # Initialize the weights and probabilities
        self.weights = np.zeros(n_arms) # \tilde{L}_{t}(a) in the paper
        self.iter = 1

        # History of probabilities for analysis
        self.save_probs = []
    
    def estimate_gaps(self):
        """Estimate the gaps between the current best arm and the others."""
        estimated_means = self.weights / self.iter # Estimated mean rewards for each arm
        gaps = estimated_means - np.min(estimated_means)

        return np.minimum(1, gaps) # Ensure gaps are in [0,1]

    def _calculate_eps(self, beta, xi):
        # Calculate epsilon: The exploration exploitation trade-off parameter
        eps = np.minimum(0.5 * (1/self.n_arms), beta, xi)
        return eps

    def _calculate_beta(self):
        # Calculate eta: The learning rate
        beta = 0.5 * np.sqrt(np.log(self.n_arms) / (self.iter * self.n_arms))
        return beta
    
    def _calculate_eta(self, beta):
        """Calculate the learning rate. Paper suggests to use eta >= beta"""
        return beta
    
    def _calculate_xi(self):
        """Calculate xi: The gap based exploration parameter."""
        # Calculate the empirical gap between arms.
        gaps = self.estimate_gaps()
        gaps = gaps + 1e-10 # To avoid division by zero

        xi = (self.c * (np.log(self.iter) ** 2)) / (self.iter * (gaps ** 2))
        return xi

    def _compute_probs(self):
        "Compute the probability distribution over actions."

        eta = self._calculate_eta(self.beta)

        # Calculate the probabilities over arms
        weights_normalized = self.weights - min(self.weights)
        exp_weights = np.exp(-eta * weights_normalized)
        probs = exp_weights / np.sum(exp_weights)

        return probs
    
    def cal_probs(self):
        """Calculate the probability distribution over arms."""
        self.beta = self._calculate_beta()
        self.xi = self._calculate_xi() # Gap-based exploration parameter.
        self.eps = self._calculate_eps(self.beta, self.xi)

        probs = self._compute_probs()
        probs = (1 - sum(self.eps)) * probs + self.eps

        # Ensure probs are not very small
        probs = np.maximum(probs, 1e-10)

        return probs
    
    def choose_action(self):
        "Sample an action according to the current probability distribution."
        # Make sure all the arms are chosen atleast once
        # if self.iter <= self.n_arms:
        #     self.save_probs.append(np.ones(self.n_arms) / self.n_arms)
        #     return self.iter - 1
        self.probs = self.cal_probs()
        self.save_probs.append(self.probs)
        if np.isnan(self.probs).any():
            print("NaN detected")

        return np.random.choice(self.n_arms, p=self.probs)

    
    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iter += 1
        p = self.probs[action]
        loss = 1 - reward # Convert to loss
        x_hat = loss / p # Weighted loss estimate.
        self.weights[action] = self.weights[action] + x_hat


class GapEstimation(EXP3PlusPlus):
    name = 'gap_estimation_exp3++'
    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.weights = np.zeros(n_arms) # \hat{L}_{t}(a)
        self.counts = np.zeros(n_arms) # N_{t}(a), counts action selections.
        self.rng = np.random.default_rng(kwargs.get('seed', 42))

        self.initial_exploration_order = self.initial_exploration()

    def initial_exploration(self):
        """
        To ensure that all arms are explored at least once.
        """
        # Randomize the order of arms selected.
        arms = np.arange(self.n_arms, dtype=int)
        self.rng.shuffle(arms)
        return arms
    
    def cal_ucb(self):
        

    def cal_lcb(self):
        pass

    def cal_gaps(self, lcb, ucb):
        pass

    def choose_action(self):
        "Sample an action according to the current probability distribution."
        # Ensure all arms are explored at least once.
        if self.iter <= self.n_arms:
            action = self.initial_exploration_order[self.iter - 1]
            self.save_probs.append(np.ones(self.n_arms) / self.n_arms)
            return action
        
        self.probs = self.cal_probs()
        self.save_probs.append(self.probs)
        if np.isnan(self.probs).any():
            print("NaN detected")

        return np.random.choice(self.n_arms, p=self.probs)

    def update(self, action: int, reward: float):
        "Update the weights based on the received reward."
        # reward in [0,1]
        self.iter += 1
        loss = 1 - reward # Convert to loss
        self.weights[action] = self.weights[action] + loss
        self.weights[action] = self.weights[action] + 1



if __name__ == "__main__":
    n_arms = 4
    args = {}
    learner = EXP3PlusPlus(n_arms, args)
    action = learner.choose_action()
    learner.update(action, reward=1)