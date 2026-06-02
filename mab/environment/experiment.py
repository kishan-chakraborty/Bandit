"""
Script to run MAB experiments with different environments and algorithms.
"""

import numpy as np

from mab.environment.stationary.utils import cal_stochastic_regret


class Experiment:
    def __init__(self, env, agents, horizon, n_rounds, **kwargs):
        """
        Run a MAB experiment.

        Args:
            env: Environment
                The MAB environment to run the experiment on.
            agents: list of Agent
                The list of agents each equipped with different learning strategies (diff algos or algo with diff params).
            horizon: int
                The number of time steps to run the experiment for.
            n_rounds: int
                The number of rounds to average the results over.
        """
        self.env = env
        self.agents = agents
        self.horizon = horizon
        self.n_rounds = n_rounds

        self.n_agents = len(self.agents)

        # Initialize the reward and action history for each agent.
        self.reset_agent_histories()

        # Store the data for each run.
        self.experiment_data = {
            "agent_rewards": np.zeros((self.n_agents, self.n_rounds, self.horizon)),
            "agent_actions": np.zeros((self.n_agents, self.n_rounds, self.horizon), dtype=int),
        }

        # Whether to store rewards for each time step.
        self.store_rewards = kwargs.get("store_rewards", False)
    
    def reset_agent_histories(self):
        """
        Reset the reward and action history for each agent.
        """
        for agent in self.agents:
            agent.action_hist = np.zeros(self.horizon, dtype=int)
            agent.reward_hist = np.zeros(self.horizon, dtype=float)

    def run(self):
        """
        Run the experiment and compute the regret for each algorithm.
        """
        for round in range(self.n_rounds):
            self.reset_agent_histories()  # Reset histories for each round.
            for t in range(self.horizon):
                for i, agent in enumerate(self.agents):
                    action = agent.select_action()
                    reward = self.env.step(action)
                    agent.algo.update(action, reward)

                    # Store the reward and action.
                    agent.action_hist[t] = action
                    if self.store_rewards:
                        agent.reward_hist[t] = reward

            # Store the data for this round.
            for i, agent in enumerate(self.agents):
                self.experiment_data["agent_actions"][i, round, :] = agent.action_hist
                if self.store_rewards:
                    self.experiment_data["agent_rewards"][i, round, :] = agent.reward_hist

    def compute_average_regret(self):
        """
        Compute the cumulative regret for each agent based on the action history and reward history.
        """
        mean_rewards = self.env.oracle()["mean_rewards"]

        for i, agent in enumerate(self.agents):
            regret = np.zeros(self.horizon)
            for _ in range(self.n_rounds):
                chosen_arms = self.experiment_data["agent_actions"][i, :, :].flatten()
                current_regret = cal_stochastic_regret(mean_rewards, chosen_arms)
                regret += current_regret
            regret = regret / self.n_rounds  # Average over rounds
            agent.regret = regret
            
        regret_dict = {f"Agent_{i}": agent.regret for i, agent in enumerate(self.agents)}
        return regret_dict

    def save_data(self, filename):
        """
        Save the experiment data to a file.

        Args:
            filename: str
                The name of the file to save the data to.
        """
        pass
