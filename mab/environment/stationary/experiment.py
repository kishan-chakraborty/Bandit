"""
Script to run Stationary MAB experiments with different environments and algorithms.
"""
import numpy as np

from ..experiment import Experiment
from .utils import cal_stochastic_regret

class StationaryExperiment(Experiment):
    def __init__(self, env, agents, horizon, n_rounds, **kwargs):
        super().__init__(env, agents, horizon, n_rounds, **kwargs)

    def compute_average_regret(self):
        """
        Calculate average regret for stationary environment.
        """
        mean_rewards = self.env.oracle()["mean_rewards"]

        for i, agent in enumerate(self.agents):
            regret = np.zeros(self.horizon)
            for round in range(self.n_rounds):
                chosen_arms = self.experiment_data["agent_actions"][
                    i, round, :
                ].flatten()
                current_regret = cal_stochastic_regret(mean_rewards, chosen_arms)
                regret += current_regret
            regret = regret / self.n_rounds  # Average over rounds
            agent.regret = regret

        regret_dict = {
            f"Agent_{i}_{agent.algo.name}": agent.regret for i, agent in enumerate(self.agents)
        }
        return regret_dict