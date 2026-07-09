"""
Experiment for multi-player game environment.
"""
import numpy as np
from mab.environment.experiment import Experiment as BaseExperiment

class Experiment(BaseExperiment):
    def __int__(self, env, agents, horizon, n_rounds, **kwargs):
        super().__init__(env, agents, horizon, n_rounds, **kwargs)

    def initialize_experiment(self):
        """
        Initialize the script.
        """
        # Initialize the reward and action history for each agent.
        self.reset_agent_histories()

        # Store the data for each run.
        self.experiment_data = {
            "agent_probs": np.zeros(
                (self.n_agents, self.n_rounds, self.horizon, self.env.n_resources)
            ),
            "agent_actions": np.zeros((self.n_agents, self.n_rounds, self.horizon), dtype=int)
        }
    
    def store_data_round(self, round):
        """
        Store history per round
        """
        # Store the data for this round.
        for i, agent in enumerate(self.agents):
            self.experiment_data["agent_probs"][i, round, :, :] = agent.algo.save_probs
            if self.store_rewards:
                self.experiment_data["agent_rewards"][
                    i, round, :
                ] = agent.reward_hist

    def run(self):
        """
        Run the experiment and compute the regret for each algorithm.
        """
        for round in range(self.n_rounds):
            self.reset_agent_histories()  # Reset histories for each round.
            for t in range(self.horizon):
                # Find the joint profile
                profile = np.empty(self.n_agents, dtype=int)
                for i, agent in enumerate(self.agents):
                    res = agent.select_action()
                    profile[i] = res

                # Utilities (cost) of each agent for the current joint profile
                utils = self.env.step(profile)

                # Update the algorithm of each agent
                for i, agent in enumerate(self.agents):
                    agent.algo.update(profile[i], utils[i])

                    # Store the reward and action.
                    agent.action_hist[t] = profile[i]

            self.store_data_round(round)

    def cal_alt_util_tab(self, player_id, round):
        """
        Calculate the utility of alternate actions for a given user.
        """
        joint_actions = self.experiment_data["agent_actions"][:, round, :]  # Joint actions of the agents for an entire horizon.
        util_table = np.empty((self.horizon, self.env.n_resources)) # Alternate utility table for player_id.

        for iter in range(self.horizon):
            joint_profile = joint_actions[:, iter]
            util_table[iter] = self.env.cal_alt_utilities(player_id, joint_profile)

        return util_table

    def cal_avg_bandit_regret(self, player_id):
        """
        Calculate regret following the definition of adversarial MAB.
        Args:
            player_id: Id of the player whose regret is calculated.
        """
        regrets = np.zeros(self.horizon)
        for round in range(self.n_rounds):
            foc_probs = self.experiment_data["agent_probs"][player_id, round, :, :]   # horizon x n_resources
            # Calculate the reward table for choosing alternate resource.
            util_table = self.cal_alt_util_tab(player_id, round)    # (horizon x n_resources).

            # Calculate cumsum of utilities.
            util_tab_cumsum = np.cumsum(util_table, axis=0)
            max_utils = np.max(util_tab_cumsum, axis=1) # Best reward (arm) in hindsight.
            mean_utils = np.array([np.dot(foc_probs[iter], util_table[iter]) for iter in range(self.horizon)])

            regrets += max_utils - mean_utils

        return regrets / self.n_rounds

if __name__ == "__main__":
    pass