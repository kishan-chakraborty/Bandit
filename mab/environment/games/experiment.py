"""
Experiment for multi-player game environment.
"""
import numpy as np
from mab.environment.experiment import Experiment as BaseExperiment

class Experiment(BaseExperiment):
    def __int__(self, env, agents, horizon, n_rounds, **kwargs):
        super().__init__(env, agents, horizon, n_rounds, **kwargs)

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

                # Utilities of each agent for the current joint profile
                utils = self.env.step(profile)

                # Update the algorithm of each agent
                for i, agent in enumerate(self.agents):
                    agent.algo.update(profile[i], utils[i])

                    # Store the reward and action.
                    agent.action_hist[t] = profile[i]
                    if self.store_rewards:
                        agent.reward_hist[t] = utils[i]

            self.store_data_round(round)

    def cal_alt_util_tab(self, player_id):
        """
        Calculate the utility of alternate actions for a given user.
        """
        util_table = np.empty((self.horizon, self.env.n_resources))

        for iter in range(self.horizon):
            joint_profile = [agent.action_hist[iter] for agent in self.agents]
            util_table[iter] = self.env.cal_alt_utilities(player_id, joint_profile)

        return util_table

    def cal_bandit_regret(self, player_id):
        """
        Calculate regret following the definition of adversarial MAB.
        Args:
            player_id: Id of the player whose regret is calculated.
        """
        foc_player = self.agents[player_id]     # Player on focus
        foc_probs = foc_player.algo.save_probs  # Probs of the foc player.
        foc_rewards = foc_player.reward_hist    # Rewards obtained by the foc player.

        # Calculate the reward table for choosing alternate resource (dim T x n_resources).
        util_table = self.cal_alt_util_tab(player_id)

        # Calculate cumsum of utilities.
        util_tab_cumsum = np.cumsum(util_table, axis=0)
        max_utils = np.max(util_tab_cumsum, axis=1) # Best reward (arm) in hindsight.
        mean_utils = np.array([np.dot(foc_probs[iter], foc_rewards[iter]) for iter in range(self.horizon)])

        # Per step regret
        regret = max_utils - mean_utils
        return regret
    
if __name__ == "__main__":
    pass