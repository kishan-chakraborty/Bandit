"""
This module allows a flexible structure to create congestion game environment.
By default, the environment will deal with games with Bandit feedback but it can
be easily modified for full information setting.

Author: Kishan Chakraborty
"""

from collections import Counter
import numpy as np
from mab.environment.games.utils import AffineResource

class SingletonCongestionGame:
    """
    This environment corresponds to singleton events where an agent can select
    one arm at a time (not a subset).
    """
    def __init__(self, resources):
        """
        Args:
            resources: Resources (arm) to choose from. Choose only one arm at a time.
        """
        self.resources = resources
        self.n_resources = len(resources)

    def step(self, res_profile):
        """
        Args:
            actions: A list of resource indices chosen by each agent.
        Returns:
            A list of feedbacks observed by each agent based on load on each resource.
        """
        # Cal no. of times a resource is chosen by an agent.
        n_agents = len(res_profile)
        resource_loads = Counter(res_profile)

        # for each selected resource, cal its cost based on the load.
        for idx_load in resource_loads.items():
            idx, load = idx_load

            # Cost corresponding to each resource based on load (congestion).
            resource_loads[idx] = self.resources[idx].cost(load)

        cost_profile = np.asarray(res_profile, copy=True)  # cost incurred by each agent.

        # Corresponding to each resource, assign its cost for current iteration.
        for agent_idx in range(n_agents):
            agent_res = res_profile[agent_idx]
            cost_profile[agent_idx] = resource_loads[agent_res]

        return cost_profile
    
    def cal_utility(self, player_id, profile):
        """
        Calculate the utility of player idx, for the given joint action profile.
                                        u(a_i, a_{-i})
        Args:
            idx: Player index.
            profile: Current joint action profile.
        Return:
            Utility corresponding to user idx for the given profile.
        """
        chosen_resource = profile[player_id]    # Resource chosen by the user
        load = profile.count(chosen_resource)   # Load on the current resource
        util = self.resources[chosen_resource].cost(load)   # Utility of the chosen resource.

        return util
    
    def cal_alt_utilities(self, player_id, joint_profile):
        """
        Calculate alternate utility if the player had chosen a different resource.
                        u(a_i, a_{-i}) for all a_i's
        Args:
            player_id: Given player.
            profile: Current joint action profile.
        """
        current_load = Counter(joint_profile)   # Current load on the resources.
        current_res_id = joint_profile[player_id]   # index of the current res chosen by user.
        current_res = self.resources[current_res_id]    # Current res chosen by the user.
        current_util = current_res.cost(current_load[current_res_id])   # Utility of the player for current res.

        alt_utils = np.asarray(self.resources)

        for i, res in enumerate(self.resources):
            if i == current_res_id:
                alt_utils[current_res_id] = current_util
                
            alt_load = current_load[i] + 1 # If the user had chosen res, load -> load + 1
            alt_util = res.cost(alt_load)
            alt_utils[i] = alt_util

        return alt_utils
        
    
    def find_best_response(self, player_id, profile):
        """
        Find the best arm for the player (player_id) for the given action profile.
                                max_{a_i} u(a_i, a_{-i})
        Args:
            player_id: Given player.
            profile: Current joint action profile.
        """
        alt_utils = self.cal_alt_utilities(player_id, profile)
        best_res = alt_utils.argmin()   # Res with minimum cost.
        return best_res
    
    def cal_nash_eq(self, n_players, max_iter=1000):
        """
        Calculate the Nash equilibrium for the given no. of players.
        Args:
            n_players: No. of player playing the congestion game.
            max_iter: Max convergence criteria.
        """
        pass

if __name__ == "__main__":
    from utils import AffineResource

    res1 = AffineResource(1, 0)
    res2 = AffineResource(2, 0)
    res3 = AffineResource(0.5, 5)

    resources = [res1, res2, res3]

    game = SingletonCongestionGame(resources)

    res_profile = [2, 0, 1]
    print(game.step(res_profile))
    print(game.cal_utility(0, res_profile))
    print(game.cal_alt_utilities(0, res_profile))
    print(game.find_best_response(0, res_profile))