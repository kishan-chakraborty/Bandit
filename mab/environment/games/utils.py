"""
Utility functions supporting congestio game.
"""
class Resource:
    def __init__(self, cost_fun) -> None:
        """
        Args:
            cost_fun: A cost function. Preferably the values lie between 0 and 1 (Bandit algorithms).
        """
        self.cost_fun = cost_fun

    def cost(self, load):
        return self.cost_fun(load)
    
class AffineResource:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def cost(self, load):
        return self.a * load + self.b