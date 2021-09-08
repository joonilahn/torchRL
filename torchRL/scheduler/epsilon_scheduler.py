from .builder import SCHEDULERS

@SCHEDULERS.register_module()
class LinearAnnealingScheduler:
    def __init__(self, cfg):
        self.e_greedy_min, self.e_greedy_max = cfg.EPSILON_GREEDY_MINMAX
        self.start_decay, self.stop_decay = cfg.DECAY_PERIOD
        self.epsilon = self.e_greedy_max

    def step(self, n):
        """Get epsilon value based on linear annealing"""
        self.epsilon = max(
            self.e_greedy_min,
            self.e_greedy_max
            - (n - self.start_decay)
            * (self.e_greedy_max - self.e_greedy_min)
            / (self.stop_decay - 1)
        )
        return self.epsilon