import torch

from ..builder import TRAINERS
from .sarsa import SARSATrainer


@TRAINERS.register_module()
class QLearningTrainer(SARSATrainer):
    """Trainer class for Vanilla Q-Learning."""

    def __init__(self, env, cfg):
        super(QLearningTrainer, self).__init__(env, cfg)

    def estimate_target_values(self, next_state):
        """Estimate the target value based on Q-Learning.

        target <- R + gamma * max_a_Q(S',a)
        """
        with torch.no_grad():
            value_target = self.net(next_state).max(1)[0]
        return value_target
