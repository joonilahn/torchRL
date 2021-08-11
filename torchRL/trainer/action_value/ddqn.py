import torch

from ..builder import TRAINERS
from .dqn import DQNTrainer


@TRAINERS.register_module()
class DDQNTrainer(DQNTrainer):
    """Trainer class for Double DQN."""

    def __init__(self, env, cfg):
        super(DDQNTrainer, self).__init__(env, cfg)

    def estimate_target_values(self, next_states):
        """Estimate the target values based on Double DQN.
        Target network is used to estimate values for the next states.

        A' <- Q_target(S', epsilon;theta)
        targets <- R + gamma * Q_action(S',A')
        """
        with torch.no_grad():
            # choose actions of the next states based on the target net argmaxQ(s',a;\theta)
            actions_next = self.q_target(next_states).argmax(1).unsqueeze(-1)
            # estimate the next action values q(s',a';\theta^-)
            values_target = self.q_net(next_states).gather(1, actions_next).squeeze(-1)

        return values_target
