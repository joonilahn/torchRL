import torch.nn as nn

from .mlp import ActionValueMLP
from ..builder import NETS


@NETS.register_module()
class DuelingMLP(ActionValueMLP):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super(DuelingMLP, self).__init__(
            state_dim, action_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.value_fn = nn.Linear(self.hidden_dim, 1)
        self.advantage_fn = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, state):
        features = self.net(state)
        value = self.value_fn(features)
        advantages = self.advantage_fn(features)
        return value + advantages - (advantages.mean() / self.action_dim)