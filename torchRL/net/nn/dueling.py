import numpy as np
import torch
import torch.nn as nn

from ..builder import NETS
from .conv import BaseDQN
from .mlp import ActionValueMLP


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


@NETS.register_module()
class DuelingDQN(BaseDQN):
    def __init__(self, action_dim=2, hidden_dim=512, in_channels=4):
        super(DuelingDQN. self).__init__(
            action_dim=action_dim, hidden_dim=hidden_dim, in_channels=in_channels
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 2 * hidden_dim, 7, stride=1, bias=False),
            nn.ReLU(),
        )
        self.value_fn = nn.Linear(hidden_dim, 1)
        self.advantage_fn = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x):
        x = self.cnn(x)
        feats_a, feats_v = torch.split(x, self.hidden_dim, 1)
        feats_a = feats_a.flatten(1, -1)
        feats_v = feats_v.flatten(1, -1)
        value = self.value_fn(feats_v)
        advantages = self.advantage_fn(feats_a)

        return (
            value + advantages - (advantages.mean(-1, keepdim=True) / self.action_dim)
        )

    def predict(self, state):
        state = self.conditional_unsqueeze(state)
        with torch.no_grad():
            q_values = self.forward(state)
        q_value, action = q_values.max(1)
        return int(action), float(q_value)

    def predict_e_greedy(self, state, env, e):
        action, q_value = self.predict(state)
        if np.random.rand() < e:
            action = env.action_space.sample()
        return action, q_value
