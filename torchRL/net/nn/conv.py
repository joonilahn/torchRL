import numpy as np
import torch
import torch.nn as nn

from ..builder import NETS

@NETS.register_module()
class SmallCNN(nn.Module):
    def __init__(self, action_dim=2, hidden_dim=256, in_channels=1):
        super().__init__()
        self.action_dim = action_dim
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2592, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1, -1)
        x = self.fc(x)
        return x

    def predict(self, state):
        state = self.conditional_unsqueeze(state)
        with torch.no_grad():
            actions = self.forward(state)
        return int(actions.argmax(1)[0])

    def predict_e_greedy(self, state, env, e):
        if np.random.rand() > e:
            action = self.predict(state)
        else:
            action = env.action_space.sample()
        return action

    def conditional_unsqueeze(self, x):
        if len(x.size()) == 3 and x.size(0) == self.in_channels:
            return x.unsqueeze(0)
        else:
            return x

    def copy_parameters(self, model):
        self.load_state_dict(model.state_dict())