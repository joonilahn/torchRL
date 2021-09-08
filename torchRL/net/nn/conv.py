import numpy as np
import torch
import torch.nn as nn

from ..builder import NETS


@NETS.register_module()
class BaseDQN(nn.Module):
    def __init__(self, action_dim=2, hidden_dim=512, in_channels=4):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels

    def forward(self, x):
        pass
    
    def predict(self, x):
        pass

    def predict_e_greedy(self, state, env, e):
        pass

    def conditional_unsqueeze(self, x):
        if len(x.size()) == 3 and x.size(0) == self.in_channels:
            return x.unsqueeze(0)
        else:
            return x

    def copy_parameters(self, model):
        self.load_state_dict(model.state_dict())

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

@NETS.register_module()
class DQN(BaseDQN):
    def __init__(self, action_dim=2, hidden_dim=512, in_channels=4):
        super(DQN, self).__init__(
            action_dim=action_dim, hidden_dim=hidden_dim, in_channels=in_channels
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

        self._init_params()

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1, -1)
        x = self.fc(x)
        return x

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
