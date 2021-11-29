import numpy as np
import torch
import torch.nn as nn

from ..builder import NETS


@NETS.register_module()
class ActionValueMLP(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128, num_layers=2):
        super(ActionValueMLP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.net = build_mlp(num_layers, state_dim, hidden_dim, action_dim)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = self._to_tensor(state)
        return self.net(state)

    def _to_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return x

    def predict(self, state, num_output=1):
        if not isinstance(state, torch.Tensor):
            state = self._to_tensor(state)
        with torch.no_grad():
            q_values = self.forward(state).cpu()
        if num_output > 1:
            q_values, actions = q_values.sort(descending=True)
            return actions[0, :num_output].tolist(), float(
                q_values[0, :num_output].mean()
            )
        else:
            q_value, action = q_values.max(1)
            return int(action), float(q_value)

    def predict_e_greedy(self, state, env, e, num_output=1):
        action, q_value = self.predict(state, num_output=num_output)
        if np.random.rand() < e:
            action = env.action_space.sample()
            if num_output == 1 and isinstance(action, list):
                action = action[0]
        return action, q_value

    def copy_parameters(self, model):
        self.load_state_dict(model.state_dict())


def build_mlp(num_layers, input_dim, hidden_dim, output_dim):
    layers = []
    if num_layers == 1:
        return nn.Linear(input_dim, output_dim)

    for i in range(1, num_layers + 1):
        if i == 1:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
        elif i < num_layers:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)
