import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .mlp import ActionValueMLP, build_mlp
from ..builder import NETS


@NETS.register_module()
class ActorMLP(ActionValueMLP):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128, num_layers=2):
        super(ActorMLP, self).__init__(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = self._to_tensor(state)
        return F.softmax(self.net(state), dim=-1)

@NETS.register_module()
class CriticMLP(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, num_layers=2):
        super(CriticMLP, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = build_mlp(num_layers, state_dim, hidden_dim, 1)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = self._to_tensor(state)
        return self.net(state)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

@NETS.register_module()
class ActorCriticMLP(nn.Module):    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128, num_layers=2):
        super(ActorCriticMLP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        num_layers = num_layers if num_layers > 1 else 1
        self.feature_extractor = build_mlp(num_layers, state_dim, hidden_dim, hidden_dim)
        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(-1)
        )
        self.critic_layer = nn.Linear(hidden_dim, 1)

    def forward(self, states):
        return self.forward_actor(states)

    def forward_actor(self, states):
        if not isinstance(states, torch.Tensor):
            states = self._to_tensor(states)
        features = self.feature_extractor(states)
        return self.actor_layer(features)
    
    def forward_critic(self, states):
        if not isinstance(states, torch.Tensor):
            states = self._to_tensor(states)
        features = self.feature_extractor(states)
        return self.critic_layer(features)

    def predict(self, state):
        probs =  self.forward_actor(state)
        action = Categorical(probs).sample().item()
        return action

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)