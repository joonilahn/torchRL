import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .mlp import build_mlp
from .conv import DQN, BaseDQN
from ..builder import NETS


@NETS.register_module()
class ActorCriticMLP(nn.Module):    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128, num_layers=2):
        super(ActorCriticMLP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        num_layers = num_layers if num_layers > 1 else 1
        self.actor_layer = build_mlp(num_layers, state_dim, hidden_dim, action_dim)
        self.critic_layer = build_mlp(num_layers, state_dim, hidden_dim, 1)

    def forward(self, states):
        return self.forward_actor(states)

    def forward_actor(self, states):
        if not isinstance(states, torch.Tensor):
            states = self._to_tensor(states)
        return F.softmax(self.actor_layer(states), dim=-1)
    
    def forward_critic(self, states):
        if not isinstance(states, torch.Tensor):
            states = self._to_tensor(states)
        return self.critic_layer(states)

    def predict(self, state):
        probs =  self.forward_actor(state)
        action = Categorical(probs).sample().item()
        return action

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


@NETS.register_module()
class ActorCriticCNN(BaseDQN):    
    def __init__(self, action_dim=4, hidden_dim=512, in_channels=4):
        super(ActorCriticCNN, self).__init__(
            action_dim=action_dim, hidden_dim=hidden_dim, in_channels=in_channels
        )

        self.actor_layer = DQN(action_dim=action_dim, hidden_dim=hidden_dim, in_channels=in_channels)
        self.critic_layer = DQN(action_dim=1, hidden_dim=hidden_dim, in_channels=in_channels)

    def forward(self, states):
        return self.forward_actor(states)

    def forward_actor(self, states):
        return F.softmax(self.actor_layer(states), dim=-1)
    
    def forward_critic(self, states):
        return self.critic_layer(states)

    def predict(self, state):
        state = self.actor_layer.conditional_unsqueeze(state)
        with torch.no_grad():
            probs = self.forward_actor(state)
        action = Categorical(probs).sample().item()
        return action