import torch.nn as nn


class PolicyNet(nn.Module):
    """Base class for Policy Network."""
    def __init__(self):
        pass

    def forward(self, state):
        pass