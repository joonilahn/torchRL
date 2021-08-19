from torchRL.trainer.base import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from ...net import build_Qnet
from ...data import build_pipeline
from ..builder import TRAINERS
from ..base import BaseTrainer


@TRAINERS.register_module()
class QTrainer(BaseTrainer):
    """Base class for Q (action value) Trainer."""

    def __init__(self, env, cfg):
        super(QTrainer, self).__init__(env, cfg)
        self.q_net = build_Qnet(cfg.NET)
        if self.use_gpu:
            self.q_net.to(torch.device("cuda"))
        self.pipeline = build_pipeline(cfg.DATASET)
        self.criterion = nn.MSELoss()
        self.lr = self.cfg.TRAIN.LEARNING_RATE
        self.optimizer = Adam(self.q_net.parameters(), self.lr)
        self.e_greedy_min, self.e_greedy_max = cfg.TRAIN.EPSILON_GREEDY_MINMAX
        self.stop_decay_epsilon = cfg.TRAIN.STOP_DECAY_EPSILON
        self.epsilon = self.e_greedy_max
        self.losses = [np.inf]
        self.global_iters = 0

    def e_greedy_linear_annealing(self, episode_num):
        """Get epsilon value based on linear annealing."""
        self.epsilon = max(
            self.e_greedy_min,
            self.e_greedy_max
            - episode_num
            * (self.e_greedy_max - self.e_greedy_min)
            / (self.stop_decay_epsilon - 1),
        )
        return self.epsilon

    def save_model(self, suffix=""):
        self._save_model(self.q_net, suffix=suffix)
