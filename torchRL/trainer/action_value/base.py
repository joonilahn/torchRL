from collections import deque

import numpy as np
import torch

from ...net import build_Qnet
from ...optim import build_optimizer
from ...scheduler import build_scheduler
from ..base import BaseTrainer
from ..builder import TRAINERS


@TRAINERS.register_module()
class QTrainer(BaseTrainer):
    """Base class for Q (action value) Trainer."""

    def __init__(self, env, cfg):
        super(QTrainer, self).__init__(env, cfg)
        self.net = build_Qnet(cfg.NET)

        # load pretrained weight
        if cfg.TRAIN.PRETRAINED != "":
            self.load_model(cfg.TRAIN.PRETRAINED)

        # move model to gpu if needed
        if self.use_gpu:
            self.net.to(torch.device("cuda"))

        self.optimizer = build_optimizer(self.net, cfg.OPTIMIZER)
        self.epsilon_scheduler = build_scheduler(cfg.SCHEDULER)
        self.epsilon = self.epsilon_scheduler.epsilon
        self.q_values = deque(maxlen=cfg.TRAIN.HISTORY_SIZE)

        # initialize log dict
        self._init_log_dict()

    def update_e_greedy_param(self):
        """Get epsilon value based on linear annealing."""
        if self.cfg.TRAIN.TRAIN_BY_EPISODE:
            n = self.episode_num
        else:
            n = self.frame_num
        if n > self.cfg.TRAIN.START_TRAIN:
            self.epsilon = self.epsilon_scheduler.step(n)
        return self.epsilon

    def save_model(self):
        self._save_model(self.net)

    def load_model(self, weight):
        self.net.load_state_dict(torch.load(weight))
        self.logger.info("Loaded pretrained weight.")

    def _init_log_dict(self):
        # Default log info
        self.log_dict = {
            "Episode": self.episode_num,
            "Frame Num": self.frame_num,
            "Train Iter": self.train_iters,
            "epsilon": self.epsilon,
            "Reward": 0,
            "Steps": self.steps,
            "Loss": 0.0,
            f"Last {self.cfg.TRAIN.HISTORY_SIZE} Avg Rewards": 0.0,
            "Avg Loss": 0.0,
        }

    def log_info(self, additional_log_dict=None):
        # update log info
        self.log_dict = {
            "Episode": self.episode_num,
            "Frame Num": self.frame_num,
            "Train Iter": self.train_iters,
            "epsilon": self.epsilon,
            "Reward": self.rewards_history[-1],
            "Steps": self.steps,
            "Loss": self.losses[-1],
            f"Last {self.cfg.TRAIN.HISTORY_SIZE} Avg Rewards": np.mean(self.rewards_history),
            "Avg Loss": np.mean(self.losses),
        }
        if additional_log_dict:
            self.log_dict = {**self.log_dict, **additional_log_dict}
        
        # log info
        self._log_info()
        