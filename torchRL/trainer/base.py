import os
import numpy as np
import torch

from .builder import TRAINERS
from ..utils import get_logger


@TRAINERS.register_module()
class BaseTrainer:
    """Base class for All Trainers."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.use_gpu = cfg.USE_GPU
        self.logger = self._get_logger()
        self.rewards_history = []

    def run_single_episode(self):
        """Run a single episode for episodic environment."""
        pass

    def train(self):
        """Train"""
        self._train()
        if self.cfg.LOGGER.SAVE_MODEL:
            self.save_model(suffix="last")

    def _train(self):
        """Train"""
        pass

    def estimate_target_values(self, next_states):
        """Estimate target values using TD(0), MC, or TD(lambda)."""
        pass

    def early_stopping_condition(self):
        """Early stop if running mean score is larger the the terminate threshold."""
        if (
            np.mean(self.rewards_history[-self.cfg.TRAIN.AVERAGE_SIZE :])
            > self.cfg.TRAIN.AVG_REWARDS_TO_TERMINATE
        ):
            self.logger.info(
                f"Last {self.cfg.TRAIN.AVERAGE_SIZE} avg rewards exceeded "
                f"{self.cfg.TRAIN.AVG_REWARDS_TO_TERMINATE} times. Quit training."
            )
            return True
        else:
            return False

    def _get_logger(self):
        if self.cfg.LOGGER.LOG_FILE:
            if not os.path.isdir(self.cfg.LOGGER.OUTPUT_DIR):
                os.mkdir(self.cfg.LOGGER.OUTPUT_DIR)
            log_file = os.path.join(self.cfg.LOGGER.OUTPUT_DIR, self.cfg.LOGGER.LOG_NAME + ".txt")
        else:
            log_file = None

        return get_logger("torchRL", log_file=log_file)

    def log_info(self, episode_num):
        """log current information for the training."""
        self.logger.info(
            f"Episode: {episode_num + 1}, Reward: {self.rewards_history[-1]}, "
            f"epsilon: {self.epsilon:.2f}, "
            f"Last {self.cfg.TRAIN.AVERAGE_SIZE} "
            f"Avg Rewards: {np.mean(self.rewards_history[-self.cfg.TRAIN.AVERAGE_SIZE:]):.2f}, "
            f"Loss: {self.losses[-1]:.6f}, "
            f"Last {self.cfg.TRAIN.AVERAGE_SIZE} "
            f"Avg Loss: {np.mean(self.losses[-self.cfg.TRAIN.AVERAGE_SIZE:]):.6f}"
        )

    def _save_model(self, model, suffix=""):
        """Save model weight."""
        if not os.path.isdir(self.cfg.LOGGER.OUTPUT_DIR):
            os.mkdir(self.cfg.LOGGER.OUTPUT_DIR)
        save_path = os.path.join(self.cfg.LOGGER.OUTPUT_DIR, f"{self.cfg.LOGGER.LOG_NAME}_episode_{suffix}.pth")
        torch.save(model.state_dict(), save_path)
        self.logger.info(f"Saved model to {save_path}")

    def save_model(self, suffix=""):
        pass

    def set_device(self, x):
        if self.use_gpu:
            return x.to(torch.device("cuda"))
        return x
