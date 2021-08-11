import numpy as np
import torch.nn as nn
from torch.optim import Adam

from .builder import TRAINERS


@TRAINERS.register_module()
class BaseTrainer:
    """Base class for All Trainers."""
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

    def run_single_episode(self):
        """Run a single episode for episodic environment."""
        pass

    def train(self):
        """Train the q network"""
        pass

    def estimate_target_values(self, next_states):
        """Estimate target values using TD(0), MC, or TD(lambda)."""
        pass

    def early_stopping_condition(self):
        """Early stop if running mean score is larger the the terminate threshold."""
        if (
            np.mean(self.steps_history[-self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS :])
            > self.cfg.TRAIN.AVG_STEPS_TO_TERMINATE
        ):
            print(
                f"Last {self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS} avg steps exceeded "
                f"{self.cfg.TRAIN.AVG_STEPS_TO_TERMINATE} times. Quit training."
            )
            return True
        else:
            return False

    def log_info(self, episode_num):
        """log current information for the training."""
        print(
            f"Episode: {episode_num + 1}, Step: {self.steps_history[-1]}, "
            f"epsilon: {self.epsilon:.2f}, "
            f"Last {self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS} "
            f"Avg Steps: {np.mean(self.steps_history[-self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS:])}, "
            f"Loss: {self.losses[-1]:.6f}, "
            f"Last {self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS} "
            f"Avg Loss: {np.mean(self.losses[-self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS:]):.6f}"
        )