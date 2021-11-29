import os
from collections import deque

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from ..data import build_pipeline, to_grayscale_and_resize
from ..env import EnvWrapper
from ..loss import build_loss
from ..utils import TensorboardLogger, get_logger

from .builder import TRAINERS


@TRAINERS.register_module()
class BaseTrainer:
    """Base class for All Trainers."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.use_gpu = cfg.USE_GPU
        self.net = None
        self.pipeline = build_pipeline(cfg.DATASET)
        self.criterion = build_loss(cfg.LOSS_FN)
        self.optimizer = None

        self.logger = self._get_logger()
        self.tb_logger = self._get_tb_logger()
        self.log_dict = {}

        # loggable informations
        self.rewards_history = deque(maxlen=cfg.TRAIN.HISTORY_SIZE)
        self.losses = deque([np.nan], maxlen=cfg.TRAIN.HISTORY_SIZE)
        self.episode_num = 0
        self.frame_num = 0
        self.train_iters = 0
        self.steps = 0
        self.game_has_life = False
        self.curr_lives = 0
        self.best_avg_reward = -np.inf
        self.num_output = cfg.ENV.get("NUM_OUTPUT", 1)

        # for evaluation
        if cfg.ENV.get("EVAL_DATA", '') != '':
            self.env_eval = EnvWrapper(cfg.ENV, train_data=cfg.ENV.EVAL_DATA)
        else:
            self.env_eval = None

    def run_single_episode(self):
        """Run a single episode for episodic environment."""
        pass

    def train(self):
        """Train"""
        self.logger.info(f"Start training with following settings.\n{self.cfg}")
        self._train()
        if self.cfg.LOGGER.SAVE_MODEL:
            self.save_model()
        
        # close
        self.env.close()

        if self.tb_logger:
            self.tb_logger.close()

    def _train(self):
        """Train"""
        pass

    def evaluate(self, e=0.0):
        reward_total = 0
        reward_per_life = 0
        done = False
        done_life = True
        state_init = self.env_eval.reset()
        state = state_init
        steps = 0

        while not done:
            steps += 1

            # get action (e-greedy)
            try:
                action = self.net.predict_e_greedy(self.pipeline(state), self.env, e, num_output=self.num_output)[
                    0]
            except:
                action = self.net.predict(self.pipeline(state), num_output=self.num_output)

            # take the action (step)
            next_state, reward, done, lives = self.env_eval.env.step(action)

            # save frame
            if self.cfg.ENV.TYPE == "Atari":
                next_state = to_grayscale_and_resize(next_state)

            # For Atari, stack the next state to the current states
            if self.cfg.ENV.TYPE == "Atari":
                state[:, :, 4] = next_state

            reward_total += reward
            reward_per_life += reward

            # check whether game's life has changed
            if (steps == 1) and (self.cfg.ENV.TYPE == "Atari"):
                self.set_init_lives(lives)
                done_life = self.is_done_for_life(lives, reward)

            # set the current state to the next state (state <- next_state)
            if self.cfg.ENV.TYPE == "Atari":
                state = np.concatenate(
                    [state[:, :, 1:], np.expand_dims(next_state, axis=2)], axis=2
                )
            else:
                state = next_state

            if steps == self.env_eval.env.__max_episode_steps:
                break

        self.logger.info(f"Evaluation Reward: {reward_total}")
        if self.tb_logger:
            self.tb_logger.log_info("Eval Reward", reward_total, self.episode_num)

    def estimate_target_values(self, next_states):
        """Estimate target values using TD(0), MC, or TD(lambda)."""
        pass
    
    def gradient_descent(self, params, loss):
        """Update parameters using gradient descent."""
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.OPTIMIZER.CLIP_GRAD:
            params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
            clip_grad_norm_(params, self.cfg.OPTIMIZER.CLIP_GRAD_VALUE)
        self.optimizer.step()

    def early_stopping_condition(self):
        """Early stop if running mean score is larger the the terminate threshold."""
        if (
            np.mean(self.rewards_history)
            > self.cfg.TRAIN.AVG_REWARDS_TO_TERMINATE
        ):
            self.logger.info(
                f"Last {self.cfg.TRAIN.HISTORY_SIZE} avg rewards exceeded "
                f"{self.cfg.TRAIN.AVG_REWARDS_TO_TERMINATE} times. Quit training."
            )
            return True
        else:
            return False

    def _get_logger(self):
        if self.cfg.LOGGER.LOG_FILE:
            if not os.path.isdir(self.cfg.LOGGER.OUTPUT_DIR):
                os.makedirs(self.cfg.LOGGER.OUTPUT_DIR, exist_ok=True)
            log_file = os.path.join(
                self.cfg.LOGGER.OUTPUT_DIR, self.cfg.LOGGER.LOG_NAME + ".txt"
            )
        else:
            log_file = None

        return get_logger("torchRL", log_file=log_file)

    def _get_tb_logger(self):
        if self.cfg.LOGGER.LOG_TENSORBOARD:
            if not os.path.isdir(self.cfg.LOGGER.OUTPUT_DIR):
                os.makedirs(self.cfg.LOGGER.OUTPUT_DIR, exist_ok=True)
            return TensorboardLogger(self.cfg.LOGGER.OUTPUT_DIR)
        return None

    def _log_info(self):
        """log current information for the training."""
        log_msg = ""

        def parse_text_for_log(v):
            if not isinstance(v, float):
                return v
            if v > 0.1:
                return f"{v:.2f}"
            else:
                return f"{v:.4f}"
        
        # update log_msg
        is_first_msg = True
        for k, v in self.log_dict.items():
            if is_first_msg:
                log_msg += f"{k}: {parse_text_for_log(v)}"
                is_first_msg = False

            log_msg += f", {k}: {parse_text_for_log(v)}"
            if "time" in k:
                log_msg += "s"
            
            if self.tb_logger:
                self.tb_logger.log_info(k, v, self.frame_num)
        
        # log info
        self.logger.info(log_msg)
        
    def _save_model(self, model, suffix=None):
        """Save model weight."""
        if not os.path.isdir(self.cfg.LOGGER.OUTPUT_DIR):
            os.mkdir(self.cfg.LOGGER.OUTPUT_DIR)

        if suffix is None:
            suffix = (
                f"episode_{self.episode_num}"
                if self.cfg.TRAIN.TRAIN_BY_EPISODE
                else f"frame_{self.frame_num}"
            )

        # save new best checkpoints
        if "best" in suffix:
            suffix += f"_frame_{self.frame_num}"
            for f in os.listdir(self.cfg.LOGGER.OUTPUT_DIR):
                if (f.endswith("pth")) and ("best" in f):
                    os.remove(os.path.join(self.cfg.LOGGER.OUTPUT_DIR, f))
        
        save_path = os.path.join(
            self.cfg.LOGGER.OUTPUT_DIR,
            f"{self.cfg.LOGGER.LOG_NAME}_{suffix}.pth",
        )
        torch.save(model.state_dict(), save_path)
        self.logger.info(f"Saved model to {save_path}")

    def save_model(self):
        pass

    def set_device(self, x):
        if self.use_gpu:
            return x.to(torch.device("cuda"))
        return x

    def convert_input_type(self, x):
        if isinstance(x, (int, float)):
            x = torch.tensor([x], dtype=torch.float32)
        elif isinstance(x, bool):
            x = torch.tensor([int(x)], dtype=torch.int8)
        else:
            raise NotImplementedError
        return self.set_device(x)

    def set_init_lives(self, lives):
        self.curr_lives = lives.get("ale.lives", None)
        if self.curr_lives is None:
            self.game_has_life = False
        else:
            if self.curr_lives > 0:
                self.game_has_life = True
            else:
                self.game_has_life = False
    
    def is_done_for_life(self, lives, reward):
        # games that have lives
        if self.game_has_life:
            if lives["ale.lives"] < self.curr_lives:
                done = True
            else:
                done = False
            self.curr_lives = lives["ale.lives"]
    
        else:
            if reward < 0:
                done = True
            else:
                done = False
        
        return done