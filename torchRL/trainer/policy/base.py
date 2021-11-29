from collections import deque

import numpy as np
import torch

from ...data import build_dataset
from ...net import build_ActorCritic
from ...optim import build_optimizer
from ..base import BaseTrainer
from ..builder import TRAINERS


@TRAINERS.register_module()
class BasePolicyGradientTrainer(BaseTrainer):
    """Actor-Critic Policy Gradient Trainer."""

    def __init__(self, env, cfg):
        super(BasePolicyGradientTrainer, self).__init__(env, cfg)
        self.net = build_ActorCritic(cfg.NET)

        # load pretrained weight
        if cfg.TRAIN.PRETRAINED != "":
            self.load_model(cfg.TRAIN.PRETRAINED)

        # move model to gpu
        if self.use_gpu:
            self.net.to(torch.device("cuda"))

        self.buffer = build_dataset(cfg)
        self.optimizer = build_optimizer(self.net, cfg.OPTIMIZER)
        self.losses_actor = deque(maxlen=cfg.TRAIN.HISTORY_SIZE)
        self.losses_critic = deque(maxlen=cfg.TRAIN.HISTORY_SIZE)

    def run_single_episode(self):
        """Run a single episode for episodic environment.
        Every observation is stored in the memory buffer.
        """
        game_rewards = 0.0
        done = False
        done_life = False
        state = self.env.reset()
        self.steps = 0

        while not done:
            self.steps += 1
            self.frame_num += 1

            # clear the buffer
            self.buffer.clear()

            for _ in range(self.cfg.TRAIN.BATCH_SIZE):
                # get action using epsilon greedy
                action = self.net.predict(
                    self.pipeline(state), num_output=self.num_output
                )

                # take the action (step)
                next_state, reward, done, info = self.env.step(action)

                # For Atari, stack the next state to the current states
                if self.cfg.ENV.TYPE == "Atari":
                    state[:, :, 4] = next_state

                    # check whether game's life has changed
                    if self.steps == 1:
                        self.set_init_lives(info)
                    done_life = self.is_done_for_life(info, reward)

                # update reward
                game_rewards += reward
                reward *= self.cfg.ENV.REWARD_SCALE
                reward = np.clip(reward, -1, 1)

                # update
                if isinstance(action, list):
                    action = action[0]
                self.update(state, next_state, reward, action, done or done_life)

                # set the current state to the next state (state <- next_state)
                if self.cfg.ENV.TYPE == "Atari":
                    state = np.concatenate(
                        [state[:, :, 1:], np.expand_dims(next_state, axis=2)], axis=2
                    )
                else:
                    state = next_state

                if done:
                    break

                self.frame_num += 1

        return game_rewards

    def update(self, state, next_state, reward, action, done):
        pass

    def _train(self):
        """Train the Policy."""
        for episode_num in range(1, self.cfg.TRAIN.NUM_EPISODES + 1):
            self.episode_num = episode_num
            rewards = self.run_single_episode()
            self.rewards_history.append(rewards)

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info()

            # save model
            if self.cfg.LOGGER.SAVE_MODEL:
                if episode_num > 1:
                    if episode_num % self.cfg.LOGGER.SAVE_MODEL_INTERVAL == 0:
                        self._save_model(self.net)

                    # save the best model
                    if self.best_avg_reward < np.mean(self.rewards_history):
                        self.best_avg_reward = np.mean(self.rewards_history)
                        self._save_model(
                            self.net,
                            suffix=f"best_{int(self.best_avg_reward)}",
                        )

            # evaluate
            if self.cfg.TRAIN.get("EVALUATE_INTERVAL", None) and self.env_eval and episode_num % self.cfg.TRAIN.EVALUATE_INTERVAL == 0:
                self.evaluate()

            if self.early_stopping_condition():
                return True

    def log_info(self, additional_log_dict=None):
        """log current information for the training."""
        self.log_dict = {
            "Episode": self.episode_num,
            "Frame Num": self.frame_num,
            "Train Iter": self.train_iters,
            "Reward": self.rewards_history[-1],
            "Steps": self.steps,
            "Loss Total": self.losses[-1],
            "Loss Actor": self.losses_actor[-1],
            "Loss Critic": self.losses_critic[-1],
            f"Last {self.cfg.TRAIN.HISTORY_SIZE} Avg Rewards": np.mean(
                self.rewards_history
            ),
            "Avg Loss": np.mean(self.losses),
        }

        if additional_log_dict:
            self.log_dict = {**self.log_dict, **additional_log_dict}

        # log info
        self._log_info()

    def save_model(self):
        self._save_model(self.net)

    def load_model(self, weight):
        self.net.load_state_dict(torch.load(weight))
        self.logger.info("Loaded pretrained weight.")
