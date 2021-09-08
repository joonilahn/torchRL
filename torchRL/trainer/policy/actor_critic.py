from collections import deque
from copy import deepcopy

import numpy as np
import torch

from ...data import build_dataset
from ...net import build_ActorCritic
from ...optim import build_optimizer
from ..base import BaseTrainer
from ..builder import TRAINERS


@TRAINERS.register_module()
class ActorCriticTrainer(BaseTrainer):
    """Actor-Critic Policy Gradient Trainer."""

    def __init__(self, env, cfg):
        super(ActorCriticTrainer, self).__init__(env, cfg)
        self.net = build_ActorCritic(cfg.NET)

        # load pretrained weight
        if cfg.TRAIN.PRETRAINED != "":
            self.load_model(cfg.TRAIN.PRETRAINED)

        # move model to gpu
        if self.use_gpu:
            self.net.to(torch.device("cuda"))

        self.buffer = build_dataset(cfg)
        self.optimizer = build_optimizer(self.net, cfg.OPTIMIZER)
        self.losses_actor = deque([np.nan], maxlen=cfg.TRAIN.HISTORY_SIZE)
        self.losses_critic = deque([np.nan], maxlen=cfg.TRAIN.HISTORY_SIZE)

    def run_single_episode(self):
        """Run a single episode for episodic environment.
        Every observation is stored in the memory buffer.
        """
        game_rewards = 0.0
        done = False
        done_life = False
        state = self.env.reset()
        self.steps = 0
        self.net.train()

        while not done:
            self.steps += 1
            self.frame_num += 1

            # clear the buffer
            self.buffer.clear()

            for _ in range(self.cfg.TRAIN.BATCH_SIZE):
                # get action using epsilon greedy
                action = self.net.predict(self.pipeline(state))

                # take the action (step)
                next_state, reward, done, info = self.env.step(action)

                # For Atari, stack the next state to the current states
                if self.cfg.ENV.TYPE == "Atari":
                    state[:, :, 4] = next_state

                # check whether game's life has changed
                if (self.steps == 1) and (self.cfg.ENV.TYPE == "Atari"):
                    self.set_init_lives(info)
                done_life = self.is_done_for_life(info, reward)

                # update reward
                game_rewards += reward
                reward *= self.cfg.ENV.REWARD_SCALE
                reward = np.clip(reward, -1, 1)

                # stack the data to the buffer (memory)
                self.buffer.stack(
                    (deepcopy(state), next_state, reward, action, done or done_life)
                )

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

            self.update_batch()

        return game_rewards

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        # estimate TD errors
        with torch.no_grad():
            target = (
                reward
                + self.cfg.TRAIN.DISCOUNT_RATE
                * self.net.forward_critic(self.pipeline(next_state)).squeeze(
                    -1
                )
                * ~done
            )
            td_error = target - self.net.forward_critic(
                self.pipeline(state)
            ).squeeze(-1)

        # calculate loss for the actor
        pred = self.net.forward_actor(self.pipeline(state))[0][
            action
        ].unsqueeze(0)
        loss_actor = -(torch.log(pred) * td_error.detach())

        # calculate loss for the critic
        preds_value = self.net.forward_critic(self.pipeline(state))
        loss_critic = self.criterion(preds_value, target.squeeze(-1).detach())

        # update parameters for the critic
        loss = loss_actor + loss_critic
        self.gradient_descent(self.net.parameters(), loss)

        # update loss history
        self.losses.append(float(loss))
        self.losses_actor.append(float(loss_actor))
        self.losses_critic.append(float(loss_critic))
        self.train_iters += 1

    def update_batch(self):
        """Update (train) the policy network."""
        for _ in range(self.cfg.TRAIN.NUM_ITERS_PER_TRAIN):
            # load data
            data = self.buffer.load(len(self.buffer))
            states = data["states"]
            next_states = data["next_states"]
            rewards = self.set_device(data["rewards"])
            actions = self.set_device(data["actions"])
            dones = self.set_device(data["dones"])

            # estimate TD errors
            with torch.no_grad():
                targets = (
                    rewards
                    + self.cfg.TRAIN.DISCOUNT_RATE
                    * self.net.forward_critic(next_states).squeeze(-1)
                    * (1 - dones)
                )
                td_errors = targets - self.net.forward_critic(states).squeeze(
                    -1
                )

            # calculate loss for the actor
            preds = (
                self.net.forward_actor(states).gather(1, actions).squeeze(-1)
            )
            loss_actor = -torch.log(preds) * td_errors
            loss_actor = loss_actor.mean()

            # calculate loss for the critic
            preds_values = self.net.forward_critic(states).squeeze(-1)
            loss_critic = self.criterion(preds_values, targets)

            # update parameters for the critic
            loss = loss_actor + loss_critic
            self.gradient_descent(self.net.parameters(), loss)

            self.losses.append(float(loss))
            self.losses_actor.append(float(loss_actor))
            self.losses_critic.append(float(loss_critic))
            self.train_iters += 1

    def _train(self):
        """Train the Policy."""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
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
