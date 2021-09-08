import time
from copy import deepcopy

import numpy as np
import torch

from ...data import build_dataset
from ...net import build_Qnet
from ...utils import freeze_net
from ..builder import TRAINERS
from .qlearning import QLearningTrainer


@TRAINERS.register_module()
class DQNTrainer(QLearningTrainer):
    """Trainer class for Deep Q-Networks"""

    def __init__(self, env, cfg):
        super(QLearningTrainer, self).__init__(env, cfg)
        self.target_net = build_Qnet(cfg.NET)
        if self.use_gpu:
            self.target_net.to(torch.device("cuda"))
        self.target_net.copy_parameters(self.net)
        self.target_net = freeze_net(self.target_net)
        self.buffer = build_dataset(cfg)

    def sync_target_net(self):
        self.target_net.copy_parameters(self.net)
        self.logger.info("Copied Q-Net Parameters to Q-TargetNet")

    def run_single_episode(self):
        """Run a single episode for episodic environment.
        Every observation is stored in the memory buffer.
        """
        game_rewards = 0.0
        done = False
        done_life = False
        state = self.env.reset()
        e = self.update_e_greedy_param()
        self.steps = 0

        while not done:
            self.steps += 1
            self.frame_num += 1
            
            # get action using epsilon greedy
            action, q_value = self.net.predict_e_greedy(self.pipeline(state), self.env, e)
            self.q_values.append(q_value)

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

            # update the q_net (experience replay)
            if (
                self.cfg.TRAIN.TRAIN_BY_EPISODE == False
                and self.frame_num > self.cfg.TRAIN.START_TRAIN
                and self.frame_num % self.cfg.TRAIN.TRAIN_INTERVAL == 0
                and len(self.buffer) > self.cfg.TRAIN.BATCH_SIZE
            ):
                self.update_experience_replay()

        return game_rewards

    def update_experience_replay(self):
        """Update (train) the network using experience replay technique."""
        self.net.train()
        for _ in range(self.cfg.TRAIN.NUM_ITERS_PER_TRAIN):
            # load data
            data = self.buffer.load(self.cfg.TRAIN.BATCH_SIZE)
            states = data["states"]
            next_states = data["next_states"]
            rewards = self.set_device(data["rewards"])
            actions = self.set_device(data["actions"])
            dones = self.set_device(data["dones"])

            # estimate target values
            values_next = self.estimate_target_values(next_states)
            targets = rewards + self.cfg.TRAIN.DISCOUNT_RATE * values_next * (1 - dones)

            # estimate action values q(s,a;\theta)
            preds = self.net(states).gather(1, actions).squeeze(-1)
            loss = self.criterion(preds, targets)

            # update parameters
            self.gradient_descent(self.net.parameters(), loss)

            # update loss history
            self.losses.append(float(loss.detach().data))
            self.train_iters += 1

            if self.train_iters % self.cfg.TRAIN.TARGET_SYNC_INTERVAL == 0:
                self.sync_target_net()

    def _train(self):
        """Train the q network"""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
            # set start time
            episode_start_time = time.time()

            # run single episode
            self.episode_num = episode_num
            rewards = self.run_single_episode()
            episode_time = time.time() - episode_start_time

            # update rewards history
            self.rewards_history.append(rewards)

            # update the q_net (experience replay)
            if (
                self.cfg.TRAIN.TRAIN_BY_EPISODE
                and episode_num > self.cfg.TRAIN.START_TRAIN
                and episode_num % self.cfg.TRAIN.TRAIN_INTERVAL == 0
                and len(self.buffer) > self.cfg.TRAIN.BATCH_SIZE
            ):
                self.update_experience_replay()

            # log info
            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info(
                    {
                        "Avg Q value": np.mean(self.q_values),
                        "episode time": episode_time,
                    }
                )

            # save model
            if self.cfg.LOGGER.SAVE_MODEL:
                if episode_num > 1:
                    if episode_num % self.cfg.LOGGER.SAVE_MODEL_INTERVAL == 0:
                        self._save_model(self.net)
                    
                    # save the best model
                    if self.best_avg_reward < np.mean(self.rewards_history):
                        self.best_avg_reward = np.mean(self.rewards_history)
                        self._save_model(
                            self.net, suffix=f"best_{int(self.best_avg_reward)}"
                        )

            # early stop
            if self.early_stopping_condition():
                break

    def estimate_target_values(self, next_states):
        """Estimate the target values based on DQN.
        Target network is used to estimate next state's value.

        targets <- R + gamma * max_a_Q_target(S',a;theta-)
        """
        values_target = self.target_net(next_states).detach().max(1)[0]

        return values_target
