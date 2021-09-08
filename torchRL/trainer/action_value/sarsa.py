import time
from copy import deepcopy

import numpy as np
import torch

from ..builder import TRAINERS
from .base import QTrainer


@TRAINERS.register_module()
class SARSATrainer(QTrainer):
    """Trainer class for SARSA"""

    def __init__(self, env, cfg):
        super(SARSATrainer, self).__init__(env, cfg)

    def run_single_episode(self):
        """Run a single episode for episodic environment."""
        game_rewards = 0.0
        done = False
        done_life = False
        state = self.env.reset()
        e = self.update_e_greedy_param()
        self.steps = 0
        self.net.train()

        while not done:
            self.steps += 1
            self.frame_num += 1

            action, q_value = self.net.predict_e_greedy(self.pipeline(state), self.env, e)
            self.q_values.append(q_value)

            # take the action
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

            # update the q_net
            self.update(state, next_state, reward, action, done or done_life)

            # set the current state to the next state (state <- next_state)
            if self.cfg.ENV.TYPE == "Atari":
                state = np.concatenate(
                    [state[:, :, 1:], np.expand_dims(next_state, axis=2)], axis=2
                )
            else:
                state = next_state

        return game_rewards

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

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info({"episode time": episode_time})

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

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        # estimate target values
        value_next = self.estimate_target_values(self.pipeline(next_state))
        target = reward + self.cfg.TRAIN.DISCOUNT_RATE * value_next * ~done

        # estimate action values q(s,a;\theta)
        pred = self.net(state)[0][action]

        # update parameters
        loss = self.criterion(pred, target)
        self.gradient_descent(self.net.parameters(), loss)

        # update loss history
        self.losses.append(float(loss.detach().data))
        self.train_iters += 1

    def estimate_target_values(self, next_state):
        """Estimate the target value based on SARSA.

        A' <- Q(S', epsilon)
        target <- R + gamma * Q(S',A')
        """
        self.net.eval()
        next_action, _ = self.net.predict_e_greedy(next_state, self.env, self.epsilon)
        with torch.no_grad():
            value_target = self.net(next_state)[0][next_action]
        return value_target
