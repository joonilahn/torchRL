import time
from copy import deepcopy

import numpy as np
import torch

from ...data import build_dataset
from ..builder import TRAINERS
from .base import QTrainer


@TRAINERS.register_module()
class MCTrainer(QTrainer):
    """Trainer class for Monte-Carlo Control"""

    def __init__(self, env, cfg):
        super(MCTrainer, self).__init__(env, cfg)
        self.buffer = build_dataset(cfg)

    def run_single_episode(self):
        """Run a single episode for episodic environment."""
        game_rewards = 0.0
        done = False
        done_life = False
        state = self.env.reset()
        e = self.update_e_greedy_param()
        self.steps = 0

        while not done:
            self.steps += 1
            self.frame_num += 1

            action, q_value = self.net.predict_e_greedy(self.pipeline(state), self.env, e, num_output=self.num_output)
            self.q_values.append(q_value)

            # take the action
            next_state, reward, done, info = self.env.step(action)
            if isinstance(action, list):
                action = action[0]

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

        # update the q_net using Monte-Carlo
        self.update()
        self.buffer.clear()

        return game_rewards

    def _train(self):
        """Train the q network"""
        for episode_num in range(1, self.cfg.TRAIN.NUM_EPISODES + 1):
            # set start time
            episode_start_time = time.time()

            # run single episode
            self.episode_num = episode_num
            rewards = self.run_single_episode()
            episode_time = time.time() - episode_start_time
            self.rewards_history.append(rewards)

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info({"episode time": episode_time})

            # save model
            if self.cfg.LOGGER.SAVE_MODEL:
                if episode_num > 1:
                    if episode_num % self.cfg.LOGGER.SAVE_MODEL_INTERVAL == 1:
                        self._save_model(self.net)

                    # save the best model
                    if self.best_avg_reward < np.mean(self.rewards_history):
                        self.best_avg_reward = np.mean(self.rewards_history)
                        self._save_model(
                            self.net, suffix=f"best_{int(self.best_avg_reward)}"
                        )

            # evaluate
            if self.cfg.TRAIN.get("EVALUATE_INTERVAL", None) and self.env_eval and episode_num % self.cfg.TRAIN.EVALUATE_INTERVAL == 0:
                self.evaluate()

            # early stop
            if self.early_stopping_condition():
                break

    def update(self):
        """Update (train) the network."""
        # sum of rewards
        G_t = 0.0
        self.net.train()

        for sample in self.buffer.get_buffer(reverse=True):
            state, next_state, reward, action, done = sample

            # total reward is the target value for the update
            G_t = reward + self.cfg.TRAIN.DISCOUNT_RATE * G_t
            target = torch.tensor(G_t, dtype=torch.float32)
            target = self.set_device(target)

            # estimate action values q(s,a;\theta)
            pred = self.net(self.pipeline(state))[0][action]            
            
            # update parameters
            loss = self.criterion(pred, target)
            self.gradient_descent(self.net.parameters(), loss)

            # update loss history
            self.losses.append(float(loss.detach().data))
            self.train_iters += 1
