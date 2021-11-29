from copy import deepcopy

import numpy as np
import torch

from ..builder import TRAINERS
from .base import BasePolicyGradientTrainer


@TRAINERS.register_module()
class REINFORCETrainer(BasePolicyGradientTrainer):
    """
    REINFORCE Actor-Critic Trainer
    """

    def __init__(self, env, cfg):
        super(REINFORCETrainer, self).__init__(env, cfg)

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

            # stack frames until dataset's size reaches the batch size
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

                # stack data
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

            # update
            self.update()
            self.buffer.clear()

        return game_rewards

    def update(self):
        """Update (train) the network."""
        # sum of rewards
        G_t = 0.0
        self.net.train()

        for sample in self.buffer.get_buffer(reverse=True):
            state, next_state, reward, action, done = sample

            # total reward is the target value for the update
            G_t = reward + self.cfg.TRAIN.DISCOUNT_RATE * G_t
            target = torch.tensor([G_t], dtype=torch.float32)
            target = self.set_device(target)
            value = self.net.estimate_values(self.pipeline(state), action).squeeze(-1)
            td_error = target - value

            # estimate actor loss
            pred = self.net.forward_actor(self.pipeline(state))[0][action].squeeze(0)
            loss_actor = -(torch.log(pred) * td_error.detach())

            # calculate critic loss
            loss_critic = self.criterion(value, target)

            # update parameters for the critic
            loss = loss_actor + loss_critic
            self.gradient_descent(self.net.parameters(), loss)

            # update loss history
            self.losses.append(float(loss))
            self.losses_actor.append(float(loss_actor))
            self.losses_critic.append(float(loss_critic))
            self.train_iters += 1
