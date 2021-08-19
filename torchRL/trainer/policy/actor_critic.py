import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from ...data import build_dataset, build_pipeline
from ...net import build_ActorCritic
from ..base import BaseTrainer
from ..builder import TRAINERS


@TRAINERS.register_module()
class ActorCriticTrainer(BaseTrainer):
    """Actor-Critic Policy Gradient Trainer."""

    def __init__(self, env, cfg):
        super(ActorCriticTrainer, self).__init__(env, cfg)
        self.actor_critic = build_ActorCritic(cfg.NET)
        if self.use_gpu:
            self.actor_critic.to(torch.device("cuda"))
        self.buffer = build_dataset(cfg)
        self.pipeline = build_pipeline(cfg.DATASET)

        self.criterion = nn.SmoothL1Loss()
        self.lr = self.cfg.TRAIN.LEARNING_RATE
        self.optimizer = Adam(self.actor_critic.parameters(), self.lr)
        self.e_greedy_min, self.e_greedy_max = cfg.TRAIN.EPSILON_GREEDY_MINMAX
        self.epsilon = self.e_greedy_max
        self.rewards_history = []
        self.losses = [np.inf]
        self.losses_actor = [np.inf]
        self.losses_critic = [np.inf]
        self.global_iters = 0

    def run_single_episode(self, episode_num):
        """Run a single episode for episodic environment.
        Every observation is stored in the memory buffer.
        """
        rewards = 0.0
        done = False
        state = self.env.reset()

        while not done:
            self.buffer.clear()
            for t in range(self.cfg.TRAIN.BATCH_SIZE):
                self.actor_critic.eval()
                action = self.actor_critic.predict(self.pipeline(state))
                next_state, reward, done, _ = self.env.step(action)
                
                # scale the reward
                reward *= self.cfg.ENV.REWARD_SCALE

                # save data to the buffer
                self.buffer.stack((state, next_state, reward, action, done))

                state = next_state
                rewards += reward / self.cfg.ENV.REWARD_SCALE
                if done:
                    break

            self.update_batch()

        return rewards

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        # estimate TD errors
        with torch.no_grad():
            target = (
                reward
                + self.cfg.TRAIN.DISCOUNT_RATE
                * self.actor_critic.forward_critic(self.pipeline(next_state)).squeeze(-1)
                * ~done
            )
            td_error = target - self.actor_critic.forward_critic(self.pipeline(state)).squeeze(-1)

        # calculate loss for the actor
        self.actor_critic.train()
        pred = self.actor_critic.forward_actor(self.pipeline(state))[0][action].unsqueeze(0)
        loss_actor = -(torch.log(pred) * td_error.detach())

        # calculate loss for the critic
        preds_value = self.actor_critic.forward_critic(self.pipeline(state))
        loss_critic = self.criterion(preds_value, target.squeeze(-1).detach())

        # update parameters for the critic
        self.optimizer.zero_grad()
        loss = loss_actor + loss_critic
        loss.backward()
        self.optimizer.step()

        self.losses.append(float(loss))
        self.losses_actor.append(float(loss_actor))
        self.losses_critic.append(float(loss_critic))
        self.global_iters += 1

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
                    * self.actor_critic.forward_critic(next_states).squeeze(-1)
                    * ~dones
                )
                td_errors = targets - self.actor_critic.forward_critic(states).squeeze(
                    -1
                )

            # calculate loss for the actor
            self.actor_critic.train()
            preds = self.actor_critic.forward_actor(states)
            preds = torch.gather(preds, 1, actions).squeeze(-1)
            loss_actor = -torch.log(preds) * td_errors.detach()
            loss_actor = loss_actor.mean()

            # calculate loss for the critic
            preds_values = self.actor_critic.forward_critic(states).squeeze(-1)
            loss_critic = self.criterion(preds_values, targets.detach())

            # update parameters for the critic
            self.optimizer.zero_grad()
            loss = loss_actor + loss_critic
            loss.backward()
            self.optimizer.step()

            self.losses.append(float(loss))
            self.losses_actor.append(float(loss_actor))
            self.losses_critic.append(float(loss_critic))
            self.global_iters += 1

    def _train(self):
        """Train the Policy."""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
            rewards = self.run_single_episode(episode_num)
            self.rewards_history.append(rewards)

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info(episode_num)

            if (episode_num > 0) and (episode_num % self.cfg.LOGGER.SAVE_MODEL_INTERVAL == 0):
                self._save_model(self.actor_critic, suffix=str(episode_num))

            if self.early_stopping_condition():
                return True

    def log_info(self, episode_num):
        """log current information for the training."""
        self.logger.info(
            f"Episode: {episode_num + 1}, Step: {self.rewards_history[-1]}, "
            f"Last {self.cfg.TRAIN.AVERAGE_SIZE} "
            f"Avg rewards: {np.mean(self.rewards_history[-self.cfg.TRAIN.AVERAGE_SIZE:])}, "
            f"Loss Total: {self.losses[-1]:.6f}, "
            f"Loss Actor: {self.losses_actor[-1]:.6f}, "
            f"Loss Critic: {self.losses_critic[-1]:.6f}, "
            f"Last {self.cfg.TRAIN.AVERAGE_SIZE} "
            f"Avg Loss: {np.mean(self.losses[-self.cfg.TRAIN.AVERAGE_SIZE:]):.6f}"
        )

    def save_model(self):
        self._save_model(self.actor_critic)
