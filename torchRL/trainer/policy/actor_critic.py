import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..builder import TRAINERS
from .base import BasePolicyGradientTrainer


@TRAINERS.register_module()
class TDActorCriticTrainer(BasePolicyGradientTrainer):
    """
    TD Actor-Critic Policy Gradient Trainer
      - On Policy (no target network)
      - Use TD(0) error for update
    """

    def __init__(self, env, cfg):
        super(TDActorCriticTrainer, self).__init__(env, cfg)

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        if self.cfg.ENV.TYPE == "Atari":
            next_state = self.pipeline(state[:, :, 1:]).unsqueeze(0)
            state = self.pipeline(state[:, :, :4]).unsqueeze(0)
            reward = self.set_device(torch.tensor([reward], dtype=torch.float32))
            action = self.set_device(torch.tensor([action], dtype=torch.int64))
            done = self.set_device(torch.tensor([done], dtype=torch.int8))
        else:
            state = self.pipeline(state)
            next_state = self.pipeline(next_state)

        # estimate TD errors
        target = (
            reward
            + self.cfg.TRAIN.DISCOUNT_RATE
            * self.estimate_target_values(next_state)
            * (1 - done)
        )
        value = self.net.estimate_values(state, action).squeeze(-1)
        td_error = target - value

        # calculate loss for the actor
        pred = self.net.forward_actor(state)[0][action].unsqueeze(0)
        loss_actor = -(torch.log(pred) * td_error.detach())

        # calculate loss for the critic
        loss_critic = self.criterion(value, target)

        # update parameters for the critic
        loss = loss_actor + loss_critic
        self.gradient_descent(self.net.parameters(), loss)

        # update loss history
        self.losses.append(float(loss))
        self.losses_actor.append(float(loss_actor))
        self.losses_critic.append(float(loss_critic))
        self.train_iters += 1

    def estimate_target_values(self, next_states):
        with torch.no_grad():
            # state values
            if isinstance(self.net.critic_layer, nn.Sequential):
                if getattr(self.net.critic_layer[-1], "out_features", None) == 1:
                    return self.net.forward_critic(next_states).squeeze(-1)
                elif isinstance(self.net.critic_layer, nn.Module):
                    if getattr(self.net.critic_layer, "action_dim", None) == 1:
                        return self.net.forward_critic(next_states).squeeze(-1)

            # state-action values
            # choose actions of the next states based on the target net argmaxQ(s',a;\theta)
            probs = self.net.forward_actor(next_states)
            actions_next = Categorical(probs).sample().unsqueeze(-1)

            # estimate the next action values q(s',a';theta^-)
            return self.net.estimate_values(next_states, actions_next).squeeze(-1)

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
            targets = (
                    rewards
                    + self.cfg.TRAIN.DISCOUNT_RATE
                    * self.estimate_target_values(next_states)
                    * (1 - dones)
            )
            td_errors = self.net.estimate_values(states, actions).squeeze(-1).detach()

            # calculate loss for the actor
            preds = (
                self.net.forward_actor(states).gather(1, actions).squeeze(-1)
            )
            loss_actor = -torch.log(preds) * td_errors
            loss_actor = loss_actor.mean()

            # calculate loss for the critic
            preds_values = self.net.estimate_values(states, actions).squeeze(-1)
            loss_critic = self.criterion(preds_values, targets)

            # update parameters for the critic
            loss = loss_actor + loss_critic
            self.gradient_descent(self.net.parameters(), loss)

            self.losses.append(float(loss))
            self.losses_actor.append(float(loss_actor))
            self.losses_critic.append(float(loss_critic))
            self.train_iters += 1


@TRAINERS.register_module()
class AdvantageActorCriticTrainer(BasePolicyGradientTrainer):
    """Advantage Actor-Critic Policy Gradient Trainer."""

    def __init__(self, env, cfg):
        super(AdvantageActorCriticTrainer, self).__init__(env, cfg)

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        if self.cfg.ENV.TYPE == "Atari":
            next_state = self.pipeline(state[:, :, 1:]).unsqueeze(0)
            state = self.pipeline(state[:, :, :4]).unsqueeze(0)
            reward = self.set_device(torch.tensor([reward], dtype=torch.float32))
            action = self.set_device(torch.tensor([action], dtype=torch.int64))
            done = self.set_device(torch.tensor([done], dtype=torch.int8))
        else:
            state = self.pipeline(state)
            next_state = self.pipeline(next_state)

        # estimate TD errors
        target = (
                reward
                + self.cfg.TRAIN.DISCOUNT_RATE
                * self.estimate_target_values(next_state)
                * (1 - done)
        )
        value = self.net.estimate_values(state, action).squeeze(-1)
        td_error = target - value

        # calculate loss for the actor
        pred = self.net.forward_actor(state)[0][action].unsqueeze(0)
        loss_actor = -(torch.log(pred) * td_error.detach())

        # calculate loss for the critic
        loss_critic = self.criterion(value, target)

        # update parameters for the critic
        loss = loss_actor + loss_critic
        self.gradient_descent(self.net.parameters(), loss)

        # update loss history
        self.losses.append(float(loss))
        self.losses_actor.append(float(loss_actor))
        self.losses_critic.append(float(loss_critic))
        self.train_iters += 1

    def estimate_target_values(self, next_states):
        with torch.no_grad():
            # state values
            if isinstance(self.net.critic_layer, nn.Sequential):
                if getattr(self.net.critic_layer[-1], "out_features", None) == 1:
                    return self.net.forward_critic(next_states).squeeze(-1)
                elif isinstance(self.net.critic_layer, nn.Module):
                    if getattr(self.net.critic_layer, "action_dim", None) == 1:
                        return self.net.forward_critic(next_states).squeeze(-1)

            # state-action values
            # choose actions of the next states based on the target net argmaxQ(s',a;\theta)
            probs = self.net.forward_actor(next_states)
            actions_next = Categorical(probs).sample().unsqueeze(-1)

            # estimate the next action values q(s',a';theta^-)
            return self.net.estimate_values(next_states, actions_next)

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
            targets = (
                    rewards
                    + self.cfg.TRAIN.DISCOUNT_RATE
                    * self.estimate_target_values(next_states)
                    * (1 - dones)
            )
            td_errors = self.net.estimate_values(states, actions).squeeze(-1).detach()

            # calculate loss for the actor
            preds = (
                self.net.forward_actor(states).gather(1, actions).squeeze(-1)
            )
            loss_actor = -torch.log(preds) * td_errors
            loss_actor = loss_actor.mean()

            # calculate loss for the critic
            preds_values = self.net.estimate_values(states, actions).squeeze(-1)
            loss_critic = self.criterion(preds_values, targets)

            # update parameters for the critic
            loss = loss_actor + loss_critic
            self.gradient_descent(self.net.parameters(), loss)

            self.losses.append(float(loss))
            self.losses_actor.append(float(loss_actor))
            self.losses_critic.append(float(loss_critic))
            self.train_iters += 1