import torch

from ..builder import TRAINERS
from .base import QTrainer


@TRAINERS.register_module()
class SARSATrainer(QTrainer):
    """Trainer class for SARSA"""

    def __init__(self, env, cfg):
        super(SARSATrainer, self).__init__(env, cfg)

    def run_single_episode(self, episode_num):
        """Run a single episode for episodic environment."""
        rewards = 0.0
        done = False
        state = self.env.reset()
        e = self.e_greedy_linear_annealing(episode_num)

        while not done:
            self.q_net.eval()
            action = self.q_net.predict_e_greedy(self.pipeline(state), self.env, e)
            next_state, reward, done, _ = self.env.step(action)

            # scale the reward
            reward *= self.cfg.ENV.REWARD_SCALE

            # update the q_net
            self.update(state, next_state, reward, action, done)

            # save data to the buffer
            state = next_state
            rewards += reward / self.cfg.ENV.REWARD_SCALE

        return rewards

    def _train(self):
        """Train the q network"""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
            rewards = self.run_single_episode(episode_num)
            self.rewards_history.append(rewards)

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info(episode_num)

            if (episode_num > 0) and (episode_num % self.cfg.LOGGER.SAVE_MODEL_INTERVAL == 0):
                self._save_model(self.q_net, suffix=str(episode_num))

            if self.early_stopping_condition():
                break

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        # estimate target values
        value_next = self.estimate_target_values(self.pipeline(next_state))
        target = reward + self.cfg.TRAIN.DISCOUNT_RATE * value_next * ~done

        # estimate action values q(s,a;\theta)
        self.q_net.train()
        pred = self.q_net(state)[0][action]
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        self.losses.append(float(loss.detach().data))
        self.global_iters += 1

    def estimate_target_values(self, next_state):
        """Estimate the target value based on SARSA.

        A' <- Q(S', epsilon)
        target <- R + gamma * Q(S',A')
        """
        self.q_net.eval()
        next_action = self.q_net.predict_e_greedy(next_state, self.env, self.epsilon)
        with torch.no_grad():
            value_target = self.q_net(next_state)[0][next_action]
        return value_target
