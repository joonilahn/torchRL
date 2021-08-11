import torch

from ...dataset import BufferData
from ..builder import TRAINERS
from .base import QTrainer


@TRAINERS.register_module()
class MCTrainer(QTrainer):
    """Trainer class for Monte-Carlo Control"""

    def __init__(self, env, cfg):
        super(MCTrainer, self).__init__(env, cfg)
        self.buffer = BufferData()

    def run_single_episode(self, episode_num):
        """Run a single episode for episodic environment."""
        steps = 0
        done = False
        state = self.env.reset()
        e = self.e_greedy_linear_annealing(episode_num)

        while not done:
            self.q_net.eval()
            action = self.q_net.predict_e_greedy(state, self.env, e)
            next_state, reward, done, _ = self.env.step(action)

            if done:
                reward = self.cfg.ENV.REWARD_DONE
            else:
                reward = self.cfg.ENV.REWARD

            # update the q_net
            self.buffer.stack((state, next_state, reward, action, done))

            # save data to the buffer
            state = next_state
            steps += 1

        # update the q_net using Monte-Carlo
        self.update()
        self.buffer.clear()

        return steps

    def train(self):
        """Train the q network"""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
            steps = self.run_single_episode(episode_num)
            self.steps_history.append(steps)

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info(episode_num)

            if self.early_stopping_condition():
                return True

    def update(self):
        """Update (train) the network."""
        # sum of rewards
        G_t = 0.0
        self.q_net.train()

        for sample in self.buffer.get_buffer(reversed=True):
            state, next_state, reward, action, done = sample

            # total reward is the target value for the update
            G_t = reward + self.cfg.TRAIN.DISCOUNT_RATE * G_t
            target = torch.tensor(G_t, dtype=torch.float32)

            # estimate action values q(s,a;\theta)
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
