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
            self.buffer.stack((state, next_state, reward, action, done))

            # save data to the buffer
            state = next_state
            rewards += reward / self.cfg.ENV.REWARD_SCALE

        # update the q_net using Monte-Carlo
        self.update()
        self.buffer.clear()

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
            target = self.set_device(target)

            # estimate action values q(s,a;\theta)
            pred = self.q_net(self.pipeline(state))[0][action]
            self.optimizer.zero_grad()
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.losses.append(float(loss.detach().data))
            self.global_iters += 1
