import torch

from ...data import build_dataset
from ...net import build_Qnet
from ..builder import TRAINERS
from .qlearning import QLearningTrainer


@TRAINERS.register_module()
class DQNTrainer(QLearningTrainer):
    """Trainer class for Deep Q-Networks"""

    def __init__(self, env, cfg):
        super(QLearningTrainer, self).__init__(env, cfg)
        self.q_target = build_Qnet(cfg.NET)
        if self.use_gpu:
            self.q_target.to(torch.device("cuda"))
        self.q_target.copy_parameters(self.q_net)
        self.q_target.eval()
        self.buffer = build_dataset(cfg)

    def run_single_episode(self, episode_num):
        """Run a single episode for episodic environment.
        Every observation is stored in the memory buffer.
        """
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

            # save data to the buffer
            self.buffer.stack((state, next_state, reward, action, done))
            state = next_state
            rewards += reward / self.cfg.ENV.REWARD_SCALE

        return rewards

    def update_experience_replay(self):
        """Update (train) the network using experience replay technique."""
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
            targets = rewards + self.cfg.TRAIN.DISCOUNT_RATE * values_next * ~dones

            # estimate action values q(s,a;\theta)
            self.q_net.train()
            preds = self.q_net(states)
            preds = torch.gather(preds, 1, actions).squeeze(-1)
            self.optimizer.zero_grad()
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()
            self.losses.append(float(loss.detach().data))
            self.global_iters += 1

    def _train(self):
        """Train the q network"""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
            rewards = self.run_single_episode(episode_num)
            self.rewards_history.append(rewards)

            # update the q_net (experience replay)
            if (
                episode_num % self.cfg.TRAIN.TRAIN_INTERVAL == 0
                and len(self.buffer) > self.cfg.TRAIN.BATCH_SIZE
            ):
                self.update_experience_replay()

                if self.global_iters % self.cfg.TRAIN.TARGET_SYNC_INTERVAL == 0:
                    self.q_target.copy_parameters(self.q_net)

            if episode_num % self.cfg.TRAIN.VERBOSE_INTERVAL == 0:
                self.log_info(episode_num)

            if (episode_num > 0) and (episode_num % self.cfg.LOGGER.SAVE_MODEL_INTERVAL == 0):
                self._save_model(self.q_net, suffix=str(episode_num))

            if self.early_stopping_condition():
                break

    def estimate_target_values(self, next_states):
        """Estimate the target values based on DQN.
        Target network is used to estimate next state's value.

        targets <- R + gamma * max_a_Q_target(S',a;theta-)
        """
        with torch.no_grad():
            values_target = self.q_target(next_states).detach().max(1)[0]

        return values_target
