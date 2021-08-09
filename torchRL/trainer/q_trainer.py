import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam

from ..dataset import BufferData
from ..net import build_net
from .builder import TRAINERS

@TRAINERS.register_module()
class QTrainer:
    """Base class for Q (action value) Trainer."""
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.q_net = build_net(cfg.NET)
        self.criterion = nn.MSELoss()
        self.lr = self.cfg.TRAIN.LEARNING_RATE
        self.optimizer = Adam(self.q_net.parameters(), self.lr)
        self.e_greedy_min, self.e_greedy_max = cfg.TRAIN.EPISILON_GREEDY_MINMAX
        self.epsilon = self.e_greedy_max
        self.steps_history = []
        self.losses = [np.inf]
        self.global_iters = 0

    def run_single_episode(self):
        """Run a single episode for episodic environment."""
        pass

    def train(self):
        """Train the q network"""
        pass

    def estimate_target_values(self, next_states):
        """Estimate target values using TD(0), MC, or TD(lambda)."""
        pass

    def e_greedy_linear_annealing(self, episode_num):
        """Get epsilon value based on linear annealing."""
        self.epsilon = max(self.e_greedy_min, self.e_greedy_max - self.e_greedy_min * (episode_num / 200))
        return self.epsilon

    def early_stopping_condition(self):
        """Early stop if running mean score is larger the the terminate threshold."""
        if (
            np.mean(self.steps_history[-self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS :])
            > self.cfg.TRAIN.AVG_STEPS_TO_TERMINATE
        ):
            print(
                f"Last {self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS} avg steps exceeded "
                f"{self.cfg.TRAIN.AVG_STEPS_TO_TERMINATE} times. Quit training."
            )
            return True
        else:
            return False

    def log_info(self, episode_num):
        """log current information for the training."""
        print(
            f"Episode: {episode_num + 1}, Step: {self.steps_history[-1]}, "
            f"epsilon: {self.epsilon:.2f}, "
            f"Last {self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS} "
            f"Avg Steps: {np.mean(self.steps_history[-self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS:])}, "
            f"Loss: {self.losses[-1]:.6f}, "
            f"Last {self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS} "
            f"Avg Loss: {np.mean(self.losses[-self.cfg.TRAIN.NUM_EPISIODES_AVG_STEPS:]):.6f}"
        )


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


@TRAINERS.register_module()
class SARSATrainer(QTrainer):
    """Trainer class for SARSA"""
    def __init__(self, env, cfg):
        super(SARSATrainer, self).__init__(env, cfg)

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
            self.update(state, next_state, reward, action, done)

            # save data to the buffer
            state = next_state
            steps += 1
        
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

    def update(self, state, next_state, reward, action, done):
        """Update (train) the network."""
        # estimate target values
        value_next = self.estimate_target_values(next_state)
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


@TRAINERS.register_module()
class QLearningTrainer(SARSATrainer):
    """Trainer class for Vanilla Q-Learning."""
    def __init__(self, env, cfg):
        super(QLearningTrainer, self).__init__(env, cfg)

    def estimate_target_values(self, next_state):
        """Estimate the target value based on Q-Learning.

        target <- R + gamma * max_a_Q(S',a)
        """
        with torch.no_grad():
            value_target = self.q_net(next_state).max(1)[0]
        return value_target


@TRAINERS.register_module()
class DQNTrainer(QLearningTrainer):
    """Trainer class for Deep Q-Networks"""
    def __init__(self, env, cfg):
        super(QLearningTrainer, self).__init__(env, cfg)
        self.q_target = build_net(cfg.NET)
        self.q_target.copy_parameters(self.q_net)
        self.q_target.eval()
        self.buffer = BufferData()

    def run_single_episode(self, episode_num):
        """Run a single episode for episodic environment.
        Every observation is stored in the memory buffer.
        """
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

            # save data to the buffer
            self.buffer.stack((state, next_state, reward, action, done))
            state = next_state
            steps += 1
        
        return steps

    def update_experience_replay(self):
        """Update (train) the network using experience replay technique."""
        for _ in range(self.cfg.TRAIN.NUM_ITERS_PER_TRAIN):
            # load data
            data = self.buffer.load(self.cfg.TRAIN.BATCH_SIZE)
            states = data["states"]
            next_states = data["next_states"]
            rewards = data["rewards"]
            actions = data["actions"]
            dones = data["dones"]

            # estimate target values
            values_next = self.estimate_target_values(next_states)
            targets = (
                rewards + self.cfg.TRAIN.DISCOUNT_RATE * values_next * ~dones
            )

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

    def train(self):
        """Train the q network"""
        for episode_num in range(self.cfg.TRAIN.NUM_EPISODES):
            steps = self.run_single_episode(episode_num)
            self.steps_history.append(steps)

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

            if self.early_stopping_condition():
                return True

    def estimate_target_values(self, next_states):
        """Estimate the target values based on DQN.
        Target network is used to estimate next state's value.

        targets <- R + gamma * max_a_Q_target(S',a;theta-)
        """
        with torch.no_grad():
            values_target = self.q_target(next_states).detach().max(1)[0]

        return values_target

@TRAINERS.register_module()
class DDQNTrainer(DQNTrainer):
    """Trainer class for Double DQN."""
    def __init__(self, env, cfg):
        super(DDQNTrainer, self).__init__(env, cfg)

    def estimate_target_values(self, next_states):
        """Estimate the target values based on Double DQN.
        Target network is used to estimate values for the next states.

        A' <- Q_target(S', epsilon;theta)
        targets <- R + gamma * Q_action(S',A')
        """
        with torch.no_grad():
            # choose actions of the next states based on the target net argmaxQ(s',a;\theta)
            actions_next = self.q_target(next_states).argmax(1).unsqueeze(-1)
            # estimate the next action values q(s',a';\theta^-)
            values_target = (
                self.q_net(next_states).gather(1, actions_next).squeeze(-1)
            )
        
        return values_target