import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import MultiLabelBinarizer

class MIND(gym.Env):
    """
    Gym environment for simplifies MIND(MIcrosoft News Dataset) dataset for RL train.

    action_space = 18 (num of news categories)
    """

    def __init__(self, train_data, news_labels, num_actions=18, seed=10):
        self.seed(seed)

        # action space: number of news categories
        self.action_space = spaces.Discrete(num_actions)

        # observation space: number of news categories (ont-hot encoding for click history)
        self.observation_space = spaces.Discrete(num_actions)

        # read data
        self.data = []
        with open(train_data, 'r') as f:
            for line in f:
                (
                    user_id,
                    history_category,
                    history_subcategory,
                    click_category,
                    click_subcategory,
                ) = line.split(',')

            if history_category is not None:
                self.data.append(
                    dict(
                        user_id=user_id,
                        history_category=self.parse_state(history_category),
                        # history_subcategory=history_subcategory,
                        click_category=int(click_category),
                        # click_subcategory=click_subcategory
                    )
                )

        # map from news_id to news_category_name
        self.news_id_to_name = {}
        with open(news_labels, 'r') as f:
            for line in f:
                news_category, idx = line.split(',')
                self.news_id_to_name[int(idx)] = news_category

        # one-hot encoder
        self.mld = MultiLabelBinarizer(classes=list(range(len(self.news_id_to_name))))

        self.cur_data = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)

    def reset(self):
        # shuffle the data
        random.shuffle(self.data)
        self.cur_data = random.sample(self.data, 1)[0]
        return self.cur_data["history_category"]

    def step(self, action):
        # if recommended news is clicked, get reward
        reward = self.compare_actions(action, self.cur_data["click_category"])
        reward = np.clip(reward, 0.0, 1.0)

        # next_state, next_customer_id
        self.cur_data = random.sample(self.data, 1)[0]

        return (self.cur_data["history_category"], reward, False, {})

    def render(self):
        raise NotImplementedError("Rendering is not provided.")

    def parse_state(self, x, mode="softmax"):
        state = np.zeros(self.action_space.n)
        for i in x.split(' '):
            state[int(i)] += 1

        if mode == "softmax":
            return self.softmax(state)
        elif mode == "hardmax":
            return self.hardmax(state)
        else:
            raise ValueError("Mode must be either softmax or hardmax.")

    def softmax(self, x):
        if x.max() == 0.0:
            return x
        return np.exp(x) / np.sum(np.exp(x))

    def hardmax(self, x):
        if x.max() == 0.0:
            return x
        return x / x.max()

    def one_hot_encoder(self, idx):
        return self.mlb.fit_transform([[idx]])[0]

    def one_hot_to_idx(self, one_hot):
        return self.mlb.inverse_transform(np.expand_dims(one_hot, 0))[0][0]

    def num_intersections(self, x, y):
        return float(len(set(x).intersection((set(y)))))

    def compare_actions(self, x, y):
        if isinstance(x, int):
            if isinstance(y, int):
                return float(x == y)
            elif isinstance(y, list):
                return float(x in y)

        elif isinstance(x, list):
            if isinstance(y, int):
                return float(y in x)
            elif isinstance(y, list):
                return self.num_intersections(x, y)

        else:
            raise ValueError