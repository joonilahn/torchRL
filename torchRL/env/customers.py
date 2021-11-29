import random

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import MultiLabelBinarizer

from .discrete_list import DiscreteList


class CustomersVisit(gym.Env):
    """
    가상 고객 데이터 환경
    고객은 고정된 ID(0 ~ n-1)로 표현된다.

    num_customers: 고객이 몇 명인지 (n)
    num_actions: 추천할 수 있는 컨텐츠의 개수 (k)
    observtion: one-hot으로 표현된 고객의 ID (num_customers)
    """

    def __init__(self, num_actions=30, num_customers=1000, num_preference=2, num_recommendations=4, prob=1.0, seed=10):
        self.seed(seed)

        # action_space: 추천 컨텐츠의 개수
        if num_recommendations > 1:
            self.action_space = DiscreteList(num_actions, num_recommendations)
        else:
            self.action_space = spaces.Discrete(num_actions)

        # observation space: 고객의 숫자
        self.observation_space = spaces.Discrete(num_customers)

        # num_customers: 고객의 숫자
        self.num_customers = num_customers

        # one-hot encoder
        self.mlb = MultiLabelBinarizer(classes=list(range(num_customers)))

        # prob: 선호하는 컨텐츠를 클릭할 확률
        self.prob = prob

        # customer
        self.customers = MultiCustomers(num_customers, num_actions, num_preference, prob=prob)

        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)

    def one_hot_encoder(self, idx):
        return self.mlb.fit_transform([[idx]])[0]

    def one_hot_to_idx(self, one_hot):
        return self.mlb.inverse_transform(np.expand_dims(one_hot, 0))[0][0]

    def reset(self):
        self.state = np.zeros(self.observation_space.n)

        # random customer id for init state
        self.state = self.mlb.fit_transform([[random.randint(0, self.num_customers - 1)]])[0]

        return torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)

    def num_intersections(self, x, y):
        return float(len(set(x).intersection(set(y))))

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

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"(action, type(action))

        # 현재 방문한 고객의 id
        curr_customer_id = self.one_hot_to_idx(self.state)

        # 확률모델로부터 고객이 어떤 컨텐츠를 클릭할지 추출
        action_true = self.customers.visit(curr_customer_id)

        # 추천 컨텐츠와 위에서 추출한 컨텐츠가 일치하면 (예측이 맞았다면) reward=1
        reward = self.compare_actions(action, action_true)
        reward = np.clip(reward, 0.0, 1.0)

        # next_state, next_customer_id 준비
        next_customer_id = random.randint(0, self.num_customers - 1)
        self.state = self.one_hot_encoder(next_customer_id)

        return (
            torch.tensor(self.state, dtype=torch.float32).unsqueeze(0),
            reward,
            False,
            {
                "customer_id": curr_customer_id,
                "action_true": action_true,
                "preference": self.customers[curr_customer_id].preference
            },
        )

    def render(self):
        raise NotImplementedError("Rendering is not provided.")


class Customer:
    def __init__(self, num_actions, preference=(0,), prob=1.0):
        assert len(preference) < num_actions
        assert isinstance(preference, (list, tuple))
        self.num_actions = num_actions
        actions = list(range(num_actions))
        self.preference = preference
        self.no_preference = sorted(list(set(actions) - set(preference)))
        self.prob = prob

    def visit(self):
        # 고객이 방문해서 자신이 선호하는 컨텐츠를 선택(반환)
        if random.random() < self.prob:
            return list(self.preference)
        # random contents 클릭
        else:
            return random.sample(self.no_preference, len(self.preference))

    def __repr__(self):
        return "Preference: " + ", ".join(list(map(lambda x: str(x), self.preference)))


class MultiCustomers:
    def __init__(self, num_customers, num_actions, num_preference, prob=1.0):
        self.num_customers = num_customers
        self.num_actions = num_actions
        self.customers = []
        for _ in range(num_customers):
            self.customers.append(
                Customer(
                    num_actions,
                    preference=sorted(
                        tuple(random.sample(list(range(num_actions)), num_preference))
                              ),
                    prob=prob
                )
            )

    def visit(self, customer_idx):
        """특정 고객이 방문했을 때 선택한 컨텐츠"""
        return self.customers[customer_idx].visit()

    def __getitem__(self, idx):
        return self.customers[idx]
