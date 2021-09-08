import random
from collections import deque

import torch

from ..builder import DATASETS, build_pipeline


@DATASETS.register_module()
class BufferDataset:
    def __init__(self, cfg):
        self.buffer_size = cfg.get("BUFFER_SIZE", 50000)
        self.buffer = deque(maxlen=self.buffer_size)

    def stack(self, data):
        self.buffer.append(data)

    def load(self, size):
        samples = random.sample(self.buffer, size)
        states = []
        next_states = []
        rewards = []
        actions = []
        dones = []

        for sample in samples:
            states.append(sample[0])
            next_states.append(sample[1])
            rewards.append(sample[2])
            actions.append(sample[3])
            dones.append(sample[4])

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.int8)

        return {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
        }

    def get_buffer(self, reverse=False):
        if reverse:
            return list(reversed(self.buffer))
        else:
            return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

@DATASETS.register_module()
class BufferFramesDataset(BufferDataset):
    def __init__(self, cfg):
        super(BufferFramesDataset, self).__init__(cfg)
        self.pipeline = build_pipeline(cfg)

    def stack(self, data):
        self.buffer.append((data[0], data[2], data[3], data[4]))

    def load(self, size):
        samples = random.sample(self.buffer, size)
        states = []
        next_states = []
        rewards = []
        actions = []
        dones = []

        for sample in samples:
            states.append(self.pipeline(sample[0][:, :,  :4]))
            next_states.append(self.pipeline(sample[0][:, :, 1:]))
            rewards.append(sample[1])
            actions.append(sample[2])
            dones.append(sample[3])

        states = torch.stack(states, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.int8)

        return {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
        }
