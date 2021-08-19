import random

import torch
from ..builder import DATASETS, build_pipeline

@DATASETS.register_module()
class BufferDataset:
    def __init__(self, cfg):
        self.buffer_size = cfg.get("BUFFER_SIZE", 50000)
        self.buffer = []

    def stack(self, data):
        if len(self) > self.buffer_size:
            self.buffer.pop(0)
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
        dones = torch.tensor(dones, dtype=torch.bool)

        return {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
        }

    def get_buffer(self, reversed=False):
        if reversed:
            return self.buffer[::-1]
        else:
            return self.buffer

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

@DATASETS.register_module()
class BufferImageDataset(BufferDataset):
    def __init__(self, cfg):
        super(BufferImageDataset, self).__init__(cfg)
        self.pipeline = build_pipeline(cfg)

    def load(self, size):
        samples = random.sample(self.buffer, size)
        states = []
        next_states = []
        rewards = []
        actions = []
        dones = []

        for sample in samples:
            states.append(self.pipeline(sample[0]))
            next_states.append(self.pipeline(sample[1]))
            rewards.append(sample[2])
            actions.append(sample[3])
            dones.append(sample[4])

        states = torch.stack(states, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool)

        return {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
        }