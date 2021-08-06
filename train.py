import argparse

import gym
import matplotlib.pyplot as plt
import torch.nn as nn

from torchRL.configs.cartpole_defaults import get_cfg_defaults
from torchRL.net import build_net
from torchRL.trainer import build_trainer


def train(cfg, env):
    # train
    trainer = build_trainer(env, cfg)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="yaml config file.")
    args = parser.parse_args()

    # get configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    gym.register(
        id=cfg.ENV.NAME,
        entry_point="gym.envs.classic_control:CartPoleEnv",
        max_episode_steps=cfg.ENV.MAX_EPISODE_STEPS,
    )
    env = gym.make(cfg.ENV.NAME)

    cfg.NET.STATE_DIM = env.observation_space.shape[0]
    cfg.NET.ACTION_DIM = env.action_space.n
    cfg.freeze()

    train(cfg, env)
