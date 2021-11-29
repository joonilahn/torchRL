import argparse
import os.path as osp

import gym

from torchRL.configs.customers_defaults import get_cfg_defaults
from torchRL.env import EnvWrapper
from torchRL.trainer import build_trainer


def train(cfg, env, args):
    # train
    trainer = build_trainer(env, cfg)

    # copy the config file
    with open(f"{cfg.LOGGER.OUTPUT_DIR}/{osp.basename(args.config)}", "w") as f:
        f.write(cfg.dump())

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="yaml config file.")
    args = parser.parse_args()

    # get configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    # make environment
    gym.envs.register(
        id="CustomersVisit-v0",
        entry_point="torchRL.env:CustomersVisit",
        max_episode_steps=cfg.ENV.MAX_EPISODE_STEPS,
        kwargs={
            "num_actions": cfg.ENV.NUM_CONTENTS,
            "num_customers": cfg.ENV.NUM_CUSTOMERS,
            "num_preference": cfg.ENV.NUM_PREFERENCE,
            "num_recommendations": cfg.ENV.NUM_OUTPUT,
            "prob": cfg.ENV.PROBABILITY,
            "seed": cfg.ENV.SEED,
        },
    )
    env = EnvWrapper(cfg.ENV)

    assert cfg.NET.STATE_DIM == env.observation_space.n
    assert cfg.NET.ACTION_DIM == env.action_space.n
    cfg.freeze()

    train(cfg, env, args)
