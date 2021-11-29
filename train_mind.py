import argparse
import os.path as osp

import gym

from torchRL.configs.mind_defaults import get_cfg_defaults
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

    # make environmemnt
    gym.envs.register(
        id="Mind-v0",
        entry_point="torchRL.env:MIND",
        max_episode_steps=cfg.ENV.MAX_EPISODE_STEPS,
        kwargs={
            "train_data": cfg.ENV.TRAIN_DATA,
            "news_labels": cfg.ENV.NEWS_LABELS,
            "num_actions": cfg.ENV.NUM_CATEGORIES,
            "seed": cfg.ENV.SEED,
        },
    )
    env = EnvWrapper(cfg.ENV)

    assert cfg.NET.STATE_DIM == env.observation_space.n
    assert cfg.NET.ACTION_DIM == env.action_space.n
    cfg.freeze()

    train(cfg, env, args)
