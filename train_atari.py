import argparse
import os.path as osp

from torchRL.configs.breakout_defaults import get_cfg_defaults
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
    env = EnvWrapper(cfg.ENV)

    cfg.NET.STATE_DIM = env.observation_space.shape[0]
    cfg.NET.ACTION_DIM = env.action_space.n
    cfg.freeze()

    train(cfg, env, args)
