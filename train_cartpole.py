import argparse

import gym
import matplotlib.pyplot as plt

from torchRL.configs.cartpole_defaults import get_cfg_defaults
from torchRL.trainer import build_trainer


def train(cfg, env, args):
    # train
    trainer = build_trainer(env, cfg)
    trainer.train()
    if args.show_plot:
        fig, axs = plt.subplots(2)
        axs[0].plot(trainer.losses, "b-")
        axs[0].set_title("Loss")
        axs[0].set(xlabel="Iteration", ylabel="Loss")
        axs[1].plot(trainer.steps_history, "r-")
        axs[1].set_title("Steps")
        axs[1].set(xlabel="Episode", ylabel="Steps")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="yaml config file.")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="show loss plot when the train is done.",
    )
    args = parser.parse_args()

    # get configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    env = gym.make(cfg.ENV.NAME)
    env.max_episode_steps = cfg.ENV.MAX_EPISODE_STEPS

    cfg.NET.STATE_DIM = env.observation_space.shape[0]
    cfg.NET.ACTION_DIM = env.action_space.n
    cfg.freeze()

    train(cfg, env, args)
