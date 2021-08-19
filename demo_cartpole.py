import argparse

import gym
import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from PIL import Image

from torchRL.configs.cartpole_defaults import get_cfg_defaults
from torchRL.net import build_Qnet


def display_frames_as_gif(frames, save=False, save_name="cartpole_result.gif"):
    """Displays a list of frames as a gif, with controls"""
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    
    if save:
        print(f"Saving the animation to {save_name}")
        anim.save(save_name, fps=30)

def play_cartpole(q_net, env):
    q_net.eval()
    state = env.reset()
    reward_sum = 0
    done = False
    frames = []
    e = 0.01
    while not done:
        frames.append(env.render("rgb_array"))
        action = q_net.predict_e_greedy(state, env, e)
        next_state, reward, done, _ = env.step(action)

        reward_sum += reward
        state = next_state
    print(f"Total Rewards: {reward_sum}")
    return frames


def load_model(config_file, weight):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    q_net = build_Qnet(cfg.NET)
    q_net.load_state_dict(torch.load(weight))
    return q_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="yaml config file.")
    parser.add_argument("weight", type=str, help="pretrained .pth file.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the result as a gif file.",
    )
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    env.max_episode_steps = 100

    # load pretrained model
    q_net = load_model(args.config, args.weight)

    # play cartpole
    frames = play_cartpole(q_net, env)

    # display animation
    save_name = args.weight.split('.')[0] + '_result.gif'
    display_frames_as_gif(frames, args.save, save_name=save_name)