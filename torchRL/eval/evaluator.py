from copy import deepcopy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import build_pipeline
from ..data.pipelines import to_grayscale_and_resize
from ..env import EnvWrapper
from ..net import build_Qnet

class Evaluator:
    def __init__(self, cfg, weight=None, seed=42):
        self.cfg = cfg
        self.pipeline = build_pipeline(cfg.DATASET)
        self.agent = build_Qnet(cfg.NET)
        if weight:
            self.agent.load_state_dict(torch.load(weight))
        if cfg.USE_GPU:
            self.agent.cuda()
        self.env = EnvWrapper(cfg.ENV)
        self.env.seed(seed)

        self.game_has_life = True
        self.curr_lives = 0
        self.reward_all = []
        self.best_frames = []
        self.best_reward = -np.inf

    def evaluate(self, num_eval=30, e=0.05):
        for i in range(num_eval):
            reward_total = 0
            reward_per_life = 0
            done = False
            state_init = self.env.reset()
            state = state_init
            frames = []
            e = e
            steps = 0

            while not done:
                steps += 1

                # get action (e-greedy)
                action = self.agent.predict_e_greedy(self.pipeline(state), self.env, e)[0]

                # take the action (step)
                next_state, reward, done, lives = self.env.env.step(action)
                frames.append(next_state)
                next_state = to_grayscale_and_resize(next_state)

                # For Atari, stack the next state to the current states
                if self.cfg.ENV.TYPE == "Atari":
                    state[:, :, 4] = next_state
                
                reward_total += reward
                reward_per_life += reward

                # check whether game's life has changed
                if (steps == 1) and (self.cfg.ENV.TYPE == "Atari"):
                    self.set_init_lives(lives)
                
                # set the current state to the next state (state <- next_state)
                if self.cfg.ENV.TYPE == "Atari":
                    state = np.concatenate(
                        [state[:, :, 1:], np.expand_dims(next_state, axis=2)], axis=2
                    )
                else:
                    state = next_state
                
            if reward_total > self.best_reward:
                self.best_frames = frames
                self.best_reward = reward_total
            self.reward_all.append(reward_total)
            print(f"Eval No; {i+1}, Total Reward: {reward_total}")

        print(f"Average reward after {num_eval} evaluation: {np.mean(self.reward_all):.2f}")

    def save_frames_as_gif(self, out=None, frames=None, fps=60, writer="pillow"):
        """Saves a list of frames as a gif"""
        if frames is None:
            frames = self.best_frames
        if out is None:
            env_name = (
                self.cfg.ENV.NAME.split('-')[0]
                .replace("Deterministic", "")
                .replace("NoFrameskip", "")
            )
            trainer = self.cfg.TRAIN.TRAINER.lower().replace("trainer", "")
            out = f"{env_name}_{trainer}_{int(self.best_reward)}"
        
        fig = plt.figure(figsize=(3, 4))
        fig.set_tight_layout(True)
        patch = plt.imshow(frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames), interval=int(1000 / fps)
        )
        anim.save(f"{out}.gif", writer=writer, fps=fps)
    
    def set_init_lives(self, lives):
        self.curr_lives = lives.get("ale.lives", None)
        if self.curr_lives is None:
            self.game_has_life = False
        else:
            if self.curr_lives > 0:
                self.game_has_life = True
            else:
                self.game_has_life = False
    
    def is_done_for_life(self, lives, reward):
        # games that have lives
        if self.game_has_life:
            if lives["ale.lives"] < self.curr_lives:
                done = True
            else:
                done = False
            self.curr_lives = lives["ale.lives"]
    
        else:
            if reward < 0:
                done = True
            else:
                done = False
        
        return done