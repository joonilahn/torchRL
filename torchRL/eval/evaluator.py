from copy import deepcopy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import build_pipeline
from ..data.pipelines import to_grayscale_and_resize
from ..env import DiscreteList, EnvWrapper
from ..net import build_Qnet

class Evaluator:
    def __init__(self, cfg, agent=None, weight=None, pipeline=None):
        self.cfg = cfg
        self.pipeline = pipeline if pipeline else build_pipeline(cfg.DATASET)
        self.agent = agent if agent else build_Qnet(cfg.NET)
        if weight:
            self.agent.load_state_dict(torch.load(weight))
            print("Loaded pretrained weight.")
        if cfg.USE_GPU:
            self.agent.cuda()
        self.env = EnvWrapper(cfg.ENV)

        self.game_has_life = True
        self.curr_lives = 0
        self.reward_all = []
        self.best_frames = []
        self.best_reward = -np.inf
        self.num_output = self.env.action_space.num_out if isinstance(self.env.action_space, DiscreteList) else 1

    def evaluate(self, num_eval=30, e=0.05, verbose=False, verbose_interval=100):
        self.reward_all = []

        for i in range(num_eval):
            reward_total = 0
            reward_per_life = 0
            done = False
            done_life = True
            state_init = self.env.reset()
            state = state_init
            frames = []
            steps = 0

            while not done:
                steps += 1

                # get action (e-greedy)
                try:
                    action = self.agent.predict_e_greedy(self.pipeline(state), self.env, e, num_output=self.num_output)[0]
                except:
                    action = self.agent.predict(self.pipeline(state), num_output=self.num_output)
                    
                # take the action (step)
                next_state, reward, done, lives = self.env.env.step(action)
                frames.append(next_state)

                # save frame
                if self.cfg.ENV.TYPE == "Atari":
                    next_state = to_grayscale_and_resize(next_state)

                # For Atari, stack the next state to the current states
                if self.cfg.ENV.TYPE == "Atari":
                    state[:, :, 4] = next_state
                
                reward_total += reward
                reward_per_life += reward

                # check whether game's life has changed
                if (steps == 1) and (self.cfg.ENV.TYPE == "Atari"):
                    self.set_init_lives(lives)
                    done_life = self.is_done_for_life(lives, reward)
                
                # set the current state to the next state (state <- next_state)
                if self.cfg.ENV.TYPE == "Atari":
                    state = np.concatenate(
                        [state[:, :, 1:], np.expand_dims(next_state, axis=2)], axis=2
                    )
                else:
                    state = next_state

            if (verbose) and (steps % verbose_interval == 0):
                print(f"Steps: {steps}")

            if steps == self.env.env.__max_episode_steps:
                break

            if reward_total > self.best_reward:
                self.best_frames = frames
                self.best_reward = reward_total
            self.reward_all.append(reward_total)
            print(f"Eval No; {i+1}, Total Reward: {reward_total}")

        print(f"Average reward after {num_eval} evaluation: {np.mean(self.reward_all):.2f}")

    def make_animation(self, one=None, frames=None, fps=60):
        """make animation"""
        if frames is None:
            frames = self.best_frames
        plt.ioff()
        fig = plt.figure(figsize=(3, 4))
        fig.set_tight_layout(True)
        patch = plt.imshow(frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames), interval=int(1000 / fps)
        )
        plt.ion()
           
        return anim
        
    def save_animation(self, anim=None, out=None, frames=None, fps=60):
        """Saves a list of frames as a gif"""
        if out is None:
            env_name = (
                self.cfg.ENV.NAME.split('-')[0]
                .replace("Deterministic", "")
                .replace("NoFrameskip", "")
            )
            trainer = self.cfg.TRAIN.TRAINER.lower().replace("trainer", "")
            out = f"{env_name}_{trainer}_{int(self.best_reward)}"

        if anim is None:
            anim = self.make_animation(frames=frames, fps=fps)
        
        anim.save(f"{out}.gif", writer="pillow", fps=fps)
        print(f"Saved {out}!")
    
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

    def load_pretrained(self, weight):
        self.agent.load_state_dict(torch.load(weight))
        print("Load pretrained weight.")