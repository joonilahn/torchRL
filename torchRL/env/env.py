import cv2
import gym
import numpy as np


class EnvWrapper:
    def __init__(self, cfg):
        self.type = cfg.TYPE
        if ("wrap" in self.type.lower()) and ("NoFrameskip" in cfg.NAME):
            from baselines.common.atari_wrappers import make_atari, wrap_deepmind
            env = make_atari(cfg.ENV.NAME)
            env = wrap_deepmind(env, frame_stack=True, scale=True)
        self.env = gym.make(cfg.NAME)
        self.prep_type = "multiframes" if "Deterministic" in cfg.NAME else "singleframe"
        self.__observation_space = self.env.observation_space
        self.__action_space = self.env.action_space
        self.curr_lives = 0

    def reset(self):
        # If training Atari games, make multiple frames (4 by default) as input
        if self.prep_type == "multiframes":
            return self.make_multiframes(self.env.reset())
        return self.env.reset()
    
    def step(self, action, preprocess=True):
        next_state, reward, done, info = self.env.step(action)
        if self.prep_type == "multiframes" and preprocess:
            next_state = self.to_grayscale_and_resize(next_state)
        return next_state, reward, done, info
    
    def render(self, mode="humn"):
        self.env.render(mode=mode)
    
    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.__observation_space
    
    @property
    def action_space(self):
        return self.__action_space

    def make_multiframes(self, img, num_frames=5):
        """Get initial states for Atari games."""
        return np.repeat(
            np.expand_dims(self.to_grayscale_and_resize(img), axis=-1),
            num_frames,
            axis=-1
        )

    def to_grayscale_and_resize(self, img, size=(84, 84)):
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[34: 34 + 160, :], size)
    
    def seed(self, seed):
        self.env.seed(seed)