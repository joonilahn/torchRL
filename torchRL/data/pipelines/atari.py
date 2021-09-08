import cv2
import numpy as np

def to_grayscale_and_resize(img, size=(84, 84)):
    """Convert to grayscale and resize the input image"""
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[34: 34 + 160, :], size)


def get_init_state(img, num_frames=5):
    """Get initial states for Atari games."""
    return np.repeat(
        np.expand_dims(to_grayscale_and_resize(img), axis=-1),
        num_frames,
        axis=-1
    )