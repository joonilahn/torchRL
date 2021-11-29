import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from ..builder import PIPELINES


@PIPELINES.register_module()
class BottomCrop:
    def __init__(self, cfg):
        pass

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, h - w, w, h))


@PIPELINES.register_module()
class Resize:
    def __init__(self, cfg):
        self.size = cfg.get("RESIZE", None)
        if self.size is None:
            raise TypeError("Size is not defined.")
        if not isinstance(self.size, (list, tuple)):
            raise TypeError("Size must be either list or tuple.")

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return F.resize(img, self.size)
        elif isinstance(img, np.ndarray):
            return cv2.resize(img, self.size)
        else:
            raise TypeError("Input should be PIL Image or numpy.ndarray")


@PIPELINES.register_module()
class Grayscale:
    def __init__(self, cfg):
        pass

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return F.to_grayscale(img, num_output_channels=1)
        elif isinstance(img, np.ndarray):
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError("Input should be PIL Image or numpy.ndarray")


@PIPELINES.register_module()
class ToPILImage:
    def __init__(self, cfg):
        self.mode = cfg.get("PIL_MODE", None)
        self.mode = self.mode if self.mode != "" else None

    def __call__(self, img):
        return F.to_pil_image(img, mode=self.mode)


@PIPELINES.register_module()
class ToTensor:
    def __init__(self, cfg):
        pass

    def __call__(self, img):
        if len(img.shape) == 1:
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        else:
            return F.to_tensor(img)


@PIPELINES.register_module()
class Normalize:
    def __init__(self, cfg):
        self.mean = cfg.get("MEAN", None)
        self.std = cfg.get("STD", None)
        if self.mean is None or self.std is None:
            raise TypeError("Mean or std is not defined.")

    def __call__(self, img):
        return F.normalize(img, mean=self.mean, std=self.std)


@PIPELINES.register_module()
class ToCuda:
    def __init__(self, cfg):
        pass

    def __call__(self, img):
        return img.to(torch.device("cuda"))


@PIPELINES.register_module()
class ToArray:
    """Convert frames numpy array"""
    def __init__(self, cfg):
        pass

    def __call__(self, imgs):
        return np.array(imgs)


@PIPELINES.register_module()
class TransformAtariInput:
    """Get first 4 frames of the array in the last axis"""

    def __init__(self, cfg):
        pass

    def __call__(self, imgs):
        if imgs.shape[-1] != 4:
            return imgs[:, :, :4]
        return imgs