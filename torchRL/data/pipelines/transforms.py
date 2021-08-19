
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from ..builder import PIPELINES


@PIPELINES.register_module()
class BottomCrop(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, img):
        w, h = img.size
        return img.crop((0, h - w, w, h))


@PIPELINES.register_module()
class Resize(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.get("RESIZE", None)
        if self.size is None:
            raise TypeError("Size is not defined.")
        if not isinstance(self.size, (list, tuple)):
            raise TypeError("Size must be either list or tuple.")

    def forward(self, img):
        return F.resize(img, self.size)


@PIPELINES.register_module()
class Grayscale(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, img):
        return F.to_grayscale(img, num_output_channels=1)


@PIPELINES.register_module()
class ToPILImage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.get("PIL_MODE", None)
        self.mode = self.mode if self.mode != "" else None

    def forward(self, img):
        return F.to_pil_image(img, mode=self.mode)


@PIPELINES.register_module()
class ToTensor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, img):
        return F.to_tensor(img)


@PIPELINES.register_module()
class Normalize(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mean = cfg.get("MEAN", None)
        self.std = cfg.get("STD", None)
        if self.mean is None or self.std is None:
            raise TypeError("Mean or std is not defined.")

    def forward(self, img):
        return F.normalize(img, mean=self.mean, std=self.std)


@PIPELINES.register_module()
class ToDevice(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_gpu = cfg.get("USE_GPU", False)

    def forward(self, img):
        if self.use_gpu:
            return img.to(torch.device("cuda"))
        return img