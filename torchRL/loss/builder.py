import inspect

import torch

from ..utils import Registry

LOSSES = Registry('loss')

def register_torch_losses():
    torch_losses = []
    for module_name in dir(torch.nn):
        if module_name.startswith('__'):
            continue
        if 'loss' not in module_name.lower():
            continue
        _loss = getattr(torch.nn, module_name)
        if inspect.isclass(_loss) and issubclass(_loss, torch.nn.modules.loss._Loss):
            LOSSES.register_module()(_loss)
            torch_losses.append(module_name)
    return torch_losses

TORCH_LOSSES = register_torch_losses()

def build_loss(cfg):
    """build loss function"""
    cfg_dict = {"type": cfg.TYPE}
    return LOSSES.build(dict(cfg_dict, **cfg.ARGS))
