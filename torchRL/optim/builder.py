import inspect

import torch

from ..utils import Registry

OPTIMIZERS = Registry('optimizer')

def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers

TORCH_OPTIMIZERS = register_torch_optimizers()

def build_optimizer(model, cfg):
    """build optimizer"""
    cfg_dict = {"type": cfg.TYPE, "params": model.parameters()}
    return OPTIMIZERS.build(dict(cfg_dict, **cfg.ARGS))
