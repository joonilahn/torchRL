from ..utils import Registry

TRAINERS = Registry("TRAINERS")


def build_trainer(env, cfg):
    """build trainer"""
    cfg_dict = {"type": cfg.TRAIN.TRAINER, "env": env, "cfg": cfg}
    return TRAINERS.build(cfg_dict)
