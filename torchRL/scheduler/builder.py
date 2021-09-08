from ..utils import Registry

SCHEDULERS = Registry("SCHEDULERS")

def build_scheduler(cfg):
    """build scheduler"""
    cfg_dict = {"type": cfg.TYPE, "cfg": cfg}
    return SCHEDULERS.build(cfg_dict)