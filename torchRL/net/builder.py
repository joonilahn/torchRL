from ..utils import Registry

NETS = Registry("NETS")

def build_net(cfg):
    """build net"""
    cfg_dict = {
        "type": cfg.NAME,
        "state_dim": cfg.STATE_DIM,
        "action_dim": cfg.ACTION_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "num_layers": cfg.NUM_LAYERS,
    }
    return NETS.build(cfg_dict)