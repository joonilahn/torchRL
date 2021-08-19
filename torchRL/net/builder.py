from ..utils import Registry

NETS = Registry("NETS")


def build_Qnet(cfg):
    """build net"""
    if "mlp" in cfg.NAME.lower():
        cfg_dict = {
            "type": cfg.NAME,
            "state_dim": cfg.STATE_DIM,
            "action_dim": cfg.ACTION_DIM,
            "hidden_dim": cfg.HIDDEN_DIM,
            "num_layers": cfg.NUM_LAYERS,
        }
    else:
        cfg_dict = {
            "type": cfg.NAME,
            "hidden_dim": cfg.HIDDEN_DIM,
            "action_dim": cfg.ACTION_DIM,
        }

    return NETS.build(cfg_dict)

def build_Vnet(cfg):
    """build net"""
    cfg_dict = {
        "type": cfg.NAME,
        "state_dim": cfg.STATE_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "num_layers": cfg.NUM_LAYERS,
    }
    return NETS.build(cfg_dict)

def build_ActorCritic(cfg):
    """build net"""
    return build_Qnet(cfg)
