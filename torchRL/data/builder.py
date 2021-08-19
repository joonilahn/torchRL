from ..utils import Registry
from torchvision.transforms import Compose

DATASETS = Registry("DATASETS")
PIPELINES = Registry("PIPELINES")

def build_dataset(cfg):
    """build dataset"""
    return DATASETS.build({"type": cfg.DATASET.TYPE, "cfg": cfg.DATASET})

def build_pipeline(cfg):
    """build pipelines"""
    pipelines = []
    for t in cfg.PIPELINES:
        pipelines.append(
            PIPELINES.build({"type": t, "cfg": cfg})
        )
    return Compose(pipelines)