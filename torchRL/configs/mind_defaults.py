from yacs.config import CfgNode as CN

_C = CN()
_C.USE_GPU = False

_C.NET = CN()
_C.NET.NAME = "ActionValueMLP"
_C.NET.NUM_LAYERS = 2
_C.NET.HIDDEN_DIM = 18
_C.NET.ACTION_DIM = 64
_C.NET.STATE_DIM = 18

_C.ENV = CN()
_C.ENV.NAME = "Mind-v0"
_C.ENV.TYPE = "ClassicControl"
_C.ENV.TRAIN_DATA = ""  # file path for the train data (csv file)
_C.ENV.EVAL_DATA = ""   # file path for the val data (csv file)
_C.ENV.NEWS_LABEL = ""  # file path for the news labels (txt file)
_C.ENV.MAX_EPISODE_STEPS = 500
_C.ENV.NUM_CATEGORIES = 18
_C.ENV.NUM_OUTPUT = 1
_C.ENV.REWARD_SCALE = 1.0
_C.ENV.SEED = 10

_C.TRAIN = CN()
_C.TRAIN.TRAINER = ""
_C.TRAIN.NUM_EPISODES = 10000
_C.TRAIN.TRAIN_BY_EPISODE = True  # if False, train will be based on global iteration number
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.START_TRAIN = 0
_C.TRAIN.TRAIN_INTERVAL = 1
_C.TRAIN.NUM_ITERS_PER_TRAIN = 50
_C.TRAIN.DISCOUNT_RATE = 0.99
_C.TRAIN.VERBOSE_INTERVAL = 10
_C.TRAIN.TARGET_SYNC_INTERVAL = 2500
_C.TRAIN.HISTORY_SIZE = 100
_C.TRAIN.AVG_REWARDS_TO_TERMINATE = 99
_C.TRAIN.PRETRAINED = ""
_C.TRAIN.EVALUATE_INTERVAL = 1000

# For algorithms using epsilon greedy exploration
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = "LinearAnnealingScheduler"
_C.SCHEDULER.EPSILON_GREEDY_MINMAX = (0.01, 0.08)
_C.SCHEDULER.DECAY_PERIOD = (0, 2000)

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "Adam"
_C.OPTIMIZER.CLIP_GRAD = False
_C.OPTIMIZER.CLIP_GRAD_VALUE = 10.0
_C.OPTIMIZER.ARGS = CN(new_allowed=True)
_C.OPTIMIZER.ARGS.lr = 0.001

_C.LOSS_FN = CN()
_C.LOSS_FN.TYPE = "SmoothL1Loss"
_C.LOSS_FN.ARGS = CN(new_allowed=True)

_C.DATASET = CN()
_C.DATASET.TYPE = "BufferDataset"
_C.DATASET.BUFFER_SIZE = 50000
_C.DATASET.PIPELINES = []

_C.LOGGER = CN()
_C.LOGGER.LOG_NAME = "MIND"
_C.LOGGER.OUTPUT_DIR = f"./work_dir/{_C.LOGGER.LOG_NAME}"
_C.LOGGER.LOG_FILE = True
_C.LOGGER.LOG_TENSORBOARD = True
_C.LOGGER.SAVE_MODEL = True
_C.LOGGER.SAVE_MODEL_INTERVAL = 1000


def get_cfg_defaults():
    return _C.clone()