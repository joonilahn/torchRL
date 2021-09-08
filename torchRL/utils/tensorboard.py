from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_info(self, tag, val, n):
        if isinstance(val, str):
            self.writer.add_text(tag, val, n)
        else:
            self.writer.add_scalar(tag, val, n)
    
    def close(self):
        self.writer.close()