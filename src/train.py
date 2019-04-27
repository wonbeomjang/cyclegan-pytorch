from model.model import ResidualGenerator
from model.model import Discriminator
from model.model import weights_init
import torch


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkporint_dir