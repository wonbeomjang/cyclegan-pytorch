from src.train import Trainer
from src.test import Tester
from dataloader.dataloader import get_loader
import os
from config.config import get_config


def main(config):
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    print(f"{config.from_style} to {config.to_style} CycleGAN")

    data_loader, val_data_loader = get_loader(config.from_style, config.to_style, config)
    trainer = Trainer(config, data_loader)
    trainer.train()

    tester = Tester(config, val_data_loader)
    tester.test()


if __name__ == "__main__":
    config = get_config()
    main(config)
