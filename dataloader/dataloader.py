from torch.utils.data import DataLoader
from dataloader.dataset import GanDataset


def get_loader(from_style, to_style, config):
    data_loader = DataLoader(GanDataset(config.dataset, from_style, to_style, image_size=config.image_size),
                             batch_size=config.batch_size, num_workers=config.workers)

    val_data_loader = DataLoader(GanDataset(config.dataset, from_style, to_style, image_size=config.image_size),
                                 batch_size=1, num_workers=config.workers)

    return data_loader, val_data_loader
