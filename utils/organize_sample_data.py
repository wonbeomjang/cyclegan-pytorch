import shutil
import os
from config.config import get_config
from glob import glob

config = get_config()

sample_epoch = glob(config.sample_dir)

for epoch in sample_epoch:
    