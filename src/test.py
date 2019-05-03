from model import get_sample_model
import os
import torch
from torchvision.utils import save_image


class Tester:
    def __init__(self, config, data_loader):
        self.data_loader = data_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epoch = config.num_epoch
        self.from_style = config.from_style
        self.to_style = config.to_style
        self.config = config
        self.sample_dir = config.sample_dir
        self.total_step = len(self.data_loader)

    def test(self):
        style_dir = os.path.join(self.sample_dir, f"{self.from_style}2{self.to_style}")
        if not os.path.exists(style_dir):
            os.makedirs(style_dir)

        for epoch in range(1, self.num_epoch):
            generator_ab, generator_ba = get_sample_model(self.config, self.from_style, self.to_style, epoch)
            if not os.path.exists(os.path.join(style_dir, str(epoch))):
                os.makedirs(os.path.join(style_dir, str(epoch)))
            for step, image in enumerate(self.data_loader):

                real_image_a = image["A"].to(self.device)
                real_image_b = image["B"].to(self.device)

                fake_image_b = generator_ab(real_image_a)
                fake_image_a = generator_ba(real_image_b)

                to_style_image = torch.cat([real_image_a, fake_image_b], 2)
                from_style_image = torch.cat([real_image_b, fake_image_a], 2)

                save_image(to_style_image, f"{self.sample_dir}/{self.from_style}2{self.to_style}/{epoch}/{step}_{self.from_style}2{self.to_style}.png", normalize=False)
                save_image(from_style_image, f"{self.sample_dir}/{self.from_style}2{self.to_style}/{epoch}/{step}_{self.to_style}2{self.from_style}.png", normalize=False)
                print(f"[epoch: {epoch}][{step}/{self.total_step}]save image")
  #              if step % 10 == 9:
  #                  break

