from model.model import ResidualGenerator, Discriminator
import os
from glob import glob
import torch


def build_model(config, from_style, to_style):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_ab = ResidualGenerator(config.image_size, config.num_residual_blocks).to(device)
    generator_ba = ResidualGenerator(config.image_size, config.num_residual_blocks).to(device)
    discriminator_a = Discriminator(config.image_size).to(device)
    discriminator_b = Discriminator(config.image_size).to(device)

    generator_ab_param = glob(os.path.join(config.checkporint_dir, f"{from_style}_{to_style}", "GeneratorAB*.pth"))
    generator_ba_param = glob(os.path.join(config.checkporint_dir, f"{from_style}_{to_style}", "GeneratorBA*.pth"))
    discriminator_a_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}_{to_style}", "DiscriminatorA*.pth"))
    discriminator_b_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}_{to_style}", "DiscriminatorB*.pth"))

    print(f"[*] Load checkpoint in {config.checkporint_dir}")
    if not os.path.exists(config.checkpoint_dir):
        print(f"[!] No checkpoint in {config.checkpoint_dir}")
        os.makedirs(config.checkpoint_dir)
        generator_ab.apply(weights_init)
        generator_ba.apply(weights_init)
        discriminator_a.apply(weights_init)
        discriminator_b.apply(weights_init)

    else:
        generator_ab.load_state_dict(torch.load(generator_ab_param, map_location=device))
        generator_ba.load_state_dict(torch.load(generator_ba_param, map_location=device))
        discriminator_a.load_state_dict(torch.load(discriminator_a_param, map_location=device))
        discriminator_b.load_state_dict(torch.load(discriminator_b_param, map_location=device))

    return generator_ab, generator_ba, discriminator_a, discriminator_b


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
