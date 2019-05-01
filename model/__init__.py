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

    generator_ab_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}", f"generator_ab_{config.epoch-1}.pth"))
    generator_ba_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}", f"generator_ba_{config.epoch-1}.pth"))
    discriminator_a_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}", f"discriminator_a_{config.epoch-1}.pth"))
    discriminator_b_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}", f"discriminator_b_{config.epoch-1}.pth"))

    print(f"[*] Load checkpoint in {config.checkpoint_dir}")
    if not os.path.exists(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}")):
        os.makedirs(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}"))

    if len(os.listdir(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}"))) == 0:
        print(f"[!] No checkpoint in {config.checkpoint_dir}")
        generator_ab.apply(weights_init)
        generator_ba.apply(weights_init)
        discriminator_a.apply(weights_init)
        discriminator_b.apply(weights_init)
    else:
        generator_ab.load_state_dict(torch.load(generator_ab_param[-1], map_location=device))
        generator_ba.load_state_dict(torch.load(generator_ba_param[-1], map_location=device))
        discriminator_a.load_state_dict(torch.load(discriminator_a_param[-1], map_location=device))
        discriminator_b.load_state_dict(torch.load(discriminator_b_param[-1], map_location=device))

    return generator_ab, generator_ba, discriminator_a, discriminator_b


def get_sample_model(config, from_style, to_style, epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_ab = ResidualGenerator(config.image_size, config.num_residual_blocks).to(device)
    generator_ba = ResidualGenerator(config.image_size, config.num_residual_blocks).to(device)
    generator_ab_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}", f"generator_ab_{epoch}.pth"))
    generator_ba_param = glob(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}", f"generator_ab_{epoch}.pth"))
    print(f"[*] Load checkpoint in {epoch}")
    print(f"[*] load generator_ab_{epoch}.pth")
    if len(os.listdir(os.path.join(config.checkpoint_dir, f"{from_style}2{to_style}"))) == 0:
        raise Exception(f"[!] No checkpoint in {config.checkpoint_dir}")
    else:
        generator_ab.load_state_dict(torch.load(generator_ab_param[-1], map_location=device))
        generator_ba.load_state_dict(torch.load(generator_ba_param[-1], map_location=device))

    return generator_ab, generator_ba


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
