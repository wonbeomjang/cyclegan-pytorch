import torch
import os
from PIL import Image
from model.model import Generator
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--param_path', default='parameters/monet.pth')
parser.add_argument('--input_dir')
parser.add_argument('--output_dir', default='results')

args = parser.parse_args()

images_name = sorted(os.listdir(args.input_dir))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    net = Generator().to(device).eval()
    net.load_state_dict(torch.load(args.param_path))

    for image_name in images_name:
        image = Image.open(os.path.join(args.input_dir, image_name)).convert('RGB')
        image = TF.to_tensor(image).to(device).unsqueeze(dim=0)
        image = net(image)
        save_image(image,  os.path.join(args.output_dir, image_name))
        print(f'save {image_name}')

