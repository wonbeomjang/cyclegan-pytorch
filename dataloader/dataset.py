from torch.utils.data import Dataset
from PIL import Image
import os
import glob
from torchvision import transforms


class GanDataset(Dataset):
    def __init__(self, data_folder, from_style, to_style, image_size):
        self.data_folder = data_folder
        self.image_size = image_size

        if not os.path.exists(data_folder):
            raise Exception(f"[!] {self.data_folder} not exited")

        self.files_a = sorted(glob.glob(os.path.join(data_folder, f"{from_style}", "*.*")))
        self.files_b = sorted(glob.glob(os.path.join(data_folder, f"{to_style}", "*.*")))

    def __getitem__(self, item, transform=True):
        image_a = self.files_a[item]
        image_b = self.files_b[item]

        image_a = Image.open(image_a).convert('RGB')
        image_b = Image.open(image_b).convert('RGB')

        if transform:
            transform_a = transforms.Compose([
                transforms.CenterCrop(min(image_a.size[0], image_a.size[1])),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_b = transforms.Compose([
                transforms.CenterCrop(min(image_b.size[0], image_b.size[1])),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            image_a = transform_a(image_a)
            image_b = transform_b(image_b)


        return {'A': image_a, 'B': image_b}

    def __len__(self):
        return min(len(self.files_a), len(self.files_b))-1
