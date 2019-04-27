from torch.utils.data import Dataset
from PIL import Image
import os
import glob
from torchvision import transforms

class GanDataset(Dataset):
    def __init__(self, data_folder, from_style, to_style, image_size=224):
        self.data_folder = data_folder
        self.image_size = image_size

        if not os.path.exists(data_folder):
            raise Exception(f"[!] {self.data_folder} not exited")

        self.files_A = sorted(glob.glob(os.path.join(data_folder, f"{from_style}", "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(data_folder, f"{to_style}", "*.*")))

    def __getitem__(self, item, transform=False):
        image_A = self.files_A[item]
        image_B = self.files_B[item]

        image_A = Image.open(image_A).convert('RGB')
        image_B = Image.open(image_B).convert('RGB')

        if transform:
            transform_A = transforms.Compose([
                transforms.CenterCrop(min(image_A.size[0], image_A.size[1])),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transform_B = transforms.Compose([
                transforms.CenterCrop(min(image_B.size[0], image_B.size[1])),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            image_A = transform_A(image_A)
            image_B = transform_B(image_B)

        return {'A':image_A, 'B':image_B}
