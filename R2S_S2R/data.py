from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os
path_r = "";path_f = ""

class RealData(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(path, "*"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert("RGB")

        if(self.transform):
            image = self.transform(image)
        
        return image

class FakeData(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(path, "*"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert("RGB")

        if(self.transform):
            image = self.transform(image)
        
        return image  


