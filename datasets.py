from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from typing import List
from pathlib import Path
from PIL import Image
import numpy as np


class PaddyDataset(Dataset):
    def __init__(self, files:List[Path],label_idx:List[str], transform) -> None:
        super().__init__()
        self.img_files:List[Path] = files    
        self.labels = [label_idx[p.parts[-2]] for p in self.img_files]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_files)