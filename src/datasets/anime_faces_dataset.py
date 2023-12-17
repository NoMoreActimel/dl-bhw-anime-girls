import json
import os
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.utils import ROOT_PATH


class AnimeFacesDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42

    def __init__(self,
                 data_dir,
                 val_size = 0.1,
                 image_reshape_size = (64, 64),
                 train = True,
                 *args,
                 **kwargs):
        
        self.train = train
        self.image_reshape_size = image_reshape_size
        self.val_size = val_size

        self.data_dir = Path(data_dir)
        self.index = self.create_index()

        self.transform = v2.Compose([
            v2.Resize(size=self.image_reshape_size, antialias=True),
            # v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def create_index(self):
        if "train" not in os.listdir(self.data_dir):
            self.image_files_list = list(os.listdir(self.data_dir))
            image_files_train, image_files_val = train_test_split(
                self.image_files_list,
                test_size=self.val_size,
                random_state=self.TRAIN_VAL_RANDOM_SEED
            )

            image_files_dict = {"train": image_files_train, "val": image_files_val}
            for part, files_list in image_files_dict.items():
                with open(self.data_dir / part, 'w+', encoding='utf-8') as f:
                    f.writelines(files_list)        

        index = image_files_dict["train" if self.train else "val"]
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ind):
        image_filename = self.index[ind]
        image = Image.open(self.data_dir / image_filename).convert("RGB")
        image = self.transform(image)
        return image
