"""

- Implementation of the GuSADataset Class.
- Initialization of the Dataset Loader.

"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from utils.transform import transform

class GuSADataset(Dataset):
    def __init__(self, annotations_file, img_dir,mask_dir,img_size=None,patch_size=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size=img_size
        self.patch_size=patch_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name=self.annotations.iloc[idx, 0]

        img_path = os.path.join(self.img_dir,img_name )
        mask_name=self.annotations.iloc[idx, 1]
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        image_tensor=transform(image,img_size=self.img_size,patch_size=self.patch_size)
        mask_tensor=transform(mask,img_size=self.img_size,patch_size=self.patch_size)
        label = self.annotations.iloc[idx, 3]

        return({"image_tensor":image_tensor,"img_path":img_path,"mask_tensor":mask_tensor,"label":label})
    def split(self,random_seed=None,train_fraction=0.9,validation_fraction=0):
        train_len=int(len(self)*train_fraction)
        val_len=int(len(self)*validation_fraction)
        test_len=len(self)-(train_len+val_len)
        if (random_seed):
            train_dataset,validation_dataset, test_dataset=random_split(self,[train_len,val_len,test_len], generator=torch.Generator().manual_seed(random_seed))
        else:
            train_dataset,validation_dataset, test_dataset=random_split(self,[train_len,val_len,test_len], generator=torch.Generator())
        return(train_dataset,validation_dataset, test_dataset)
def make_loader(dataset:GuSADataset, batch_size=4):
    loader = DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         pin_memory=True, num_workers=2,shuffle=True)
    return loader
