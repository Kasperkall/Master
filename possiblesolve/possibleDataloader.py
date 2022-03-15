import os
import sys
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import matplotlib.pyplot as plt



class ScanDataset(Dataset):
    def __init__(self, ds_dir, input_img_name, gt_img_name, transforms=ToTensorV2()):
        super().__init__()

        self.transforms = transforms

        self.imgs_paths = [os.path.join(ds_dir, scan_dir, input_img_name) for scan_dir in os.listdir(ds_dir)]
        self.segs_paths = [os.path.join(ds_dir, scan_dir, gt_img_name) for scan_dir in os.listdir(ds_dir)]

        self.imgs_paths.sort()
        self.segs_paths.sort()


    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):

        img = np.array(Image.open(self.imgs_paths[idx]).convert("RGB"))
        seg = np.array(Image.open(self.segs_paths[idx]).convert("L"))
        seg = np.where(seg>20, 1, 0)
        seg = seg.clip(max=1)

        if self.transforms:
            augmentations = self.transforms(image=img, mask=seg)
            img = augmentations["image"]
            seg = augmentations["mask"]

        #img = torch.tensor(img, dtype=torch.float32).permute([2,0,1])
        #seg = torch.tensor(seg, dtype=torch.torch.int64)
        img = img.float()
        return img,seg

def get_dataloader(batch_size, ds_dir, input_img_name, gt_img_name, transforms=ToTensorV2(), shuffle=False):
    ds = ScanDataset(ds_dir, input_img_name, gt_img_name, transforms)
    loader = DataLoader(ds, batch_size, shuffle)
    return loader
    




if __name__ == '__main__':
    loader = get_dataloader(2, "scan-dataset", "img_l.png", "gt_scan.png")

    for x,y in loader:
        print(x.shape)
        print(y.shape)
        print(type(x[0][0][0]))
        break