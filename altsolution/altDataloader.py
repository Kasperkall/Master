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


class SegmentationDataset(Dataset):

    def __init__(self, input_dir, gt_dir, ds_length=4000, transforms=ToTensorV2()):
            super().__init__()
            if ds_length>4000:
                assert("Max number of images is 4000")
            self.transforms = transforms
            self.imgs_paths = [os.path.join(input_dir, img_file) for img_file in os.listdir(input_dir)]
            self.segs_paths = [os.path.join(gt_dir, img_file) for img_file in os.listdir(gt_dir)]
            self.imgs_paths.sort()
            self.segs_paths.sort()
            #This is to shorten the dataset if wanted 
            self.imgs_paths = self.imgs_paths[:ds_length]
            self.segs_paths = self.segs_paths[:ds_length]



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

    def get_dataloaders(input_dir, gt_dir, batch_size, val_frac, ds_length=4000, transforms=ToTensorV2(), shuffle=False):

        dataset = SegmentationDataset(input_dir, gt_dir, ds_length, transforms) #Here we get the entire dataset
        num_val_images = int(val_frac*len(dataset)) #Get the amount images that are going into the validation set based on the validation fraction
        num_train_images = len(dataset) - num_val_images  #Get the amount of the other part that goes to the training set

        train_set, val_set = torch.utils.data.random_split(dataset, [num_train_images, num_val_images]) #Randomly choosing which pair of images and ground truth that goes to the training set and the validation set
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) #Creating a dataloader for the training set
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True) #Creating dataloader for the validation set
        return train_loader, val_loader 



    if __name__ == '__main__':
        loader,val_loader = get_dataloaders("data/train_images", "data/train_masks",8, val_frac = 0.2)

        for x,y in loader:
            print(x.shape)
            print(y.shape)
            print(type(x[0][0][0]))

            fig,ax = plt.subplots(1,1)
            ax.set_axis_off()
            ax.set_title("input image")
            ax.imshow(x[0].detach().cpu().permute([1,2,0]))
            break