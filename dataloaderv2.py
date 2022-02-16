import numpy as np
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import glob
from skimage import io


class SegmentationDataset(Dataset):
    """
    The __init__() method loads data into memory from file using the NumPy loadtxt() function 
    and then converts the data to PyTorch tensors.
    gt stands for ground truth.
    We have not yet divided into training and validation dataloaders, therefor we use terms images and gt.
    """
    def __init__(self, img_dir, gt_dir,img_transform=transforms.Compose([transforms.ToTensor()]), #kan velge å legge til transformasjoner her
        gt_transform=transforms.Compose([transforms.ToTensor(),])):

        super().__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_paths = glob.glob(img_dir + "/*.png")
        self.gt_paths = glob.glob(gt_dir + "/*.png")

        self.img_transform = img_transform
        self.gt_transform = gt_transform
        
        self.train_paths.sort()
        self.gt_paths.sort()
        """
        self.length = min(len(self.train_paths), len(self.gt_paths))
        """
        #print(len(self.img_paths)) #How many images that are in the train_images
        #print(len(self.gt_paths)) #How many gt-images that are in the train_masks
        #print(self.img_paths[:3]) #Name of the images
        #print(self.gt_paths[:3]) #Name of the gt

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = io.imread(self.img_paths[index])
        gt = io.imread(self.gt_paths[index])

        if self.img_transform:
            img = self.img_transform(img)
            gt = self.gt_transform(gt)
            gt = torch.flatten(gt)

        return (img, gt) #tuppel since we want them together

def get_dataloaders(train_dir, gt_dir, batch_size, validation_frac): #Here we divide into training and validation dataloaders
    dataset = SegmentationDataset(train_dir, gt_dir) #Here we get the entire dataset
    num_val_images = int(validation_frac*len(dataset)) #Get the amount images that are going into the validation set based on the validation fraction
    num_train_images = len(dataset) - num_val_images  #Get the amount of the other part that goes to the training set

    train_set, val_set = torch.utils.data.random_split(dataset, [num_train_images, num_val_images]) #Randomly choosing which pair of images and ground truth that goes to the training set and the validation set
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) #Creating a dataloader for the training set
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True) #Creating dataloader for the validation set
    return train_loader, val_loader 

if __name__ == '__main__': #Har veldig lyst til å få til yaml filer 
    import argparse
    import yaml
    #from utils import get_tensor_as_image_grayscale, get_tensor_as_image
    import matplotlib.pyplot as plt

    batch_size = 2
    learning_rate= 0.0003
    input_channels= 3
    input_width= 300
    input_height= 300
    output_size= 90000
    epochs= 15

    train_dir_path = "data/train_images" #OBS pass på at pathen er riktig!
    gt_dir_path = "data/train_masks"
    val_frac = 0.2

    train_loader, val_loader = get_dataloaders(train_dir_path, gt_dir_path, batch_size, val_frac)
    train_iter = iter(train_loader)
    print(type(train_iter))
    images, groundt = train_iter.next()
    images, groundt = train_iter.next()
    print('images shape on batch size = {}'.format(images.size()))
    print('groundt shape on batch size = {}'.format(groundt.size()))
    grid = torchvision.utils.make_grid(images)
    
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.show()
    f, axarr = plt.subplots(1,2)
    image = groundt[0].reshape(300,300)
    image2= groundt[1].reshape(300,300)
    axarr[0].imshow(image, cmap='gray')
    axarr[1].imshow(image2, cmap='gray')
    #plt.imshow(image, cmap='gray')
    #plt.imshow(image2, cmap='gray')
    plt.show()