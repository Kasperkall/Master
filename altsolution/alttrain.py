import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from altUnet import Unet2D

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from altDataloader import *

import torch.optim as optim
import torch.nn.functional as F
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from PIL import Image

from torchgeometry.losses.dice import DiceLoss

import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on ", device)

def predb_to_mask(pred_batch, idx):
    pred = pred_batch[idx]
    #Hadde linjen nedenfor der fra for men tror ikke den er nodvendig, softmax konverter til sannsynligheter
    #men den med max sannsynlighet er den som har hoyest aktivering
    #pred = torch.functional.F.softmax(pred, 0)     
    return pred.argmax(0)

def dice_coef(y_true, y_pred):
    #https://stackoverflow.com/questions/61488732/how-calculate-the-dice-coefficient-for-multi-class-segmentation-task-using-pytho
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    dice_score  = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice_score

def mean_dice_coef_over_batch(ground_truth, model_outputs):
    ground_truth = ground_truth.detach().cpu().numpy()
    model_outputs = model_outputs.detach().cpu().numpy()
    cumulative_dice = 0.0
    for i in range(ground_truth.shape[0]):
        y_segm_pred = predb_to_mask(model_outputs, i)
        cumulative_dice += dice_coef(ground_truth[i], y_segm_pred)
    avg_dice = cumulative_dice / ground_truth.shape[0]
    return avg_dice


def save_images(X_batch, Y_batch, outputs, epoch, save_dir):

    input_img = X_batch[0].detach().cpu().permute([1,2,0]).numpy()
    input_img = (input_img*254).astype(np.uint8)
    gt_img = Y_batch[0].detach().cpu().numpy()
    gt_img = (gt_img*254).astype(np.uint8)
    bg_pred = outputs[0][0].detach().cpu().numpy()
    bg_pred /= np.max(bg_pred)
    laser_pred = outputs[0][1].detach().cpu().numpy()
    laser_pred /= np.max(laser_pred)
    segm = predb_to_mask(outputs, 0).cpu().numpy()
    segm = (segm*255).astype(np.uint8)


    #MATPLOTLIB PLOTS
    fig,ax = plt.subplots(1,5)
    ax[0].set_axis_off()
    ax[0].set_title("input image")
    ax[0].imshow(input_img)
    ax[1].set_axis_off()
    ax[1].set_title("ground truth")
    ax[1].imshow(gt_img)
    ax[2].set_axis_off()
    ax[2].set_title("BG pred")
    ax[2].imshow(bg_pred) # class 0: background pred
    ax[3].set_axis_off()
    ax[3].set_title("laser pred")
    ax[3].imshow(laser_pred) # class 1: laser pred
    ax[4].set_axis_off()
    ax[4].set_title("segm pred")
    ax[4].imshow(segm) # segmentation prediction
    fig.savefig(os.path.join(save_dir, "epoch_"+format(epoch, "02d")+".png"), dpi=600)
    
    #FULL IMAGES
    pred_seg = Image.fromarray(segm)
    segm_save_dir = os.path.join(save_dir, "segmentations_preds", )
    os.makedirs(segm_save_dir, exist_ok=True)
    segm_save_path = os.path.join(segm_save_dir, "epoch"+format(epoch, "02d")+".png")
    pred_seg.save(segm_save_path)

    segm_gt_comp = np.dstack((segm, gt_img, np.zeros_like(segm)))
    segm_gt_comp = Image.fromarray(segm_gt_comp)
    segm_gt_save_dir = os.path.join(save_dir, "segm_gt_comparison", )
    os.makedirs(segm_gt_save_dir, exist_ok=True)
    segm_gt_save_path = os.path.join(segm_gt_save_dir, "epoch"+format(epoch, "02d")+".png")
    segm_gt_comp.save(segm_gt_save_path)


def train_step(X_batch, Y_batch, optimizer, model, loss_fn):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = loss_fn(outputs, Y_batch)
    #print(loss.item())
    loss.backward()
    optimizer.step()
    return outputs

def train(model, num_classes, train_dl, loss_fn, optimizer, epochs):

    for epoch in range(epochs):
        print("EPOCH", epoch)
        model.train()
        savefig=True 
        for X_batch, Y_batch in tqdm(train_dl):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = train_step(X_batch, Y_batch, optimizer, model, loss_fn)
            if savefig:
                avg_batch_dice_score = mean_dice_coef_over_batch(Y_batch, outputs)
                print("Batch dice score", avg_batch_dice_score)
                save_images(X_batch, Y_batch, outputs, epoch, "plots")
                savefig=False


def main():
    
    #HYPERPARAMETERS
    LEARNING_RATE = 3e-4
    CR_ENTR_WEIGHTS = torch.tensor([0.1,0.9]).to(device)
    BATCH_SIZE = 2
    INPUT_CHANNELS = 3
    NUM_CLASSES = 2

    #Getting hyperparameters form config-file 
    config_file = open("configs/configtest.yaml")
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    img_dir = cfg['img_dir']
    gt_dir = cfg['gt_dir']
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    validation_frac = cfg['validation_fraction']
    input_channels = cfg['input_channels']
    validation_cadence = cfg['validation_cadence']
    loss_weights = torch.tensor(cfg['loss_weights'])
    save_dir = cfg['save_dir']
    learning_rate = cfg['learning_rate']

    
    DS_LENGTH = 50 #set number of images that is loaded and used for training (max = 400)
    # if training on GPU, set to 400, else set lower since training on CPU is slow
    
    # TRAIN TRANSFORMS
    tf = A.Compose([
        A.Normalize(mean=[0,0,0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    #train_loader = get_dataloader(BATCH_SIZE, "scan-dataset", "img_l.png", "gt_scan.png", transforms=tf, shuffle=True)
    train_loader = get_dataloader(batch_size, img_dir, gt_dir, transforms=tf, shuffle=True)

    unet = Unet2D(input_channels,NUM_CLASSES)
    unet.to(device)




    opt = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    #loss_fn = torch.nn.CrossEntropyLoss(weight=CR_ENTR_WEIGHTS)
    loss_fn = DiceLoss()

    
    train(unet, NUM_CLASSES, train_loader, loss_fn, opt, epochs)



if __name__ == '__main__':
    main()