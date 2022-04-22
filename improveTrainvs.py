import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from models import unet

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

import dataloaderv3
import dataloaderv4

import torch.optim as optim
import torch.nn.functional as F
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
from numpy.core.numeric import cross
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
import torchgeometry as tgm
from torchgeometry.losses.dice import DiceLoss
import torchvision.transforms as transforms
import os
import collections
import seaborn as sn
import dataloaderv3
import yaml
from models import simplecnn
from models import unet
torch.manual_seed(123)
from datetime import datetime

from PIL import Image

#Here we import the yaml file and set some of the variables
config_file = open("configs/configAll.yaml")
cfg = yaml.load(config_file, Loader=yaml.FullLoader)

img_dir_first = cfg['img_dir_first']
gt_dir_first = cfg['gt_dir_first']
img_dir_transfer = cfg['img_dir_transfer']
gt_dir_transfer = cfg['gt_dir_transfer']

epochs_first = cfg['epochs']
epochs_transfer = cfg['epochs_transfer']
batch_size_first = cfg['batch_size']
batch_size_transfer = cfg['batch_size_transfer']

validation_cadence = cfg['validation_cadence']
validation_frac = cfg['validation_fraction']
loss_weights = torch.tensor(cfg['loss_weights'])
learning_rate_first = cfg['learning_rate']
learning_rate_transfer = cfg['learning_rate_transfer']

now = datetime.now() # current date and time
save_dir = cfg['save_dir'] + now.strftime("%H-%M")
os.makedirs(save_dir, exist_ok=True)

save_name =""
tracked_train_acc = []
tracked_val_acc = []
tracked_dice = []
tracked_train_loss = collections.OrderedDict() #Dictonary for training loss
tracked_val_loss = collections.OrderedDict() #Dictonary for validation loss

tracked_val_acc_real = []
tracked_dice_real = []
tracked_val_loss_real = collections.OrderedDict() #Dictonary for validation loss

global global_step
global_step = 0 #This is the value we use to keep track of the loss in the dictonaries


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

def getAccuracy(model,train_dl, val_dl):
        trainacc = 0
        valacc = 0
        valvalues = []
        for name, loader in [("train", train_dl), ("val", val_dl)]:
            tn = 0
            tot_tn = 0
            tp = 0 #true positive
            tot_tp = 0
            fp = 0
            fn = 0
            correct = 0
            total = 0
            
            with torch.no_grad(): #Dont want to update parameters
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.shape[0]*labels.shape[1]*labels.shape[2]
                    correct += int((predicted == labels).sum())
                    if name == "val":
                        tp += int(((predicted == labels)&(labels==1)).sum()) #Vil vi ha precision og recall gjennom læringen eller bare på siste epoke?  Spare tid på kjøring ved å kun ha til slutt
                        tot_tp += int((labels == 1).sum())

                        tn += int(((predicted == labels)&(labels==0)).sum())
                        tot_tn += int((labels == 0).sum())

                        fp += int(((predicted != labels)&(labels==0)).sum())

                        fn += int(((predicted != labels)&(labels==1)).sum())

                        #print("tp/tot_tp=",tp,tot_tp," tn/tot_tn=",tn,tot_tn," fp=",fp," fn=",fn)

            print("Accuracy {}: {}".format(name , correct / total))
            if name == "train":
                trainacc = correct/total
            if name == "val":
                valacc = correct/total
                valvalues = [tp,tot_tp,tn, tot_tn, fp ,fn]
                recall = tp/(tp + fn) #True positive rate
                if (tp+fp)==0:
                    precision = 0
                else:
                    precision = tp/(tp+fp) #Positive predictive value
                specificity = tn/(tn+fp) #True negative rate
                if precision == 0:
                    f1 = 0
                else:
                    f1 = 2*(precision * recall)/(precision + recall)
                dice_score = (2*tp)/((tp+fp)+(tp+fn)) #F1 and dice is the same, remove this
                print("dice:",dice_score, "   recall/sensitivity/TPR:",recall,"  precision/PPV:",precision, "  specificity/TNR:",specificity, "  f1:",f1)
                
        return trainacc,valacc, dice_score, valvalues

def plotAccuracy(train,val,val_real,dice,dice_real):
    fig,ax = plt.subplots()
    plt.plot(train, label="Train accuracy datagenerated images")
    plt.xticks(np.arange(len(train)), np.arange(1, len(train)+1))
    plt.plot(val, label="Val acccuracy datagenerated images")
    plt.xticks(np.arange(len(val)), np.arange(1, len(val)+1))

    plt.plot(val_real, label="Val acccuracy real images")
    plt.xticks(np.arange(len(val_real)), np.arange(1, len(val_real)+1))

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Pixel accuracy")
    #plt.ylim(0.85,1)
    fig.savefig(os.path.join(save_dir, "accuracy.png"), dpi=600)
    plt.close(fig)

    fig,ax = plt.subplots()
    plt.plot(dice, label="Validation dice score datagenerated images")
    plt.xticks(np.arange(len(dice)), np.arange(1, len(dice)+1))
    plt.plot(dice_real, label="Validation dice score real images")
    plt.xticks(np.arange(len(dice_real)), np.arange(1, len(dice_real)+1))

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Dice score")
    #plt.ylim(0.85,1)
    fig.savefig(os.path.join(save_dir, "dicescore.png"), dpi=600)
    plt.close(fig)

def plotCM(values):#plots the confussion matrix for the last validation
        values=[values[2],values[4],values[5],values[0]]
        cf_matrix = np.asarray(values).reshape(2,2)
        test = np.array([[0,0],[0,0]])

        group_names = ['True Negative','False Positive','False Negative','True Positive']

        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]

        #group_percentages = ["{0:.2%}".format(value) for value in
        #                   cf_matrix.flatten()/np.sum(cf_matrix)]

        labels = [f"{v1}\n{v2}" for v1, v2 in #\n{v3}
                zip(group_names,group_counts)] #,group_percentages
        labels = np.asarray(labels).reshape(2,2)

        ax = sn.heatmap(test, annot=labels, fmt='', cbar=False, cmap='Blues',linewidths=0.5, linecolor='black')

        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
        ax.hlines([3, 6, 9], *ax.get_xlim())
        plt.savefig(os.path.join(save_dir, "CFmatrix.png"), dpi=600)
        plt.close()

def save_images(X_batch, Y_batch, outputs, epoch, save_dir, mode):

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
    image = X_batch[0].detach().cpu()
    image = image - image.min()
    image = image/image.max()
    image = image.numpy()
    image = np.moveaxis(image,0,2)

    _, predicted = torch.max(outputs, dim=1)

    fig,ax = plt.subplots(1,3)
    ax[0].set_axis_off()
    ax[0].set_title("input image")
    #ax[0].imshow(X_batch[0].detach().cpu().permute([1,2,0]))
    ax[0].imshow(image)
    ax[1].set_axis_off()
    ax[1].set_title("ground truth")
    ax[1].imshow(Y_batch[0].detach().cpu(), cmap='gray')
    ax[2].set_axis_off()
    ax[2].set_title("model pred") 
    ax[2].imshow(predicted[0].detach().cpu(), cmap='gray') # class 1: laser pred
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "img_compare_"+ mode+format(epoch, "02d")+"First.png"), dpi=600)
    plt.close(fig)

    """
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
    fig.savefig(os.path.join(save_dir, mode +"epoch_"+format(epoch, "02d")+".png"), dpi=600)
    plt.close(fig)
    """
    
    #FULL IMAGES
    pred_seg = Image.fromarray(segm)
    segm_save_dir = os.path.join(save_dir, "segmentations_preds", )
    os.makedirs(segm_save_dir, exist_ok=True)
    segm_save_path = os.path.join(segm_save_dir, mode + "epoch"+format(epoch, "02d")+".png")
    pred_seg.save(segm_save_path)

    segm_gt_comp = np.dstack((segm, gt_img, np.zeros_like(segm)))
    segm_gt_comp = Image.fromarray(segm_gt_comp)
    segm_gt_save_dir = os.path.join(save_dir, "segm_gt_comparison", )
    os.makedirs(segm_gt_save_dir, exist_ok=True)
    segm_gt_save_path = os.path.join(segm_gt_save_dir, mode + "epoch"+format(epoch, "02d")+".png")
    segm_gt_comp.save(segm_gt_save_path)

def plotLoss(train_dict, val_dict, val_dict_real):
    fig,ax = plt.subplots()
    global_steps = list(train_dict.keys())
    losst = list(train_dict.values())
    losst_float = list(map(float,losst))
    plt.plot(global_steps, losst_float, label="Train Loss datagenerated images")

    global_steps = list(val_dict.keys())
    lossv = list(val_dict.values())
    lossv_float = list(map(float,lossv))
    plt.plot(global_steps, lossv_float, label="Val Loss datagenerated images")

    global_steps = list(val_dict_real.keys())
    lossvr = list(val_dict_real.values())
    lossvr_float = list(map(float,lossvr))
    plt.plot(global_steps, lossvr_float, label="Val Loss real images")

    plt.legend()
    plt.xlabel("Global Training Step")
    plt.ylabel("Cross Entropy Loss")
    plt.ylim(0,0.2)
    fig.savefig(os.path.join(save_dir, "loss.png"), dpi=600)
    plt.close(fig)


def train_step(X_batch, Y_batch, optimizer, model, loss_fn):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = loss_fn(outputs, Y_batch)
    #print(loss.item())
    loss.backward()
    optimizer.step()
    return outputs,loss

def val_step(X_batch, Y_batch, optimizer, model, loss_fn):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = loss_fn(outputs, Y_batch)
    return outputs,loss

def runit(model, train_dl, val_dl,val_dl_real, loss_fn, optimizer, batch_size, epochs, cadence,mode):
    global global_step
    #training
    for epoch in range(1, epochs + 1):
        print("\nEPOCH", epoch)
        model.train()
        savefig=True 
        for X_batch, Y_batch in (train_dl):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs,train_loss = train_step(X_batch, Y_batch, optimizer, model, loss_fn)
            tracked_train_loss[global_step] = train_loss 
            global_step += batch_size   
            
        #validation
        if epoch == 1 or epoch % validation_cadence == 0 or epoch == epochs: #this is where the validation happens
            model.eval()
            
            with torch.no_grad():
                for X_batch, Y_batch in val_dl:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    outputs,val_loss = val_step(X_batch, Y_batch, optimizer, model, loss_fn)
                    tracked_val_loss[global_step] = val_loss
                    if savefig:
                        avg_batch_dice_score = mean_dice_coef_over_batch(Y_batch, outputs)
                        #print("Batch dice score", avg_batch_dice_score)
                        save_images(X_batch, Y_batch, outputs, epoch, save_dir, mode)
            
            with torch.no_grad():
                for X_batch, Y_batch in val_dl_real:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    outputs,val_loss_real = val_step(X_batch, Y_batch, optimizer, model, loss_fn)
                    tracked_val_loss_real[global_step] = val_loss_real
                    if savefig:
                        avg_batch_dice_score_real = mean_dice_coef_over_batch(Y_batch, outputs)
                        #print("Batch dice score real", avg_batch_dice_score_real)
                        save_images(X_batch, Y_batch, outputs, epoch, save_dir, mode + "real")
                        savefig=False
            
        
        temp_acc_train, temp_acc_val, temp_dice, valvalues = getAccuracy(model,train_dl,val_dl) #Gets the training and validation accuracy
        print("Val loss", val_loss)
        tracked_train_acc.append(temp_acc_train)
        tracked_val_acc.append(temp_acc_val)
        tracked_dice.append(temp_dice)

        print("Now for real images")
        temp_acc_train_real, temp_acc_val_real, temp_dice_real, valvalues_real = getAccuracy(model,train_dl,val_dl_real) #Gets the training and validation accuracy
        tracked_val_acc_real.append(temp_acc_val_real)
        tracked_dice_real.append(temp_dice_real)
        print("Val loss real images", val_loss_real)
    plotCM(valvalues_real) 

def transferit(model, train_dl, val_dl, loss_fn, optimizer, batch_size, epochs, cadence):
    global global_step
    #training
    for epoch in range(1, epochs + 1):
        print("\nEPOCH", epoch)
        model.train()
        savefig=True 
        for X_batch, Y_batch in (train_dl):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs,train_loss = train_step(X_batch, Y_batch, optimizer, model, loss_fn)
            tracked_train_loss[global_step] = train_loss 
            global_step += batch_size   
            
        #validation
        if epoch == 1 or epoch % validation_cadence == 0 or epoch == epochs: #this is where the validation happens
            model.eval()
            
            with torch.no_grad():
                for X_batch, Y_batch in val_dl:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    outputs,val_loss = val_step(X_batch, Y_batch, optimizer, model, loss_fn)
                    tracked_val_loss[global_step] = val_loss
                    if savefig:
                        avg_batch_dice_score = mean_dice_coef_over_batch(Y_batch, outputs)
                        #print("Batch dice score", avg_batch_dice_score)
                        save_images(X_batch, Y_batch, outputs, epoch, save_dir)
                        savefig=False
        
        temp_acc_train, temp_acc_val, temp_dice, valvalues = getAccuracy(model,train_dl,val_dl) #Gets the training and validation accuracy
        tracked_train_acc.append(temp_acc_train)
        tracked_val_acc.append(temp_acc_val)
        tracked_dice.append(temp_dice)
    plotCM(valvalues)
        

def main():
    global global_step
    # TRAIN TRANSFORMS
    tf = A.Compose([
        A.Normalize(mean=[0,0,0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    train_dl,val_dl = dataloaderv3.get_dataloaders(img_dir_first, gt_dir_first, batch_size_first,validation_frac)
    val_dl_real = dataloaderv4.get_dataloaders(img_dir_transfer,gt_dir_transfer,batch_size_transfer)
    print(len(train_dl), "train")
    print(len(val_dl),"val")
    print(len(val_dl_real),"real val")

    theunet = unet.UNET()
    theunet.to(device)

    opt = torch.optim.Adam(theunet.parameters(), lr=learning_rate_first)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights).to(device)
    #loss_fn = DiceLoss()

    
    runit(theunet, train_dl, val_dl, val_dl_real, loss_fn, opt, batch_size_first, epochs_first,validation_cadence,"first_") #første dataset

   # train_dl,val_dl = dataloaderv3.get_dataloaders(img_dir_transfer, gt_dir_transfer, batch_size_transfer,validation_frac)
   # runit(theunet, train_dl, val_dl, loss_fn, opt, batch_size_transfer, epochs_transfer,validation_cadence,"transfer_") #transfer learning til andre dataset

    plotAccuracy(tracked_train_acc,tracked_val_acc,tracked_val_acc_real, tracked_dice, tracked_dice_real)
    plotLoss(tracked_train_loss,tracked_val_loss, tracked_val_loss_real) #Plots the loss for the entire training loop



if __name__ == '__main__':
    main()
