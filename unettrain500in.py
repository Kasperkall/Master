#matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.numeric import cross
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torchgeometry as tgm
#import torch.nn.functional as F
import torch.optim as optim
import os
import imageio
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import dataloaderv3
import yaml
from tqdm import tqdm
from models import simplecnn
from models import unet
import torchvision.transforms as transforms
import collections

#torch.set_printoptions(edgeitems=2)


#Here we import the yaml file and set some of the variables
config_file = open("configs/config.yaml")
cfg = yaml.load(config_file, Loader=yaml.FullLoader)

img_dir = cfg['img_dir']
gt_dir = cfg['gt_dir']
epochs = cfg['epochs']
batch_size = cfg['batch_size']
validation_frac = cfg['validation_fraction']
input_channels = cfg['input_channels']
output_channels = cfg['output_size']
validation_cadence = cfg['validation_cadence']
loss_weights = torch.tensor(cfg['loss_weights'])
save_dir = cfg['save_dir']
avg_per_im = cfg['avg_per_im']

class TrainingLoop:
    def __init__(self, sys_argv=None):
        self.use_cuda = torch.cuda.is_available() #Checks if we can run on GPU
        self.device = torch.device("cuda" if self.use_cuda else "cpu") #If we can we do, if not we run on cpu

        self.model = self.initModel() #Initializing the model 
        self.optimizer = self.initOptimizer() #Initializing the optimizer
    
    def initModel(self):
        model = unet.UNET()
        if self.use_cuda: 
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1: #checks to see if there are multiple GPUs
                model = nn.DataParallel(model)
            model = model.to(self.device) #Sends model parameters to the GPU
        return model

    def initOptimizer(self):
        return Adam(self.model.parameters()) #Adam maintains a separate learning rate for each parameter and automatically updates that learning rate as training progresses

    def initTrainAndValDl(self):
        train_dl,val_dl = dataloaderv3.get_dataloaders(img_dir, gt_dir, batch_size,validation_frac)
        return train_dl, val_dl
        
    def getBatchLoss(self, model_pred,gt_batch): 
        model_pred = model_pred.to(self.device)
        gt_batch = gt_batch.to(self.device)
        #lossfn = nn.CrossEntropyLoss(weight = loss_weights).to(self.device)
        #output = lossfn(model_pred, gt_batch)
        output = tgm.losses.dice.dice_loss(model_pred,gt_batch)
        #print(output)
        return output

    def main(self):
        train_dl,val_dl = self.initTrainAndValDl() # Getting the dataloaders for training and validation
        print("Starting {}, and running the model: {} with {} training / {} validation batches of size {}*{}".format(type(self).__name__, type(self.model),len(train_dl),len(val_dl),batch_size,(torch.cuda.device_count() if self.use_cuda else 1)))

        global_step = 0 #This is the value we use to keep track of the loss in the dictonaries 
        tracked_train_loss = collections.OrderedDict() #Dictonary for training loss
        tracked_val_loss = collections.OrderedDict() #Dictonary for validation loss
        
        for epoch_index in range(1, epochs + 1): #Looping through the epochs
            print("Epoch {} of {}, doing {} training / {} validation batches of size {}".format(epoch_index,epochs,len(train_dl),len(val_dl),batch_size,))

            self.model.train() #Tells our model we are training. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.
            train_loss_tot = 0.0
            val_loss_tot = 0.0
            avg_loss = 0

            for batch_i, (x_batch, gt_batch) in enumerate(train_dl):
                x_batch = x_batch.to(self.device)
                gt_batch = gt_batch.to(self.device)
                self.optimizer.zero_grad() #we have to null out the grad from the previous step so that it dosent accumulate
                train_pred = self.model(x_batch)
                train_loss = self.getBatchLoss(train_pred,gt_batch)
                train_loss.backward()
                self.optimizer.step()
                train_loss_tot = train_loss.mean()
                avg_loss += train_loss.cpu().detach().item()

                if epoch_index == epochs and batch_i<= 5: #Makes a image that shows the input,pred and gt at the last epoch
                    self.saveImages(x_batch,gt_batch,train_pred,batch_i,"train")

                """
                #For a smoother function of the plotloss, but we like to se the independent changes of the batches
                if batch_i % (avg_per_im//batch_size) == 0 and batch_i != 0: # Track the average loss for every x-th image (set in config.yaml)
                    avg_loss /= (avg_per_im//batch_size)
                    tracked_train_loss[global_step] = avg_loss 
                    
                    #OBS!OBS!OBS! Since we track the avg-loss for every x-th image, make sure that avg-per-im is set small enough compared to the epoch to get the graph
                    #For example: If the dataset contains 10 images and we run 5epochs, but have avg_per_im at 51, we wont get a value to plot since we never reach the necessary images to average over!
                    
                    avg_loss = 0
                """    
                tracked_train_loss[global_step] = train_loss_tot    
                global_step += batch_size

            if epoch_index == 0 or epoch_index % validation_cadence == 0 or epoch_index == epochs: #this is where the validation happens
                with torch.no_grad(): #We dont want our model to be influenced by the validation
                    self.model.eval() #Tells the model we are evaluating not training
                    val_loss = 0.0
                    val_loss_avg = 0
                    total_steps = 0
                    tot_im = 0
                    tot_cor = 0

                    for x_batch, gt_batch in val_dl:
                        x_batch = x_batch.to(self.device)
                        gt_batch = gt_batch.to(self.device)
                        val_pred = self.model(x_batch)
                        val_loss = self.getBatchLoss(val_pred,gt_batch)
                        val_loss_tot = val_loss.mean()
                        val_loss_avg += val_loss.cpu().item()
                        total_steps += 1
                        #tot_cor += (val_pred == gt_batch).sum().item()
                        tot_im += val_pred.shape[0]

                        if epoch_index == epochs:
                            self.saveImages(x_batch,gt_batch,train_pred,batch_i,"val") #prints two of the predictions and their gt from the last validation
                    
                    val_loss_avg = val_loss_avg / total_steps
                    #accuracy = tot_cor / tot_im
                    tracked_val_loss[global_step] = val_loss_avg

                print("Epoch {}/{} with Training Loss: {} and Validation Loss {}".format(epoch_index,epochs,tracked_train_loss[global_step],tracked_val_loss[global_step]))

                self.getAccuracy(train_dl,val_dl) #Gets the training and validation accuracy
        self.plotLoss(tracked_train_loss,tracked_val_loss) #Plots the loss for the entire training loop
                    

    def showImages(self,pred,gt_batch):
        f, axarr = plt.subplots(2,2)
        gt1 = gt_batch[0]
        gt2 = gt_batch[1]
        pred1 = pred[0][1]
        pred2 = pred[1][1]
        axarr[0,0].imshow(gt1.detach())
        axarr[0,1].imshow(gt2.detach())
        axarr[1,0].imshow(pred1.detach())
        axarr[1,1].imshow(pred2.detach())
        plt.show()
    
    def getStats(self,pred,gt):
        pred = torch.round(pred)
        all_pos = int((gt==1).sum())
        all_neg = int((gt==0).sum())

        correct = (pred == gt).float().sum()
        acc = float(correct/(all_pos+all_neg))
        true_pos = int(((pred==1) & (gt==1)).sum())
        false_pos = int(((pred==1) & (gt==0)).sum())
        true_neg = int(((pred==0) & (gt==0)).sum())
        false_neg = int(((pred==0) & (gt==1)).sum())

        dice = (2*true_pos) / (2*true_pos + false_neg + false_pos)
        #print("acc",acc)
        print("true_pos {} / {} and false_pos {}".format(true_pos,false_pos,all_pos))
        print("true_neg {} / {} and false_neg {}".format(true_neg,all_neg,false_neg))
        #print("DICE", dice) #This is wrong
        return dice
       
    def saveImages(self, X_batch, Y_batch, outputs, batch,name):
        #print("X",X_batch.shape)
        #print("Y",Y_batch.shape)
        #print("out",outputs.shape)
        #Have to normalize image to be shown by mathplotlib
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
        fig.savefig(os.path.join(save_dir, name+"_ubatch_"+format(batch, "02d")+".png"))

    """
    def saveImages(self,X_batch, Y_batch, outputs, epoch):
        fig,ax = plt.subplots(2,2)
        ax[0][0].set_axis_off()
        ax[0][0].set_title("input image")
        ax[0][0].imshow(X_batch[0].detach().cpu().permute([1,2,0]))
        ax[0][1].set_axis_off()
        ax[0][1].set_title("ground truth")
        ax[0][1].imshow(Y_batch[0].detach().cpu())
        ax[1][0].set_axis_off()
        ax[1][0].set_title("background pred")
        ax[1][0].imshow(outputs[0][0].detach().cpu()) # class 0: background pred
        ax[1][1].set_axis_off()
        ax[1][1].set_title("laser pred")
        ax[1][1].imshow(outputs[0][1].detach().cpu()) # class 1: laser pred
        fig.savefig(os.path.join(save_dir, "epoch_"+format(epoch, "02d")+".png"))
    """

    def getAccuracy(self,train_dl, val_dl):
        for name, loader in [("train", train_dl), ("val", val_dl)]:
            correct = 0
            total = 0

            with torch.no_grad(): #Dont want to update parameters
                for imgs, labels in loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(imgs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.shape[0]*labels.shape[1]*labels.shape[2]
                    correct += int((predicted == labels).sum())

            print("Accuracy {}: {}".format(name , correct / total))

    def plotLoss(self,train_dict, val_dict):
        fig,ax = plt.subplots()
        global_steps = list(train_dict.keys())
        losst = list(train_dict.values())
        plt.plot(global_steps, losst, label="Train Loss")

        global_steps = list(val_dict.keys())
        lossv = list(val_dict.values())
        plt.plot(global_steps, lossv, label="Val Loss")

        plt.legend()
        plt.xlabel("Global Training Step")
        plt.ylabel("Cross Entropy Loss")
        plt.ylim(0,0.1)
        fig.savefig("results/loss.png")



if __name__ == '__main__':
    TrainingLoop().main() #instantiates the application object and invokes the main method