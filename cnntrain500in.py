#matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.numeric import cross
import torch
import torch.nn as nn
import torchgeometry as tgm
#import torch.nn.functional as F
import torch.optim as optim
import imageio
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import dataloaderv2
import yaml
from tqdm import tqdm
from models import simplecnn
import os
import collections
#from sklearn import utils
#torch.set_printoptions(edgeitems=2)
torch.set_printoptions(threshold=np.inf)

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
save_dir = cfg['save_dir']
avg_per_im = cfg['avg_per_im']
loss_weights = torch.tensor(cfg['loss_weights'])

class TrainingLoop:
    def __init__(self, sys_argv=None):
        self.use_cuda = torch.cuda.is_available() #Checks if we can run on GPU
        self.device = torch.device("cuda" if self.use_cuda else "cpu") #If we can we do, if not we run on cpu
        self.model = self.initModel() #Initializing the model 
        self.optimizer = self.initOptimizer() #Initializing the optimizer
    
    def initModel(self):
        model = simplecnn.CustomModel()
        if self.use_cuda: 
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1: #checks to see if there are multiple GPUs
                model = nn.DataParallel(model)
            model = model.to(self.device) #Sends model parameters to the GPU
        return model

    def initOptimizer(self):
        return Adam(self.model.parameters()) #Adam maintains a separate learning rate for each parameter and automatically updates that learning rate as training progresses

    def initTrainAndValDl(self):
        train_dl,val_dl = dataloaderv2.get_dataloaders(img_dir, gt_dir, batch_size, validation_frac)
        return train_dl, val_dl
        
    def getBatchLoss(self, model_pred,gt_batch): 
        model_pred = model_pred.to(self.device)
        gt_batch = gt_batch.to(self.device)
        lossfn = nn.CrossEntropyLoss().to(self.device)
        output = lossfn(model_pred, torch.max(gt_batch, 1)[1])
        #output = tgm.losses.dice.dice_loss(model_pred,gt_batch)
        #print("output",output)
        return output

    def main(self):
        train_dl,val_dl = self.initTrainAndValDl() # Getting the dataloaders for training and validation
        #The training and validation sets are generated only once to make sure that images dont swap sets between epochs, making our validation set corrupted
        print("Starting {}, and running the model: {} with {} training / {} validation batches of size {}*{}".format(type(self).__name__, type(self.model),len(train_dl),len(val_dl),batch_size,(torch.cuda.device_count() if self.use_cuda else 1)))

        global_step = 0 #This is the value we use to keep track of the loss in the dictonaries 
        tracked_train_loss = collections.OrderedDict() #Dictonary for training loss
        tracked_val_loss = collections.OrderedDict() #Dictonary for validation loss

        for epoch_index in range(1, epochs + 1): #Looping through the epochs
            print("Epoch {} of {}, doing {} training / {} validation batches of size {}".format(epoch_index,epochs,len(train_dl),len(val_dl),batch_size,))

            self.model.train() #Tells our model we are training. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.
            #train_dl.dataset.shuffleSamples() #Makes sure our batches included different combination of the images every batch and epoch
            train_loss_tot = 0.0
            val_loss_tot = 0.0
            avg_loss = 0

            for batch_i, (x_batch, gt_batch) in enumerate(train_dl):
                x_batch = x_batch.to(self.device)
                gt_batch = gt_batch.to(self.device)
                self.optimizer.zero_grad() #we have to null out the grad from the previous step so that it dosent accumulate
                train_pred = self.model(x_batch)
                train_loss = self.getBatchLoss(train_pred,gt_batch) #vil ha en variabel som "Recombines the loss per sample into a single value"
                train_loss.backward()
                self.optimizer.step()
                avg_loss += train_loss.cpu().detach().item()
                train_loss_tot = train_loss.mean()
                
                global_step += batch_size
                tracked_train_loss[global_step] = train_loss_tot

                if epoch_index == epochs and batch_i <=5:
                    self.saveImages(x_batch,gt_batch,train_pred,batch_i,"train")

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
                        val_loss_avg += val_loss.cpu().item()
                        val_loss_tot = val_loss.mean()
                        total_steps += 1
                        tot_cor += (val_pred == gt_batch).sum().item()
                        tot_im += val_pred.shape[0] * val_pred.shape[1]

                        if epoch_index == epochs:
                            self.saveImages(x_batch,gt_batch,val_pred,batch_i,"val")

                    val_loss_avg = val_loss_avg / total_steps
                    accuracy = tot_cor / tot_im
                    tracked_val_loss[global_step] = val_loss_avg

                    print("Epoch {}/{} with Training Loss: {} and Validation Loss {}".format(epoch_index,epochs,tracked_train_loss[global_step],tracked_val_loss[global_step]))
                    self.getAccuracy(train_dl,val_dl) #Gets the training and validation accuracy
        self.plotLoss(tracked_train_loss,tracked_val_loss) #Plots the loss for the entire training loop
    
    def getStats(self,pred,gt):
        pred = torch.round(pred)
        length = (gt.shape[0] * gt.shape[1])
        all_pos = int((gt==1).sum())
        all_neg = int((gt==0).sum())

        correct = (pred == gt).float().sum()
        acc = float(correct/length)
        true_pos = int(((pred==1) & (gt==1)).sum())
        false_pos = int(((pred==1) & (gt==0)).sum())
        true_neg = int(((pred==0) & (gt==0)).sum())
        false_neg = int(((pred==0) & (gt==1)).sum())

        dice = (2*true_pos) / (2*true_pos + false_neg + false_pos)

        print("Train accuracy:",acc)
        #print("true_pos {} / {} and false_pos {}".format(true_pos,all_pos,false_pos))
        #print("true_neg {} / {} and false_neg {}".format(true_neg,all_neg,false_neg))
        #print("DICE", dice)
       
    def saveImages(self, X_batch, Y_batch, outputs, batch, name):
        fig,ax = plt.subplots(1,3)
        ax[0].set_axis_off()
        ax[0].set_title("input image")
        ax[0].imshow(X_batch[0].detach().cpu().permute([1,2,0]))
        ax[1].set_axis_off()
        ax[1].set_title("ground truth")
        ax[1].imshow(Y_batch[0].detach().cpu().reshape(500,500), cmap='gray')
        ax[2].set_axis_off()
        ax[2].set_title("model pred") 
        ax[2].imshow(outputs[0].detach().cpu().reshape(500,500), cmap='gray') # class 1: laser pred
        fig.savefig(os.path.join(save_dir, name+"_firstcnnbatch_"+format(batch, "02d")+".png"))


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
        fig.savefig("results/loss.png")

    def getAccuracy(self,train_dl, val_dl):
        for name, loader in [("train", train_dl), ("val", val_dl)]:
            correct = 0
            total = 0

            with torch.no_grad(): #Dont want to update parameters
                for imgs, labels in loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(imgs)
                    total += labels.shape[0]*labels.shape[1]
                    correct += int((outputs == labels).sum())

            print("Accuracy {}: {}".format(name , correct / total))

if __name__ == '__main__':
    TrainingLoop().main() #instantiates the application object and invokes the main method
  
