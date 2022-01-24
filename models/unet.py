# This U-Net model is based on the model made by Aladdin Persson in his tutorail about Umage Segmentation with U-Net
# Tutoral: https://www.youtube.com/watch?v=IHq1t7NxS8k
# U-Net mode: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

import torch
from torch._C import Size
import torch.nn as nn 
import torchvision.transforms.functional as TF 

# The double conv inbetween on the same level in the U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential( 
            # in-channels to out-channels
            nn.Conv2d(
                in_channels, 
                out_channels, 
                3, # kernal size = 3
                1, # stride = 1
                1, # padding = 2
                bias=False), # same-convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # out-channels to out-channels
            nn.Conv2d(
                out_channels, 
                out_channels, 
                3, # kernal size = 3
                1, # stride = 1
                1, # padding = 2 
                bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]): 
        # features as in number of channels on each level
        super(UNET, self).__init__()
        
        # initializing lists
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # will floor division 161x161, output: 160x160


        # Down part of UNET
        for feature in features:
            # map from the previous layer to the next, and exercise all the conv layers
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET 
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) # skip connections 
            self.ups.append(DoubleConv(feature*2, feature))

        # the bottom of the U
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) 

        # final step
        self.finalConv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # doing a forward step
    def forward(self, x):
        skipConnections = []

        # all steps down to the bottom of the U-Net
        for down in self.downs:
            x = down(x)
            skipConnections.append(x)
            x = self.maxPool(x)

        # the bottom step
        x = self.bottleneck(x) 
        skipConnections = skipConnections[::-1] # reversing skipConnections

        # up sampling, getting the skip connection then concatnationg 
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skipCon = skipConnections[idx//2]

            if x.shape != skipCon.shape:
                x = TF.resize(x, size=skipCon.shape[2:]) # resize to the hight and width of skipCon

            concatSkip = torch.cat((skipCon, x), dim=1)
            x = self.ups[idx+1](concatSkip)

        return self.finalConv(x)

# test
def test():
    test = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(test)
    print(preds.shape)
    print(test.shape)
    assert preds.shape == test.shape, "preds did not match test with preds shape: {} and test.shape {}".format(preds.shape,test.shape)
    print("Finished without errors")

if __name__ == "__main__":
    test() # want the same shape, 