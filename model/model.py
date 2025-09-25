import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional








class SudokuNet(nn.Module):
    def __init__(self):
        super(SudokuNet,self).__init__()
        self.conv_layer_1 = nn.Sequential(
            #input should be (n,1,9,9)
            nn.Conv2d(in_channels=1 , out_channels=32 , kernel_size=3 ,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #after layer size is (9,9,32)

            nn.Conv2d(in_channels = 32 , out_channels=64,kernel_size = 3 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #after layer size is (9,9,64)

            nn.Conv2d(in_channels = 64 , out_channels=128,kernel_size = 3 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #after layer size is (9,9,128)
            nn.Conv2d(in_channels = 128 , out_channels=256,kernel_size = 3 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #after layer size is (9,9,256)
        )
        self.classification_layer = nn.Sequential(
            nn.Conv2d(in_channels=256 , out_channels=9 , kernel_size=1 ,padding=0)
            
        )
    def forward(self,x):
        x = self.conv_layer_1(x)
        x = self.classification_layer(x)
        return x