import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchsummary import summary
import matplotlib.pyplot as plt

class BottleNeck(nn.Module):
    '''
    Bottleneck modules (with skip connnections)
    '''

    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        '''
        Parameter initialization.
        '''
        super(BottleNeck, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        self.identity_connection = nn.Sequential()
        if stride != 1 or in_channels != expansion*out_channels:
            self.identity_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*expansion)
            )


    def forward(self, x):
        '''
        Forward Propagation.
        '''

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.identity_connection(x) #identity connection/skip connection
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    '''
    ResNet-50 Architecture.
    '''

    def __init__(self, image_depth, num_classes):
        '''
        Params init and build arch.
        '''
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.expansion = 4
        self.num_blocks = [3, 4, 6, 3]

        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=7, stride=2, in_channels=image_depth, out_channels=self.in_channels, padding=3, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(stride=2, kernel_size=3, padding=1))

        self.layer1 = self.make_layer(out_channels=64, num_blocks=self.num_blocks[0], stride=1)
        self.layer2 = self.make_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2)
        self.layer3 = self.make_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=2)
        self.layer4 = self.make_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.linear = nn.Linear(512*self.expansion, num_classes)


    def make_layer(self, out_channels, num_blocks, stride):
        '''
        To construct the bottleneck layers.
        '''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion))
            self.in_channels = out_channels * self.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        '''
        Forward propagation of ResNet50.
        '''

        x = self.conv_block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv = self.layer4(x)
        # x = self.avgpool(x_conv)
        x = nn.Flatten()(x_conv) 
        x = self.linear(x)

        return x
