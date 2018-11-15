# -*- coding:utf-8 -*-

import torchvision.models as models
import shutil
import random
import scipy.io as sio
from PIL import Image
import os
import numpy as np
# np.set_printoptions(threshold='nan')
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchsummary import summary
import math
import os
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import cv2

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

def weights_normal_init2(model, dev_weight=0.01, dev_bais=0):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                nn.init.normal_(m.weight.data, 0.0, dev_weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, dev_bais)             


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class SA_MCNN(nn.Module):
    def __init__(self, bn=False):
        super(SA_MCNN, self).__init__()
        
        self.pool = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fusion_se = nn.Sequential(
                nn.Linear(30, 15),
                nn.ReLU(inplace=True),
                nn.Linear(15, 30),
                nn.Sigmoid()
        )
        
        self.conv1_se = Conv2d(16, 16, 1, same_padding=True, bn=bn)
        self.conv2_se = Conv2d(20, 20, 1, same_padding=True, bn=bn)
        self.conv3_se = Conv2d(24, 24, 1, same_padding=True, bn=bn)

        self.branch1_conv1 = Conv2d(1, 16, 9, same_padding=True, bn=bn)
        self.branch1_conv2 = Conv2d(16, 32, 7, same_padding=True, bn=bn)
        self.branch1_conv3 = Conv2d(32, 16, 7, same_padding=True, bn=bn)
        self.branch1_conv4 = Conv2d(32,  8, 7, same_padding=True, bn=bn)
        
        self.branch2_conv1 = Conv2d(1, 20, 7, same_padding=True, bn=bn)
        self.branch2_conv2 = Conv2d(20, 40, 5, same_padding=True, bn=bn)
        self.branch2_conv3 = Conv2d(40, 20, 5, same_padding=True, bn=bn)
        self.branch2_conv4 = Conv2d(40, 10, 5, same_padding=True, bn=bn)
        
        self.branch3_conv1 = Conv2d(1, 24, 5, same_padding=True, bn=bn)
        self.branch3_conv2 = Conv2d(24, 48, 3, same_padding=True, bn=bn)
        self.branch3_conv3 = Conv2d(48, 24, 3, same_padding=True, bn=bn)
        self.branch3_conv4 = Conv2d(48, 12, 3, same_padding=True, bn=bn)
        
        self.fuse = Conv2d(30, 1, 1, same_padding=True, bn=bn)
       
        
        # self.conv1 = Conv2d(8, 1, 1, same_padding=True, bn=bn)
        # self.conv2 = Conv2d(10, 1, 1, same_padding=True, bn=bn)
        # self.conv3 = Conv2d(12, 1, 1, same_padding=True, bn=bn)
        
    def forward(self, x):
        x1 = self.branch1_conv1(x)       
        x2 = self.branch1_conv2(x1)       
        x2 = self.pool(x2)
        x2 = self.branch1_conv3(x2)
        x3 = self.pool(x2)
        x4 = self.avg_pool(x3)
        x4 = self.conv1_se(x4)
        x5 = x1 * x4
        x5 = self.pool2(x5)
        x6 = torch.cat((x3,x5),1)
        x6 = self.branch1_conv4(x6)
        # print(x6.shape)
        
        y1 = self.branch2_conv1(x)       
        y2 = self.branch2_conv2(y1)       
        y2 = self.pool(y2)
        y2 = self.branch2_conv3(y2)
        y3 = self.pool(y2)
        y4 = self.avg_pool(y3)
        y4 = self.conv2_se(y4)
        y5 = y1 * y4
        y5 = self.pool2(y5)
        y6 = torch.cat((y3,y5),1)
        y6 = self.branch2_conv4(y6)
        # print(y6.shape)
        
        z1 = self.branch3_conv1(x)       
        z2 = self.branch3_conv2(z1)       
        z2 = self.pool(z2)
        z2 = self.branch3_conv3(z2)
        z3 = self.pool(z2)
        z4 = self.avg_pool(z3)
        z4 = self.conv3_se(z4)
        z5 = z1 * z4
        z5 = self.pool2(z5)
        z6 = torch.cat((z3,z5),1)
        z6 = self.branch3_conv4(z6)
        # print(z6.shape)
        
        output = torch.cat((x6,y6,z6),1)
        # print(output.shape)
        
        a, b, _, _ = output.size()
        se = self.avg_pool(output).view(a, b)
        se = self.fusion_se(se).view(a, b, 1, 1)
        output = output * se
        
        output = self.fuse(output)
        # output = F.interpolate(output, scale_factor=4, mode='nearest')
                
        return output
        
def get_SAMCNN():
    model = SA_MCNN()
    # weights_normal_init2(model, dev_weight=0.01, dev_bais=0)
    weights_normal_init(model, dev=0.01)
    
    return model