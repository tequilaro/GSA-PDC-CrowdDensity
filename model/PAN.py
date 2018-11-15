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
        
class SA_MCNN(nn.Module):
    def __init__(self, bn=False):
        super(SA_MCNN, self).__init__()
        
        self.pool = nn.MaxPool2d(2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.fusion_se = nn.Sequential(                
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 512),
                nn.Sigmoid()
        )
        
        self.block1 = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True)
                )
        
        self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(inplace=True)
                )
        
        self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(inplace=True)
                )
                
        self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True)
                )
                
        self.dilated_conv1 = nn.Conv2d(256, 512, 3, stride=2, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(128, 256, 3, stride=4, padding=1, dilation=2)
        self.dilated_conv3 = nn.Conv2d(64, 128, 3, stride=8, padding=1, dilation=4)
        
        self.conv1 = nn.Conv2d(256, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(1024, 256, 3, 1, 1) 
        self.conv4 = nn.Conv2d(64, 1, 1, 1, 0)
        
        self.gsa1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, 128, 1, 1, 0),
                # nn.BatchNorm2d(128, eps=0.001, momentum=0, affine=True),
                nn.ReLU(inplace=True)
                )
        
        self.gsa2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, 256, 1, 1, 0),
                # nn.BatchNorm2d(256, eps=0.001, momentum=0, affine=True),
                nn.ReLU(inplace=True)
                )
        
        self.gsa3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(512, 512, 1, 1, 0),
                # nn.BatchNorm2d(512, eps=0.001, momentum=0, affine=True),
                nn.ReLU(inplace=True)
                )
                
    def forward(self, x):
        
        x1 = self.block1(x)
        # print(x1.shape)
        y1 = self.relu(self.dilated_conv3(x1))
        # print(y1.shape)
        
        x2 = self.block2(self.pool(x1))
        # print(x2.shape)
        y2 = self.relu(self.dilated_conv2(x2))
        # print(y2.shape)
        
        x3 = self.block3(self.pool(x2))
        # print(x3.shape)
        y3 = self.relu(self.dilated_conv1(x3))
        # print(y3.shape)
        
        x4 = self.block4(self.pool(x3))
        # print(x4.shape)
        
        a, b, _, _ = x4.size()
        se = self.avg_pool(x4).view(a, b)
        se = self.fusion_se(se).view(a, b, 1, 1)
        x4 = x4 * se
        # print(x4.shape)
        
        z1 = self.gsa3(x4)
        z1 = y3 * z1
        # print(x4.shape)
        # print(z1.shape)
        z1 = torch.cat((x4,z1), 1)
        z1 = self.relu(self.conv3(z1))
        
        z2 = self.gsa2(z1)
        z2 = y2 * z2
        z2 = torch.cat((z1,z2), 1)
        z2 = self.relu(self.conv2(z2))
        
        z3 = self.gsa1(z2)
        z3 = y1 * z3
        z3 = torch.cat((z2,z3), 1)
        z3 = self.relu(self.conv1(z3))
        
        output = self.relu(self.conv4(z3))
        
                
        return output
        
def get_SAMCNN():
    model = SA_MCNN()
    # weights_normal_init2(model, dev_weight=0.01, dev_bais=0)
    weights_normal_init(model, dev=0.01)
    
    return model