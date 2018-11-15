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
from torchsummary import summary
import math
import os
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import cv2


def get_train_path():
    path = "/home/computer/lcy/pytorch/dataset/ShanghaiTech_Dataset/A1/train_data/images/"
    
    filename = os.listdir(path)
    img_path = []
    imgs_all = []
    original_img = []
    for name in filename:
        
        if (name.find('_den') >= 0):  # find .mat file
            continue
        elif (name.find('.h5') >= 0):
            continue
        elif (name.find('.json') >= 0):
            continue
        elif (name.find('.mat') >= 0):
            continue
        else:
            # print(name)
            img1 = Image.open(os.path.join(path, name))
            img_height = img1.size[1]
            img_width = img1.size[0]
            # print(img_width, img_height)
            img_newheight = int((img_height // 8) * 8)
            img_newwidth = int((img_width // 8) * 8)
            # print(img_newwidth, img_newheight)
            
            img1 = img1.resize((img_newwidth, img_newheight))
            # print(img1.size)
            # print('-----------------------------------')
            
            if img1.mode == 'RGB':
                               
                # img11 = img1.resize((256,192))
                img11 = np.asarray(img1)
                original_img.append(img11)   

                # img1 = img1.resize((1024,768))
                img1 = img1.convert('L')
                img1 = np.asarray(img1, dtype=np.float32)
                img1.flags.writeable = True
                img1 = img1 - np.mean(img1)
                # img1 = img1 / np.std(img1)
                img1 = img1[np.newaxis, :]

                imgs_all.append(img1)
                img_path.append(os.path.join(path, name))
            else:
                
                # img11 = img1.resize((256,192))
                # img11 = img11.convert('RGB')    
                img11 = np.asarray(img1)
                original_img.append(img11)
                
                # img1 = img1.resize((1024,768))
                img1 = np.asarray(img1, dtype=np.float32)
                img1.flags.writeable = True
                img1 = img1 - np.mean(img1)
                # img1 = img1 / np.std(img1)
                img1 = img1[np.newaxis, :]

                imgs_all.append(img1)
                img_path.append(os.path.join(path, name))
 

    density_path = []
    for name in img_path:
        
        img1 = Image.open(os.path.join(path, name))
        img_height = img1.size[1]
        img_width = img1.size[0]
        img_height = int((img_height // 8) * 8)
        img_width = int((img_width // 8) * 8)
        # print(img_height, img_width)
          
        #print name
        i = name.index('.jpg')
        substr = name[0:i]
        label_name = substr + ".h5"
        #print label_name
        density_input1 = h5py.File(label_name)['density']
        density_input1 = np.asarray(density_input1, dtype=np.float32)
        # print(density_input1.shape)
        
        density_height = density_input1.shape[0]
        density_width = density_input1.shape[1]
        
        # density_newheight = int(img_height / 8)
        # density_newwidth = int(img_width / 8)
        density_newheight = int(img_height)
        density_newwidth = int(img_width)
        
        width_ratio = density_newwidth / density_width
        height_ratio = density_newheight / density_height
        density_input1 = cv2.resize(density_input1, (density_newwidth, density_newheight), fx = width_ratio, fy = height_ratio, interpolation = cv2.INTER_CUBIC) * (1/(width_ratio*height_ratio))
        
        density_input1 = density_input1[np.newaxis, :]
        density_path.append(density_input1)
        # print('\n')
    
    # original_img = np.asarray(original_img, dtype = np.uint8)
    # print(original_img.shape)

    # imgs_all = np.asarray(imgs_all, dtype = np.float32)
    # print(imgs_all.shape)
    
    # density_path = np.asarray(density_path, dtype = np.float32)
    # print(density_path.shape)
	   
    # f = h5py.File('./data_process/train_dataset.h5','w')
    # f['original_img'] = original_img
    # f['imgs_all'] = imgs_all                
    # f['density_path'] = density_path  
    # f.close()    
    
    return original_img, imgs_all, density_path


def load_pathh5():    
 
    h5_name = './data_process/train_dataset.h5'
    original_img = h5py.File(h5_name)['original_img']
    original_img = np.asarray(original_img)
    imgs_all = h5py.File(h5_name)['imgs_all']
    imgs_all = np.asarray(imgs_all)
    density_path = h5py.File(h5_name)['density_path']
    density_path = np.asarray(density_path)
    
    print(imgs_all.shape)
    print(density_path.shape)
    
    return original_img, imgs_all, density_path
    
