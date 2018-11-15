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
from model.PAN3 import get_SAMCNN
from data_process.test_dataset import get_path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device_ids = [0,1,2,3]
        
class MyDataset(data.Dataset):
    def __init__(self, original_img_path, train_img_path, train_density_path):
        self.orimg = original_img_path
        self.imgs = train_img_path
        self.density_path = train_density_path
    def __getitem__(self, index):
        return self.orimg[index], self.imgs[index], self.density_path[index]
    def __len__(self):
        return len(self.imgs)


if __name__=='__main__':
      
    density_PATH_NAME = '/home/computer/lcy/pytorch/MyProject/SACrowd/result/shanghaiPartA_test/'
    shutil.rmtree(density_PATH_NAME)
    os.makedirs(density_PATH_NAME)
    
    original_img, imgs_all, density_path = get_path()
    
    # original_img = torch.from_numpy(original_img)
    # imgs_all = torch.from_numpy(imgs_all)
    # density_path = torch.from_numpy(density_path)
     
    Batchsize = 1

    test_data = MyDataset(original_img, imgs_all, density_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=Batchsize, shuffle=False)
   
    model = get_SAMCNN()
    model.load_state_dict(torch.load('./saved_model/1/175_1_cnn.pkl'))
    model.cuda()
    # summary(model,(1,256,256))
    
    model.eval()
    
    with torch.no_grad():
        
        MAE = 0.
        MSE = 0.
        
        for batch_idx, data in enumerate(test_loader):
                       
            original_input = data[0]
            img_input = data[1]
            density_input = data[2]
            
            # print(original_input.shape)
            # print(img_input.shape)
            # print(density_input.shape)
            
            img_input = img_input.cuda()
            density_input = density_input.cuda() 

            outputs = model(img_input) 
            # print(outputs.shape)
            
            temp_mae = float(torch.abs(torch.sum(outputs)-torch.sum(density_input)))
            temp_mse = temp_mae * temp_mae
                       
            MAE += temp_mae
            MSE += temp_mse
            
            img_input2 = original_input[0]
            density_input2 = density_input[0].cpu().detach().squeeze(0).numpy()
            den_predict2 = outputs[0].cpu().detach().squeeze(0).numpy()
            
            plt.subplot(131)
            plt.axis('off')              
            plt.imshow(img_input2)

            plt.subplot(132)
            plt.title(str(np.sum(density_input2)))
            plt.axis('off')
            density_input2 = 255 * ((density_input2 -np.min(density_input2))/ (np.max(density_input2)-np.min(density_input2)))
            density_input2 = density_input2.astype(np.uint8)
            density_input2 = cv2.applyColorMap(density_input2, cv2.COLORMAP_JET)
            density_input2 = cv2.cvtColor(density_input2,cv2.COLOR_BGR2RGB)
            plt.imshow(density_input2)

            plt.subplot(133)
            plt.title(str(np.sum(den_predict2)))
            plt.axis('off')                
            den_predict2 = 255 * ((den_predict2 -np.min(den_predict2))/ (np.max(den_predict2)-np.min(den_predict2)))
            den_predict2 = den_predict2.astype(np.uint8)
            den_predict2 = cv2.applyColorMap(den_predict2, cv2.COLORMAP_JET)
            den_predict2 = cv2.cvtColor(den_predict2,cv2.COLOR_BGR2RGB)
            plt.imshow(den_predict2)

            plt.show()
            savefig(density_PATH_NAME+'%d.png' % (batch_idx+1))
            plt.close('all')   
            
            print(batch_idx+1)
            
        print ('MAE: {:.3f}\tMSE: {:.3f}'.format(MAE/len(test_data), math.sqrt(MSE/len(test_data))))
        
# MAE: 126.943    MSE: 199.174        