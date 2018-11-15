# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
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
import cv2
from model.PAN4 import get_SAMCNN
from data_process.dataset import get_train_path
from data_process.test_dataset import get_test_path

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

def val_model(epoch, density_PATH_NAME, test_dataset, test_loader):
    model.eval()
    
    test_MAE = 0.
    test_MSE = 0.
       
    with torch.no_grad():       
        
        for test_batch_idx, test_data in enumerate(test_loader):
                       
            test_original_input = test_data[0]
            test_img_input = test_data[1]
            test_density_input = test_data[2]
            
            test_img_input = test_img_input.cuda()
            test_density_input = test_density_input.cuda() 

            test_outputs = model(test_img_input) 
            # print(outputs.shape)
            
            test_temp_mae = float(torch.abs(torch.sum(test_outputs)-torch.sum(test_density_input)))
            test_temp_mse = test_temp_mae * test_temp_mae
                       
            test_MAE += test_temp_mae
            test_MSE += test_temp_mse
            
            # test_img_input2 = test_original_input[0]
            # test_density_input2 = test_density_input[0].cpu().detach().squeeze(0).numpy()
            # test_den_predict2 = test_outputs[0].cpu().detach().squeeze(0).numpy()
            
            # plt.subplot(131)
            # plt.axis('off')              
            # plt.imshow(test_img_input2)

            # plt.subplot(132)
            # plt.title(str(np.sum(test_density_input2)))
            # plt.axis('off')
            # test_density_input2 = 255 * ((test_density_input2 -np.min(test_density_input2)) // (np.max(test_density_input2)-np.min(test_density_input2)))
            # test_density_input2 = test_density_input2.astype(np.uint8)
            # test_density_input2 = cv2.applyColorMap(test_density_input2, cv2.COLORMAP_JET)
            # test_density_input2 = cv2.cvtColor(test_density_input2,cv2.COLOR_BGR2RGB)
            # plt.imshow(test_density_input2)

            # plt.subplot(133)
            # plt.title(str(np.sum(test_den_predict2)))
            # plt.axis('off')                
            # test_den_predict2 = 255 * ((test_den_predict2 -np.min(test_den_predict2)) // (np.max(test_den_predict2)-np.min(test_den_predict2)))
            # test_den_predict2 = test_den_predict2.astype(np.uint8)
            # test_den_predict2 = cv2.applyColorMap(test_den_predict2, cv2.COLORMAP_JET)
            # test_den_predict2 = cv2.cvtColor(test_den_predict2,cv2.COLOR_BGR2RGB)
            # plt.imshow(test_den_predict2)
            
            # savefig(density_PATH_NAME + str(epoch+1) + '_' + str(test_batch_idx) + '.jpg')
            # plt.close('all')   
            # print ('test_mae: {:.3f}\ttest_mse: {:.3f}'.format(test_temp_mae, test_temp_mse))
    
    print('test_MAE: {:.3f}\ttest_MSE: {:.3f}'.format(test_MAE/len(test_dataset), math.sqrt(test_MSE/len(test_dataset))))

if __name__=='__main__':
      
    density_PATH_NAME = '/home/computer/lcy/pytorch/MyProject/SACrowd/result/test2/'
    shutil.rmtree(density_PATH_NAME)
    os.makedirs(density_PATH_NAME)
    
    test_density_PATH_NAME = '/home/computer/lcy/pytorch/MyProject/SACrowd/result/test3/'
    shutil.rmtree(test_density_PATH_NAME)
    os.makedirs(test_density_PATH_NAME)
    
    original_img, imgs_all, density_path = get_train_path()
    test_original_img, test_imgs_all, test_density_path = get_test_path()
    
    # original_img = torch.from_numpy(original_img)
    # imgs_all = torch.from_numpy(imgs_all)
    # density_path = torch.from_numpy(density_path)
     
    Batchsize = 1
    EPOCH = 1000                                      
    LEARNING_RATE = 1e-7

    train_dataset = MyDataset(original_img, imgs_all, density_path)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Batchsize, shuffle = True)
    
    test_dataset = MyDataset(test_original_img, test_imgs_all, test_density_path)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)
   
    model = get_SAMCNN()
    model.load_state_dict(torch.load('./saved_model/3/565_cnn.pkl'))
    model.cuda()
    # summary(model,(1,256,256))
    
   
    cost = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.005)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    mae = 0.
    mse = 0.
    
    for epoch in range(EPOCH):
        
        model.train()
        
        MAE = 0.
        MSE = 0.
        
        for batch_idx, data in enumerate(train_loader):
                       
            original_input = data[0]
            img_input = data[1]
            density_input = data[2]
            
            # print(original_input.shape)
            # print(img_input.shape)
            # print(density_input.shape)
            
            img_input = img_input.cuda()
            density_input = density_input.cuda() 

            optimizer.zero_grad()
            outputs = model(img_input) 
            # print(outputs.shape)

            loss = cost(outputs, density_input) + (torch.abs(torch.sum(outputs)-torch.sum(density_input)))
            
            temp_mae = float(torch.abs(torch.sum(outputs)-torch.sum(density_input)))
            temp_mse = temp_mae * temp_mae
            
            mae += temp_mae
            mse += temp_mse
            
            MAE += temp_mae
            MSE += temp_mse
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                
                img_input2 = original_input[0]
                density_input2 = density_input[0].cpu().detach().squeeze(0).numpy()
                den_predict2 = outputs[0].cpu().detach().squeeze(0).numpy()
                
                # print(np.sum(density_input2), np.sum(den_predict2), np.abs(np.sum(density_input2)-np.sum(den_predict2)))
                
                plt.subplot(131)
                plt.axis('off')              
                plt.imshow(img_input2)

                plt.subplot(132)
                plt.title(str(np.sum(density_input2)))
                plt.axis('off')
                density_input2 = 255 * (density_input2 -np.min(density_input2)) // (np.max(density_input2)-np.min(density_input2))
                density_input2 = density_input2.astype(np.uint8)
                density_input2 = cv2.applyColorMap(density_input2, cv2.COLORMAP_JET)
                density_input2 = cv2.cvtColor(density_input2,cv2.COLOR_BGR2RGB)
                plt.imshow(density_input2)

                plt.subplot(133)
                plt.title(str(np.sum(den_predict2)))
                plt.axis('off')                
                den_predict2 = 255 * (den_predict2 -np.min(den_predict2)) // (np.max(den_predict2)-np.min(den_predict2))
                den_predict2 = den_predict2.astype(np.uint8)
                den_predict2 = cv2.applyColorMap(den_predict2, cv2.COLORMAP_JET)
                den_predict2 = cv2.cvtColor(den_predict2,cv2.COLOR_BGR2RGB)
                plt.imshow(den_predict2)

                # plt.show()
                savefig(density_PATH_NAME+'%d_epoch_%d_batch.png' % (epoch+1, batch_idx+1))
                plt.close('all')
                              
                # print ('Epoch [{}/{}]\tbatch [{}]\tmae: {:.3f}\tmse: {:.3f}'.format(epoch+1, EPOCH, batch_idx+1, mae/5, math.sqrt(temp_mse/5)))
                # print('--------------------------------------------------------------------------------------------------------')
                mae = 0.
                mse = 0.
                
        torch.save(model.state_dict(), './saved_model/3/' + str(epoch+1) + '_cnn.pkl')              
        
        print('########################################################################################################')
        # print('########################################################################################################')
        print ('Epoch [{}/{}]\ntrain_MAE: {:.3f}\ttrain_MSE: {:.3f}'.format(epoch+1, EPOCH, MAE/len(train_dataset), math.sqrt(MSE/len(train_dataset))))
        val_model(epoch, test_density_PATH_NAME, test_dataset, test_loader)
        # print('########################################################################################################')
        # print('########################################################################################################')