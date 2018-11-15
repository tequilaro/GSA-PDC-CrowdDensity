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
from model.SAMCNN import get_SAMCNN
from data_process.dataset import get_path

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
      
    density_PATH_NAME = '/home/computer/lcy/pytorch/MyProject/SACrowd/result/predict_density/'
    shutil.rmtree(density_PATH_NAME)
    os.makedirs(density_PATH_NAME)
    
    original_img, imgs_all, density_path = get_path()
    
    # original_img = torch.from_numpy(original_img)
    # imgs_all = torch.from_numpy(imgs_all)
    # density_path = torch.from_numpy(density_path)
     
    Batchsize = 1
    EPOCH = 200
    LEARNING_RATE = 1e-5

    train_data = MyDataset(original_img, imgs_all, density_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=Batchsize, shuffle=True)
   
    model = get_SAMCNN()
    # model.load_state_dict(torch.load('/home/computer/lcy/pytorch/MyProject/switch_net_data_process/model/22_49_cnn.pkl'))
    model.cuda()
    # summary(model,(1,256,256))
    
    model.train()

    cost = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.005)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    for epoch in range(EPOCH):

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

            loss = cost(outputs, density_input) #+ 0.1 * torch.abs(torch.sum(outputs)-torch.sum(density_input))
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:

                img_input2 = original_input[0]
                density_input2 = density_input[0].cpu().detach().squeeze(0).numpy()
                den_predict2 = outputs[0].cpu().detach().squeeze(0).numpy()
                
                print(np.sum(density_input2), np.sum(den_predict2), np.abs(np.sum(density_input2)-np.sum(den_predict2)))
                
                plt.subplot(131)
                plt.axis('off')              
                plt.imshow(img_input2)

                plt.subplot(132)
                plt.title(str(np.sum(density_input2)))
                plt.axis('off')
                plt.imshow(density_input2)

                plt.subplot(133)
                plt.title(str(np.sum(den_predict2)))
                plt.axis('off')
                # den_predict2 = 255 * den_predict2 / np.max(den_predict2)
                plt.imshow(den_predict2)

                plt.show()
                savefig(density_PATH_NAME+'%d_epoch_%d_batch.png' % (epoch+1, batch_idx+1))
                plt.close('all')
                
                
            print ('Epoch [{}/{}]\tbatch [{}]\tLoss: {:.6f}'.format(epoch+1, EPOCH, batch_idx+1, 1000000 * loss.item()))
            torch.save(model.state_dict(), './saved_model/1/' + str(epoch+1)+'_' + str(batch_idx+1) + '_cnn.pkl')
            
            print('--------------------------------------------------------------------------------------------------------')
