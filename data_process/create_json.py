# -*- coding:utf-8 -*-

import math
import time
import random
from random import shuffle
import pickle
import h5py
import glob
import shutil 
import os
import scipy.io as sio
import json
import numpy as np

def load_mat(gt_path):
    dataset = sio.loadmat(gt_path)
    dataset = dataset['image_info'][0,0]
    location = dataset['location'][0,0]

    return location

def create_json(name, location):
    i = name.index('IMG')
    j = name.index('.mat')
    json_str = name[i:j]
    json_name = json_str + ".json"
    print(json_name)
    
    with open(os.path.join(mat_path,json_name),'w') as js:
        temp_data = []
        for i in range(temp_mat.shape[0]):            
            temp_mat_row = np.asarray(temp_mat[i], dtype=np.float32)          
            data_x = str(temp_mat_row[0])
            data_y = str(temp_mat_row[1])
            data = {'x':data_x, 'y':data_y}
            temp_data.append(data)
            
        model=json.dumps(temp_data)
        js.write(model) 
    js.close()


if __name__=='__main__':

    mat_path = '/home/computer/lcy/pytorch/dataset/ShanghaiTech_Dataset/A2/test_data/images/'
    filename = os.listdir(mat_path)
       
    for name in filename:
        if(name.find('.h5') >= 0):
            continue
        elif(name.find('.json') >= 0):
            continue
        elif(name.find('.jpg') >= 0):
            continue
        else:
            print(name)
            temp_mat = load_mat(os.path.join(mat_path,name))
            create_json(name, temp_mat)
    