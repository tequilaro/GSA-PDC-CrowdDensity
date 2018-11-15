# -*- coding:utf-8 -*-

from PIL import Image 
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import scipy.io
from skimage.transform import downscale_local_mean
import os
import sys
import json
import math
import time
import random
from random import shuffle
import pickle
import h5py
import glob
import shutil 
import scipy


def load_gt_from_json(gt_file, gt_shape):
    print(gt_shape)
    gt = np.zeros(gt_shape, dtype='uint8') 
    with open(gt_file, 'r') as jf:
        for j, dot in enumerate(json.load(jf)):
            try:
                gt[int(math.floor(np.array(dot['y'], dtype=np.float32))), int(math.floor(np.array(dot['x'], dtype=np.float32)))] = 1
            except IndexError:
                print(gt_file, np.array(dot['y'], dtype=np.float32), np.array(dot['x'], dtype=np.float32), sys.exc_info())   
    return gt

    
def gaussian_filter_density(gt):
    densities = []
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print(gt_count)
    if gt_count == 0:
        return density
        
    pts = []
    for each in zip(np.nonzero(gt)[1], np.nonzero(gt)[0]):
        pts.append(np.array(each))
    pts = np.array(pts)
  
    # print(pts)
    leafsize = 2048
    # build kdtree
    print('build kdtree...')
    tree = scipy.spatial.KDTree(pts.copy(), leafsize)
    # query kdtree
    print('query kdtree...') 
    distances, locations = tree.query(pts, k=2, eps=10.)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = distances[i][1]#the nearest neighbor
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            sigma *= 0.1
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    densities.append(density)
    return densities    
 
 
def load_images_and_gts(path):
    images = []
    gts = []
    densities = []
    jsons = glob.glob(os.path.join(path, '*.json'))
    
    jsons = sorted(jsons, key = lambda x: int(x[len(path)+len('IMG_'):][:-5]))  # shanghaiTech
    # jsons = sorted(jsons, key = lambda x: int(x[len(path):][:-5]))  #UCF_CC_50

    for gt_file in jsons:
        print(gt_file)
        if os.path.isfile(gt_file.replace('.json','.jpg')):
            #img = cv2.imread(gt_file.replace('.json','.jpg'))
            img = Image.open(gt_file.replace('.json','.jpg'))
        else:
            #img = cv2.imread(gt_file.replace('.json','.png'))
            img = Image.open(gt_file.replace('.json','.png'))
        images.append(img)
        
        #load ground truth
 
        gt = load_gt_from_json(gt_file, [img.size[1], img.size[0]])
       
        gts.append(gt)
        #densities
        desnity_file = gt_file.replace('.json','.h5')
        if os.path.isfile(desnity_file):
            #load density if exist
            with h5py.File(desnity_file, 'r') as hf:
                density = np.array(hf.get('density'))
        else:
            density = gaussian_filter_density(gt)[0]
            with h5py.File(desnity_file, 'w') as hf:
                hf['density'] = density
        densities.append(density)
        
        #-------------------------
        # plt.subplot(131)  
        # plt.imshow(img)
        # plt.subplot(132)  
        # plt.imshow(density)
        # plt.subplot(133)
        # plt.imshow(gaussian_filter_density(gt)[0])
        # plt.show()
        # savefig('/home/computer/data/shanghaitech/part_A_final/train_data/images/aaa_label.jpg')
        # plt.close('all')
        #-------------------------
        
        # break
        
    print(path, len(images), 'loaded')
    
  
if __name__=='__main__':

    rootpath = '/home/computer/lcy/pytorch/dataset/ShanghaiTech_Dataset/A2/test_data/images/'
    load_images_and_gts(rootpath)

