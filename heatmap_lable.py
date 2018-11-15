# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import cv2
from PIL import Image
import h5py
import numpy as np

img_path = "/home/computer/lcy/pytorch/dataset/ShanghaiTech_Dataset/A1/test_data/images/IMG_10.jpg"
density_path = "/home/computer/lcy/pytorch/dataset/ShanghaiTech_Dataset/A1/test_data/images/IMG_10.h5"


img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
# label = Image.fromarray(cv2.cvtColor(label,cv2.COLOR_BGR2RGB)) 
# label.save('0.jpg')

label = h5py.File(density_path)['density']
label = np.asarray(label, dtype=np.float32)
label = 255 * (label -np.min(label)) // (np.max(label)-np.min(label))
label = label.astype(np.uint8)
label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
label = cv2.cvtColor(label,cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.title('Original Image')
plt.axis('off')              
plt.imshow(img)

plt.subplot(122)
plt.title('Ground Truth')
plt.axis('off')
plt.imshow(label)

savefig('4.jpg')
plt.close('all') 