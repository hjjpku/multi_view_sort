import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

kmean_dir = '/mnt/tmp1/yw/MVCNN_features/car/cluster_fea/car_0100.npz'
data = np.load(kmean_dir)
labels_cluster = data['labels']
fea = data['center_feas']
fea_idx = data['idx']
_,idx = np.where(fea_idx==0)
idx = np.expand_dims(idx,1)     # 3x1
feature_12_dir = '/mnt/tmp1/yw/MVCNN_features/car/features/car_0100.npy'
feature_12 = np.load(feature_12_dir)



img_dir = '/mnt/tmp1/yw/MVCNN_features/car/img/car_0100.npy'
img_12 = np.load(img_dir)


# process img
std=torch.tensor([0.229, 0.224, 0.225])
std = std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
std = std.expand(12,224,224,3).numpy()

mean = torch.tensor([0.485, 0.456, 0.406])
mean = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
mean = mean.expand(12,224,224,3).numpy()

# x_new = x.view(x.shape[0]//12,12,x.shape[1],x.shape[2],x.shape[3])
x_new = np.expand_dims(img_12,axis=0)
x_0 = x_new[0]
x_0 = x_0.transpose(0,2,3,1)
x_0_temp = x_0*std+mean
x_0_temp = x_0_temp*255
x_0_temp = np.round(x_0_temp)
x_0_temp = np.uint8(x_0_temp)

label_0_idx = labels_cluster==0
label_1_idx = labels_cluster==1
label_2_idx = labels_cluster==2

group_0 = x_0_temp[label_0_idx]
group_1 = x_0_temp[label_1_idx]
group_2 = x_0_temp[label_2_idx]

from PIL import Image
# plot different cluster
for i in range(group_0.shape[0]):
    im = Image.fromarray(group_0[i])
    im.show()

for i in range(group_1.shape[0]):
    im = Image.fromarray(group_1[i])
    im.show()

for i in range(group_2.shape[0]):
    im = Image.fromarray(group_2[i])
    im.show()

print(1)