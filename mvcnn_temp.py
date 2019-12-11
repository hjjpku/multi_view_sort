import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
#from models.MVCNN import MVCNN, SVCNN
from torch.autograd import Variable
from models.MVCNN_attention import SVCNN, MVCNN_attention, MVCNN_self_attention, MVCNN_attention_fc, MVCNN_attention_fc_sort

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.set_defaults(train=False)

#os.environ['CUDA_VISIBLE_DEVICES']='1'

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = not args.no_pretraining

    x = Variable(torch.randn(8 * 12, 3, 224, 224).cuda(), requires_grad=True)
    print(x.shape)

    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
    cnet = cnet.cuda()
    cnet2 = MVCNN_attention_fc_sort(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    cnet2 = cnet2.cuda()

    out = cnet2(x)
    print(out.shape)