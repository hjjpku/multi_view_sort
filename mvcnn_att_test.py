import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from torch.autograd import Variable

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
#from models.MVCNN_attention import SVCNN, MVCNN_attention, MVCNN_self_attention, MVCNN_attention_fc

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



def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':

    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.random.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)

    #path='/home/yw/Desktop/mvcnn_pytorch-master_ECCV2018/mvcnn_self_attn_1and2stage_stage_1and2/'
    path = '/home/yw/Desktop/mvcnn_pytorch-master_ECCV2018/mvcnn_run2_stage_2/'
    modelfile='model-00020.pth'

    args = parser.parse_args()

    pretraining = not args.no_pretraining


    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)


    cnet_2 = MVCNN('mvcnn_run2', cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet
    cnet_2.cuda()

    cnet_2.load(path,modelfile)
    cnet_2.eval()

    n_models_train = args.num_models * args.num_views
    log_dir=None

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views,test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    optimizer=None
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
    loss, val_overall_acc, val_mean_class_acc = trainer.update_validation_accuracy(None)


    # total_seen = 0
    # total_correct = 0
    # cnet_2.eval()
    # for _, data in enumerate(val_loader,0):
    #
    #     N, V, C, H, W = data[1].size()
    #     total_seen = total_seen + N
    #     in_data = Variable(data[1]).view(-1, C, H, W).cuda()
    #     target = Variable(data[0]).cuda()
    #
    #     out_data = cnet_2(in_data)
    #     pred = torch.max(out_data, 1)[1]
    #     correct = (pred==target).float().sum()
    #     total_correct = total_correct+correct
    #
    # print('eval accuracy: ', total_correct.item() / float(total_seen))
    # print('total_correct: ',total_correct.item())
    # print('total seen: ', total_seen)




