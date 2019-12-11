import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from torchvision import transforms, datasets
import glob
from torch.autograd import Variable

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
from models.Model import Model

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN_kmean")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features/")
parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_val/")
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


class KmeanImgDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.root_dir = root_dir
        self.test_mode = test_mode
        self.num_views = num_views

        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(root_dir+self.classnames[i]+'/features'+'/*.npy'))

            self.filepaths.extend(all_files)

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)))
            filepath_new=[]
            for  i in range(len(rand_idx)):
                filepath_new.extend(self.filepaths[rand_idx[i]:rand_idx[i]+1])
            self.filepaths = filepath_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),

            ])
        else:
            self.transform = transforms.Compose([

                transforms.ToTensor()
            ])

    def __len__(self):
        return int(len(self.filepaths))

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        im = np.load(self.filepaths[idx])
        im = torch.from_numpy(im).squeeze(0)

        return (class_id,im, self.filepaths[idx])




class ThreeView_att(Model):
    def __init__(self,model,nclasses=40):
        super(ThreeView_att,self).__init__('mvcnn_kmeans')
        self.nclasses = nclasses

        self.net_2 = model.net_2
        self.att1 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.att2 = nn.Sequential(nn.Linear(128 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 128),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), )

    def forward(self, x):   # x: B,3,512,7,7
        y = x
        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 3,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 3
            scores[i] = temp2
        scores = nn.functional.softmax(scores, 1)
        y_att = torch.bmm(y.view(y.shape[0], y.shape[1], -1).permute(0, 2, 1), scores.unsqueeze(2))
        y_att = y_att.view(y.shape[0], y.shape[2], y.shape[3], y.shape[4])
        return self.net_2(y_att.view(y.shape[0], -1))


class ThreeView_att_sort(Model):
    def __init__(self,model,nclasses=40):
        super(ThreeView_att_sort,self).__init__('mvcnn_kmeans')
        self.nclasses = nclasses

        self.net_2 = model.net_2
        # reduce dimension
        self.reduce=False
        if not self.reduce:
            self.net_2._modules['0'] = nn.Linear(512*3*7*7,4096)
        else:
            self.redu_conv_1 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU())
            self.redu_conv_2 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU())
            self.redu_conv_3 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU())
            self.net_2._modules['0'] = nn.Linear(128*3*7*7,4096)
        self.att1 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.att2 = nn.Sequential(nn.Linear(128 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 128),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), )

    def forward(self, x):   # x: B,3,512,7,7
        y = x
        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 3,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 3
            scores[i] = temp2
        scores = nn.functional.softmax(scores, 1)
        # scores = nn.functional.sigmoid(scores)
        y_att = y.view(y.shape[0], y.shape[1], -1)*scores.unsqueeze(2)
        y_att = y_att.view(y.shape)     # B,3,512,7,7
        # sort
        _, idx = torch.sort(scores, dim=1, descending=True)
        y_sort = y_att.new_zeros(y_att.shape[0], 512 * 3, y_att.shape[3], y_att.shape[4])  # B,512*3,7,7
        for i in range(0,y_att.shape[0]):
            for j in range(0,y_att.shape[1]):
                y_sort[i,j*512:(j+1)*512,:,:] = y_att[i,idx[i,j],:,:,:]

        if self.reduce:

            y_sort = self.redu_conv(y_sort)

        return self.net_2(y_sort.view(y_sort.shape[0], -1))

    def get_attention(self,x):
        y = x
        scores = y.new_zeros(y.shape[0], y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])  # 3,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1], -1)).squeeze()  # 3
            scores[i] = temp2
        # scores = nn.functional.softmax(scores, 1)
        scores = nn.functional.sigmoid(scores)
        y_att = y.view(y.shape[0], y.shape[1], -1) * scores.unsqueeze(2)
        y_att = y_att.view(y.shape)  # B,3,512,7,7
        # sort
        _, idx = torch.sort(scores, dim=1, descending=True)
        y_sort = y_att.new_zeros(y_att.shape[0], 512 * 3, y_att.shape[3], y_att.shape[4])  # B,512*3,7,7
        for i in range(0, y_att.shape[0]):
            for j in range(0, y_att.shape[1]):
                y_sort[i, j * 512:(j + 1) * 512, :, :] = y_att[i, idx[i, j], :, :, :]

        if self.reduce:
            y_sort = self.redu_conv(y_sort)

        return scores


class ThreeView_att_cat_no_sort(Model):
    def __init__(self,model,nclasses=40):
        super(ThreeView_att_cat_no_sort,self).__init__('mvcnn_kmeans')
        self.nclasses = nclasses

        self.net_2 = model.net_2
        # reduce dimension
        self.reduce=False
        if not self.reduce:
            self.net_2._modules['0'] = nn.Linear(512*3*7*7,4096)
        else:
            self.redu_conv_1 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU())
            self.redu_conv_2 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU())
            self.redu_conv_3 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU())
            self.net_2._modules['0'] = nn.Linear(128*3*7*7,4096)
        self.att1 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.att2 = nn.Sequential(nn.Linear(128 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 128),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), )

    def forward(self, x):   # x: B,3,512,7,7
        y = x
        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 3,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 3
            scores[i] = temp2
        scores = nn.functional.softmax(scores, 1)
        # scores = nn.functional.sigmoid(scores)
        y_att = y.view(y.shape[0], y.shape[1], -1)*scores.unsqueeze(2)
        y_att = y_att.view(y.shape)     # B,3,512,7,7
        # sort
        _, idx = torch.sort(scores, dim=1, descending=True)
        y_sort = y_att.new_zeros(y_att.shape[0], 512 * 3, y_att.shape[3], y_att.shape[4])  # B,512*3,7,7
        for i in range(0,y_att.shape[0]):
            for j in range(0,y_att.shape[1]):
                y_sort[i,j*512:(j+1)*512,:,:] = y_att[i,j,:,:,:]

        if self.reduce:

            y_sort = self.redu_conv(y_sort)

        return self.net_2(y_sort.view(y_sort.shape[0], -1))


class ThreeView_att_sort_reduce(Model):
    def __init__(self,model,nclasses=40):
        super(ThreeView_att_sort_reduce,self).__init__('mvcnn_kmeans')
        self.nclasses = nclasses

        self.net_2 = model.net_2
        # reduce dimension
        self.reduce=True
        if not self.reduce:
            self.net_2._modules['0'] = nn.Linear(512*3*7*7,4096)
        else:
            self.redu_conv=[]
            self.redu_conv.append(nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU()).cuda())
            self.redu_conv.append(nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU()).cuda())
            self.redu_conv.append(nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU()).cuda())
            self.net_2._modules['0'] = nn.Linear(256*3*7*7,4096)
        self.att1 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.att2 = nn.Sequential(nn.Linear(128 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 128),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), )

    def forward(self, x):   # x: B,3,512,7,7
        y = x
        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 3,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 3
            scores[i] = temp2
        #scores = nn.functional.softmax(scores, 1)
        scores = nn.functional.sigmoid(scores)
        y_att = y.view(y.shape[0], y.shape[1], -1)*scores.unsqueeze(2)
        y_att = y_att.view(y.shape)     # B,3,512,7,7
        # sort
        _, idx = torch.sort(scores, dim=1, descending=True)
        y_sort = y_att.new_zeros(y_att.shape[0], 256 * 3, y_att.shape[3], y_att.shape[4])  # B,512*3,7,7
        for i in range(0,y_att.shape[0]):
            for j in range(0,y_att.shape[1]):
                y_reduce = self.redu_conv[j](y_att[i,idx[i,j]:idx[i,j]+1,:,:,:])        # 1,256,7,7
                y_sort[i,j*256:(j+1)*256,:,:] = y_reduce.squeeze(0)



        return self.net_2(y_sort.view(y_sort.shape[0], -1))

    def get_attention(self,x):
        y = x
        scores = y.new_zeros(y.shape[0], y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])  # 3,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1], -1)).squeeze()  # 3
            scores[i] = temp2
        # scores = nn.functional.softmax(scores, 1)
        scores = nn.functional.sigmoid(scores)
        y_att = y.view(y.shape[0], y.shape[1], -1) * scores.unsqueeze(2)
        y_att = y_att.view(y.shape)  # B,3,512,7,7
        # sort
        _, idx = torch.sort(scores, dim=1, descending=True)
        y_sort = y_att.new_zeros(y_att.shape[0], 512 * 3, y_att.shape[3], y_att.shape[4])  # B,512*3,7,7
        for i in range(0, y_att.shape[0]):
            for j in range(0, y_att.shape[1]):
                y_sort[i, j * 512:(j + 1) * 512, :, :] = y_att[i, idx[i, j], :, :, :]

        if self.reduce:
            y_sort = self.redu_conv(y_sort)

        return scores






if __name__ == '__main__':

    torch.random.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)

    args = parser.parse_args()

    if True:       # train

        pretraining = not args.no_pretraining
        log_dir = args.name
        create_folder(args.name)
        config_f = open(os.path.join(log_dir, 'config.json'), 'w')
        json.dump(vars(args), config_f)
        config_f.close()


        # STAGE 1

        cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)


        create_folder(log_dir)
        cnet_2 = ThreeView_att_sort(cnet,nclasses=40)
        del cnet

        optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        train_dataset = KmeanImgDataset(args.train_path)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchSize,shuffle=False,num_workers=0)

        val_dataset = KmeanImgDataset(args.val_path)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batchSize,shuffle=False,num_workers=0)


        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: ' + str(len(val_dataset.filepaths)))

        trainer=ModelNetTrainer(cnet_2,train_loader,val_loader,optimizer,nn.CrossEntropyLoss(),'mvcnn',log_dir,num_views=args.num_views)
        trainer.train_kmean_threeview(30)

    else:       # test
        path = '~/mvcnn_pytorch-master_ECCV2018_backup_2019_11_22/MVCNN_kmean_cat_no_sort_128/'
        modelfile = 'model-00002.pth'

        pretraining = not args.no_pretraining
        log_dir = args.name
        cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

        #cnet_2 = ThreeView_att_cat_no_sort(cnet, nclasses=40)
        cnet_2 = ThreeView_att_sort(cnet, nclasses=40)
        del cnet
        cnet_2.cuda()

        cnet_2.load(path, modelfile)
        cnet_2.eval()

        train_dataset = KmeanImgDataset(args.train_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,
                                                   num_workers=0)

        val_dataset = KmeanImgDataset(args.val_path)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
        optimizer = None

        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir,
                                  num_views=args.num_views)

        loss,val_overall_acc,val_mean_class_acc  = trainer.update_validation_accuracy_kmean(None)
        print(1)






        # test two method combine


        # path_1 = '/home/yw/Desktop/mvcnn_pytorch-master_ECCV2018/MVCNN_kmean_sort_batch_128/'
        # modelfile_1 = 'model-00022.pth'
        #
        # path_2 = '/home/yw/Desktop/mvcnn_pytorch-master_ECCV2018/MVCNN_kmean_batch_64/'
        # modelfile_2 ='model-00008.pth'
        #
        # pretraining = not args.no_pretraining
        # log_dir = args.name
        # cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
        #
        # cnet_2_2 = ThreeView_att(cnet,nclasses=40)
        # cnet_2_2.cuda()
        #
        # del cnet
        #
        # cnet_2_2.load(path_2,modelfile_2)
        # cnet_2_2.eval()
        #
        # cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
        # cnet_2_1 = ThreeView_att_sort(cnet, nclasses=40)
        #
        # cnet_2_1.cuda()
        #
        # cnet_2_1.load(path_1, modelfile_1)
        # cnet_2_1.eval()
        # del cnet
        #
        #
        #
        # train_dataset = KmeanImgDataset(args.train_path)
        #
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,
        #                                            num_workers=0)
        #
        # val_dataset = KmeanImgDataset(args.val_path)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
        # optimizer = None
        #
        #
        # total_correct=0
        # total_seen=0
        #
        # for _, data in enumerate(val_loader, 0):
        #
        #     N, V, C, H, W = data[1].size()
        #     input_path = data[2]
        #     kmean_center_feature = np.zeros([N, 3, C, H, W])
        #     for i in range(N):
        #         path = input_path[i]
        #         in_feature = data[1][i]
        #
        #         item_name = path.split('/')[-1].split('.')[0]
        #         path_parient = path[:-len(item_name + '.npy') - len('features/')]
        #         kmean_path = os.path.join(path_parient, 'cluster_fea', item_name + '.npz')
        #         kmean_fea = np.load(kmean_path)
        #         feature_center = kmean_fea['center_feas']
        #         feature_center = feature_center.reshape(-1, 512, 7, 7)
        #         kmean_center_feature[i] = feature_center
        #     in_data = kmean_center_feature
        #     in_data = Variable(torch.from_numpy(in_data)).cuda().float()
        #     target = Variable(data[0]).cuda()
        #
        #     out_data_1 = cnet_2_1(in_data)
        #
        #
        #     out_data_2 = cnet_2_2(in_data)
        #
        #     out_data = out_data_1+out_data_2
        #     pred = torch.max(out_data,1)[1]
        #
        #     correct = (pred==target).float().sum()
        #     total_correct = total_correct+correct
        #     total_seen = total_seen+N
        #
        # print('eval accuracy: ', total_correct.item() / float(total_seen))
        # print('total_correct: ', total_correct.item())
        # print('total seen: ', total_seen)


        # see attention score and their image

        # path = '/home/yw/Desktop/mvcnn_pytorch-master_ECCV2018/MVCNN_kmean_sort_batch_128/'
        # modelfile = 'model-00022.pth'
        #
        # pretraining = not args.no_pretraining
        # log_dir = args.name
        # cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
        #
        # cnet_2 = ThreeView_att_sort(cnet, nclasses=40)
        # del cnet
        # cnet_2.cuda()
        #
        # cnet_2.load(path, modelfile)
        # cnet_2.eval()
        #
        # train_dataset = KmeanImgDataset(args.train_path)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
        #                                            num_workers=0)
        #
        # val_dataset = KmeanImgDataset(args.val_path)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
        # optimizer = None
        #
        # kmean_dir = '/mnt/tmp1/yw/MVCNN_features/car/cluster_fea/car_0095.npz'
        # feature_12_dir = '/mnt/tmp1/yw/MVCNN_features/car/features/car_0095.npy'
        # img_dir = '/mnt/tmp1/yw/MVCNN_features/car/img/car_0095.npy'
        # img_12 = np.load(img_dir)
        # feature_12 = np.load(feature_12_dir)        # 12x512x7x7
        # data = np.load(kmean_dir)
        # feature_center = data['center_feas']
        # feature_center = feature_center.reshape(-1, 512, 7, 7)  # 3x512x7x7
        # fea_idx = data['idx']
        # _, idx = np.where(fea_idx==0)
        # idx = np.expand_dims(idx,1)     # 3x1
        #
        #
        # feature_center_in = np.expand_dims(feature_center,0)
        # feature_center_in = torch.from_numpy(feature_center_in).cuda().float()
        # attention_score= cnet_2.get_attention(feature_center_in)
        # attention_score = attention_score.detach().cpu().numpy()
        # attention_score = attention_score.squeeze(0)
        #
        #
        # # process img
        # std = torch.tensor([0.229, 0.224, 0.225])
        # std = std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # std = std.expand(12, 224, 224, 3).numpy()
        #
        # mean = torch.tensor([0.485, 0.456, 0.406])
        # mean = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # mean = mean.expand(12, 224, 224, 3).numpy()
        #
        # x_new = np.expand_dims(img_12, axis=0)
        # x_0 = x_new[0]
        # x_0 = x_0.transpose(0, 2, 3, 1)
        # x_0_temp = x_0 * std + mean
        # x_0_temp = x_0_temp * 255
        # x_0_temp = np.round(x_0_temp)
        # x_0_temp = np.uint8(x_0_temp)
        #
        # img_center = x_0_temp[idx.squeeze(1)]
        # from PIL import Image
        # import matplotlib.pyplot as plt
        # for i in range(3):
        #     plt.subplot(3,1,i+1)
        #     plt.imshow(img_center[i])
        #     if attention_score[i]==max(attention_score):
        #         plt.title(str(attention_score[i])+'/max')
        #     elif  attention_score[i]==min(attention_score):
        #         plt.title(str(attention_score[i]) + '/min')
        #     else:
        #         plt.title(str(attention_score[i]))
        #     plt.axis('off')
        # plt.show()
        # print(1)

        # trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir,
        #                           num_views=args.num_views)
        #
        # loss,val_overall_acc,val_mean_class_acc  = trainer.update_validation_accuracy_kmean(None)