import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from torchvision import transforms, datasets
import glob
import itertools
import math
import random
from torch.autograd import Variable

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
from models.Model import Model
from tools.NEM_utilities import MyIdentity, bin_rand, bin_uniform, bin_search


parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN_kmean")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=32)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vggm")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
#parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features/")
#parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_val/")
parser.set_defaults(train=False)
#parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vggm_pretrained_imagenet_modelnet10_10hidden_train/")
#parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vggm_pretrained_imagenet_modelnet10_10hidden_val/")
#parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vggm_pretrained_imagenet_train/")
#parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vggm_pretrained_imagenet_val/")
parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vggm_pretrained_imagenet_shapenetcore_train/")
parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vggm_pretrained_imagenet_shapenetcore_test/")

#parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_resnet50_pretrained_imagenet_modelnet40_train/")
#parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_resnet50_pretrained_imagenet_modelnet40_val/")
#parser.add_argument("-train_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vgg19_pretrained_imagenet_shapenetcore_train/")
#parser.add_argument("-val_path", type=str, default="/mnt/cloud_disk/yw/MVCNN_features_vgg19_pretrained_imagenet_shapenetcore_test/")
parser.add_argument("-fea_type", type=str, help="feature type for NEM, fc or conv", default='fc')
parser.add_argument("-layer_norm", type=int, help="use layernorm for NEM or not", default=1)
parser.add_argument("-cluster_n", type=int, help="cluster number", default=4)
parser.add_argument("-rnn_hidden_size", type=int, help="rnn hidden size", default=1024)
parser.add_argument("-iter_num", type=int, help="iteration number for EM", default=15)
parser.add_argument("-stop_gamma_grad", type=int, help="stop the gradient propagation from gamma", default=0)
parser.add_argument("-if_sort", type=int, help="if sort the disentangled views or not ", default=1)
parser.add_argument("-epoch", type=int, help="epoch number", default=70)
parser.add_argument("-if_pool", type=int, help="use pooling rather than concatenation for multi-view", default=0)
parser.add_argument("-train_or_test_mode", type=str, help="train or test", default='train')
parser.add_argument("-modelfile", type=str, help="model file name", default='')
parser.add_argument("-init_type", type=str, help="initializetion strategy for NEM", default='bin_uniform')
parser.add_argument("-no_sig", type=int, help="no sigmoid after rnn-nem", default=1)
parser.add_argument("-bn_init_sigma", type=int, help="use minibatch data to init sigma", default=0)
parser.add_argument("-if_bn", type=int, help="use bn layers", default=0)
parser.add_argument("-if_gamma_prior", type=int, help="if use gamma prior loss", default=1)
parser.add_argument("-if_decoder_warm_up", type=int, help="if warm up decoder", default=0)
parser.add_argument("-if_total_loss", type=int, help="if use loss from each time stamp", default=0)
parser.add_argument("-arch", type=int, help="net architecture", default=0)
parser.add_argument("-e_sigma", type=float, help="net architecture", default=0.1)
parser.add_argument("-dataset", type=str, help="dataset to use, MNet40, MNet10, ShapeNet are supported", default='MNet40')
#os.environ['CUDA_VISIBLE_DEVICES']='1'

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print(log_dir + '   WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


class KmeanImgDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True,fea_type=None, dataset='MNet40'):
        self.dataset = dataset
        if self.dataset == 'MNet40':
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        elif self.dataset == 'MNet10':
            self.classnames = ['table','sofa','dresser','chair','bathtub','toilet','night_stand','monitor','desk','bed']
        elif self.dataset == 'ShapeNet':
            self.classnames = ['2691156', '2747177', '2773838', '2801938', '2808440', '2818832', '2828884', '2843684', '2871439', '2876657', '2880940', '2924116', '2933112', '2942699', '2946921', '2954340', '2958343', '2992529', '3001627', '3046257', '3085013', '3207941', '3211117', '3261776', '3325088', '3337140', '3467517', '3513137', '3593526', '3624134', '3636649', '3642806', '3691459', '3710193', '3759954', '3761084', '3790512', '3797390', '3928116', '3938244', '3948459', '3991062', '4004475', '4074963', '4090263', '4099429', '4225987', '4256520', '4330267', '4379243', '4401088', '4460130', '4468005', '4530566', '4554684']
        else:
            print('not supported dataset!')
        self.root_dir = root_dir
        self.test_mode = test_mode
        self.num_views = num_views

        self.filepaths = []
        for i in range(len(self.classnames)):
            if fea_type == 'fc':
                #all_files = sorted(glob.glob(root_dir + self.classnames[i] + '/fc_features' + '/*.npy'))
                #all_files = sorted(glob.glob(root_dir + self.classnames[i] + '/fc_features_vgg_m_no_pretrained' + '/*.npy'))
                #all_files = sorted(glob.glob(root_dir + self.classnames[i] + '/fc_features_vggm_pretrained_imagenet' + '/*.npy'))
                all_files = sorted(
                    glob.glob(root_dir + self.classnames[i] + '/fc_features_vggm_pretrained_imagenet' + '/*.npy'))
            else:
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



class NEM(Model):
    def __init__(self,nclasses=40,view_num=12,layer_norm=1, k=3, batch_size=32, input_dim=4096, input_type='fc', rnn_hidden_size=1024,iter_num=10, stop_gamma_grad=0, init_type='rand',no_sig=0,bn_init_sigma=0,gamma_prior_loss=0,if_decoder_warm_up=0, if_total_loss=0,arch=0,e_sigma=0.1):
        super(NEM,self).__init__('NEM')
        self.nclasses = nclasses
        self.view_dist = 'gaussian'
        self.e_sigma = e_sigma
        self.bn_init_sigma = bn_init_sigma
        self.total_sigma = 0
        self.k = k # number of clusers
        self.b = batch_size
        self.m = view_num
        self.pred_init = 0.0
        self.rnn_hidden_size = rnn_hidden_size
        self.input_dim = input_dim
        self.type = input_type # fc feature or feature_map
        self.iter_num = iter_num
        self.stop_gamma_grad = stop_gamma_grad
        self.init_type = init_type
        self.no_sig = no_sig
        self.if_gamma_prior = gamma_prior_loss
        self.if_decoder_warm_up = if_decoder_warm_up
        self.if_total_loss = if_total_loss
        self.arch=arch if self.k > 1 else 0
        #  before_NEM => rnn_NEM => after_NEM

        if True: # layer_norm
            self.enc1 = nn.Sequential(nn.Linear(self.input_dim, 1024), nn.LayerNorm(1024), nn.ELU(),
                                          nn.Dropout(0.5))


            self.enc2 = nn.Sequential( nn.Linear(1024*self.m, 4096), nn.LayerNorm(4096), nn.ELU(), nn.Dropout(0.5))

            self.rnn_nem = nn.GRUCell(4096, self.rnn_hidden_size)

            self.after_rnn = nn.Sequential(nn.LayerNorm(self.rnn_hidden_size))

            self.dec1 = nn.Sequential(nn.Linear(self.rnn_hidden_size, 4096), nn.LayerNorm(4096), nn.ReLU(), nn.Dropout(0.5))

            self.dec2 = nn.Sequential(nn.Linear(4096, 1024*self.m), nn.LayerNorm(1024*self.m))

            self.dec3  = nn.Linear(1024, self.input_dim)


    def init_state(self,x_shape,x): # x_shape: b,m,c,h,w
        B =  x_shape[0]
        m = x_shape[1]
        k = self.k
        assert m == x_shape[1], "number of view dose not match"

        # h: Bxk,rnn_h
        h = torch.zeros(B*k,self.rnn_hidden_size)
        #pred: B,k,m,c,w,h or B,k,m,d
        pred = torch.ones([B,k,m,self.input_dim]) * self.pred_init

        #gamma: B,k,m,1
        gamma_prior, gamma = torch.ones([B,k,m,1])/m, torch.ones([B,k,m,1])
        if k==1:
            gamma = torch.ones([B,k,m,1])
        else:
            # init with uniform dist
            if self.init_type == 'rand':
                gamma = torch.rand([B,k,m,1])
                gamma =  gamma/gamma.sum(1).unsqueeze(1)
                gamma_prior = gamma
            elif self.init_type == 'bin_rand':
                gamma, gamma_prior = bin_rand(B,k,m)

            elif self.init_type == 'bin_uniform':
                gamma, gamma_prior = bin_uniform(B,k,m)
            elif self.init_type == 'bin_search':
                gamma = bin_search(B,k,m,x)

        return h.cuda(),pred.cuda(),gamma.cuda(),gamma_prior.cuda()

    def run_inner_rnn(self, input, h_old):
        # input: b,k,m,c,w,h or b,k,m,d
        # h_old: bxk, rnn_h
        reshaped_input = input.view(self.b*self.k*self.m,-1)

        #### before_nem
        enc1_input = reshaped_input
        enc1_out = self.enc1(enc1_input) # bxkxm,d

        enc1_out_reshape = enc1_out.view(self.b*self.k,-1) # bxk, mxd
        enc2_out = self.enc2(enc1_out_reshape) #bxk,d

        #### em_rnn
        rnn_output_pre= self.rnn_nem(enc2_out, h_old)
        rnn_output = rnn_output_pre

        h_new = rnn_output #bxk,d

        #### after_nem
        dec1_out = self.dec1(rnn_output)
        dec2_out = self.dec2(dec1_out) #bxk,d
        dec2_out_reshape = dec2_out.view(self.b * self.k * self.m, -1)  # bxkxm,d
        dec3_out = self.dec3(dec2_out_reshape)

        output = dec3_out.view(self.b,self.k,self.m,-1)

        return output, h_new

    def compute_em_probabilities(self,pred,t_data,epsilon=1e-10):
        # pred:b,k,m,c,w,h or b,k,m,d
        # target_data: b,1,m,c,w,h or b,1,m,d
        assert self.view_dist == 'gaussian', 'unknown distribution'
        mu, sigma = pred, self.e_sigma
        if self.bn_init_sigma == 0:
            probs = (1/math.sqrt(2*np.pi*sigma**2)) * torch.exp(-(t_data-mu)**2 / (2*sigma**2))
        else:
            probs = (1/torch.clamp(torch.sqrt(2*np.pi*sigma**2),1e-6,1e6) * torch.exp(-(t_data-mu)**2 / torch.clamp(2*sigma**2,1e-6,1e6)))
        if self.type == 'fc':
            probs = probs.sum(-1).unsqueeze(-1)
        else:
            probs = probs.view(self.b,self.k,self.m,-1).sum(-1).unsqueeze(-1)
        if epsilon > 0:
            # add epsilon to probs in order to prevent 0 gamma
            probs += epsilon
        return probs # b,k,m,1

    def e_step(self, pred, target_data):
        # pred:b,k,m,c,w,h or b,k,m,d
        # target_data: b,1,m,c,w,h or b,1,m,d
        probs = self.compute_em_probabilities(pred, target_data)
        gamma = probs / probs.sum(1).unsqueeze(1)
        return gamma

    def forward(self, x, state): # x: B,1,M,512,7,7 or b,1,m,d
        input_data = x
        target_data = x
        if x.shape[0] != self.b:
            self.b = x.shape[0]
        # h: B,rnn_h; pred: B,k,m,c,w,h or b,k,m,d; gamma: B,k,m,1
        h_old, pred_old, gamma_old = state

        delta =  input_data - pred_old # implicitly broadcasts over K

        if self.type == 'fc':
             # masked_delta: B,k,m,d
            masked_delta = delta * gamma_old
        else:
             # masked_delta: B,k,m,c,w,h
            masked_delta = delta * gamma_old.unsqueeze(4).unsqueeze(5)

        pred, h = self.run_inner_rnn(masked_delta, h_old)

        # compute the new gammas
        gamma = self.e_step(pred, target_data)

        return ( h, pred, gamma)

class Multi_View_Net(Model):
    def __init__(self,model,cnn_name,nclasses=40, k=3,rnn_hidden_size=1024, if_sort=1, if_pooling=0,save_feature=0):
        super(Multi_View_Net, self).__init__('MV_C_Net')
        self.nclasses = nclasses
        self.k = k
        self.cnn_name = cnn_name
        self.rnn_hidden_size = rnn_hidden_size
        self.if_sort = if_sort
        self.if_pooling = if_pooling
        self.save_fea = save_feature

        if cnn_name.startswith('vgg'):
            self.net = model.net_2
            if self.if_pooling == 0:
                self.net._modules['0'] = nn.Linear(self.rnn_hidden_size*self.k,4096)
            else:
                self.net._modules['0'] = nn.Linear(self.rnn_hidden_size, 4096)
        elif cnn_name == 'alexnet':
            self.net = model.net_2
            if self.if_pooling == 0:
                self.net._modules['0'] = nn.Linear(self.rnn_hidden_size * self.k, 9216)
            else:
                self.net._modules['0'] = nn.Linear(self.rnn_hidden_size, 9216)
        else:
            if self.if_pooling == 0:
                self.net = nn.Sequential(nn.Linear(self.rnn_hidden_size*self.k,4096), nn.ReLU(), nn.Dropout(0.5),nn.Linear(4096,self.nclasses))
            else:
                self.net = nn.Sequential(nn.Linear(self.rnn_hidden_size, 4096), nn.ReLU(), nn.Dropout(0.5),
                                         nn.Linear(4096, self.nclasses))

        self.att = nn.Sequential(nn.Linear(self.rnn_hidden_size,1), nn.Sigmoid())

    def forward(self, input,gamma): # input: bxk,d
        scores = self.att(input) #bxk,1
        scores_reshape = scores.view(-1,self.k) #b,k
        masked_input = input.view(-1,self.k,input.shape[1]) * scores_reshape.unsqueeze(2) #b,k,d

        if self.if_pooling == 1:
            input_pooled, _ = torch.max(masked_input,1) # maxpool over k4
            if self.save_fea == 0:
                output = self.net(input_pooled)
            else:
                t1 = self.net._modules['0'](input_pooled)
                t2 = self.net._modules['1'](t1)
                t3 = self.net._modules['2'](t2)
                output = self.net._modules['3'](t3)
            return output


        #  sort and concatenate
        if self.if_sort == 1:
            _, idx = torch.sort(scores_reshape, dim=1, descending=True)
            #if self.save_fea == 1:
             #  return idx
            masked_input_sort = torch.zeros_like(masked_input)
            for i in range(masked_input.shape[0]): #b
                for j in range(self.k):
                    masked_input_sort[i,j,:] = masked_input[i,idx[i,j],:]
            masked_input_sort_reshape = masked_input_sort.view(masked_input_sort.shape[0],-1)
            if self.save_fea == 0:
                output = self.net(masked_input_sort_reshape)# b,40
            else:
                t1 = self.net._modules['0'](masked_input_sort_reshape)
                t2 = self.net._modules['1'](t1)
                t3 = self.net._modules['2'](t2)
                output = self.net._modules['3'](t3)
        elif self.if_sort == 2:
            #sort by gamma entropy
            sigma_score = torch.sqrt(torch.sum(((gamma - gamma.mean(2).unsqueeze(2))**2),2)).squeeze() # Bxk
            _, idx = torch.sort(sigma_score, dim=1, descending=True)
            masked_input_sort = torch.zeros_like(masked_input)
            for i in range(masked_input.shape[0]):  # b
                for j in range(self.k):
                    masked_input_sort[i, j, :] = masked_input[i, idx[i, j], :]
            masked_input_sort_reshape = masked_input_sort.view(masked_input_sort.shape[0], -1)
            output = self.net(masked_input_sort_reshape)  # b,40
        else:
            if self.save_fea == 0:
                output = self.net(masked_input.view(masked_input.shape[0],-1))
            else:
                t1 = self.net._modules['0'](masked_input.view(masked_input.shape[0],-1))
                t2 = self.net._modules['1'](t1)
                t3 = self.net._modules['2'](t2)
                output = self.net._modules['3'](t3)
        return output

if __name__ == '__main__':

    torch.random.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)

    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'MNet40':
        class_num = 40
    elif dataset == 'MNet10':
        class_num=10
    elif dataset == 'ShapeNet':
        class_num=55

    if args.train_or_test_mode == 'train':       # train

        pretraining = not args.no_pretraining
        log_dir = args.name
        create_folder(args.name)
        config_f = open(os.path.join(log_dir, 'config.json'), 'w')
        json.dump(vars(args), config_f)
        config_f.close()

        # STAGE 1

        cnet = SVCNN(args.name, nclasses=class_num, pretraining=pretraining, cnn_name=args.cnn_name)


        create_folder(log_dir)

        if args.cnn_name.startswith('vgg'):
            input_dim = 4096
        elif args.cnn_name == 'alexnet':
            input_dim = 4096
        elif args.cnn_name == 'resnet50':
            input_dim = 2048
        elif args.cnn_name == 'googlenet':
            input_dim =1024
        else:
            raise KeyError('the backbone is not suported yet')
        if args.fea_type == 'fc':
            type = 'fc'
        else:
            type = None
            input_dim = None

        nem = NEM(nclasses=class_num, layer_norm=args.layer_norm,k=args.cluster_n,batch_size=args.batchSize, input_dim=input_dim, input_type=type, rnn_hidden_size=args.rnn_hidden_size,iter_num=args.iter_num, stop_gamma_grad=args.stop_gamma_grad,init_type=args.init_type,no_sig=args.no_sig,bn_init_sigma=args.bn_init_sigma, gamma_prior_loss=args.if_gamma_prior,if_decoder_warm_up=args.if_decoder_warm_up,if_total_loss=args.if_total_loss,arch=args.arch,e_sigma=args.e_sigma)
        nem.cuda()

        cnet_2 = Multi_View_Net(cnet,cnn_name=args.cnn_name, nclasses=class_num,k=args.cluster_n, rnn_hidden_size=args.rnn_hidden_size,if_sort=args.if_sort, if_pooling=args.if_pool)
        del cnet
        cnet_2.cuda()

        optimizer = optim.Adam(itertools.chain(nem.parameters(),cnet_2.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))


        train_dataset = KmeanImgDataset(args.train_path, fea_type=args.fea_type,dataset=dataset) # fc feature path or conv feature path
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchSize,shuffle=True,num_workers=0)

        val_dataset = KmeanImgDataset(args.val_path,fea_type=args.fea_type,dataset=dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batchSize,shuffle=False,num_workers=0)


        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: ' + str(len(val_dataset.filepaths)))

        trainer=ModelNetTrainer((nem,cnet_2),train_loader,val_loader,optimizer,nn.CrossEntropyLoss(),None,log_dir,num_views=args.num_views,class_num=class_num)
        trainer.train_nem_mvcnn(args.epoch)

    else:       # test
        path = '/mnt/cloud_disk/huangjj/exp_mvcnn/'+args.name
        modelfile = args.modelfile

        pretraining = not args.no_pretraining
        log_dir = args.name
        cnet = SVCNN(args.name, nclasses=class_num, pretraining=pretraining, cnn_name=args.cnn_name)

        if args.cnn_name.startswith('vgg'):
            input_dim = 4096
        elif args.cnn_name == 'alexnet':
            input_dim = 4096
        elif args.cnn_name == 'resnet50':
            input_dim = 2048
        elif args.cnn_name == 'googlenet':
            input_dim =1024
        else:
            raise KeyError('the backbone is not suported yet')
        if args.fea_type == 'fc':
            type = 'fc'
        else:
            type = None
            input_dim = None

        nem = NEM(nclasses=class_num, layer_norm=args.layer_norm, k=args.cluster_n, batch_size=args.batchSize, input_dim=input_dim,
                  input_type=type, rnn_hidden_size=args.rnn_hidden_size, iter_num=args.iter_num,
                  stop_gamma_grad=args.stop_gamma_grad,init_type=args.init_type,no_sig=args.no_sig,bn_init_sigma=args.bn_init_sigma,gamma_prior_loss=args.if_gamma_prior,arch=args.arch,e_sigma=args.e_sigma)

        nem.load(path, modelfile)

        cnet_2 = Multi_View_Net(cnet, cnn_name=args.cnn_name, nclasses=class_num, k=args.cluster_n, rnn_hidden_size=args.rnn_hidden_size, if_sort=args.if_sort, if_pooling=args.if_pool,save_feature=1)
        del cnet
        cnet_2.load(path, modelfile)
        cnet_2.eval()

        val_dataset = KmeanImgDataset(args.val_path, fea_type=args.fea_type,dataset=dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)

        optimizer = None

        trainer = ModelNetTrainer((nem, cnet_2), None, val_loader, optimizer, nn.CrossEntropyLoss(), None,
                                  log_dir, num_views=args.num_views, class_num=class_num)

        trainer.save_feature_nem(None)
        #loss, val_overall_acc, val_mean_class_acc = trainer.update_validation_accuracy_nem(None)
