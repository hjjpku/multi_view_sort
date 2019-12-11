import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

class MVCNN_fourview(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN_fourview, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = 4
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))


class MVCNN_attention(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN_attention, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        self.att = nn.Sequential(nn.Conv2d(512,128,4),nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128,1,2))

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)

        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            scores[i] = self.att(y[i]).squeeze()    # N, 12
        scores = nn.functional.softmax(scores,1)
        y_att = torch.bmm(y.view(y.shape[0],y.shape[1],-1).permute(0,2,1),scores.unsqueeze(2))
        y_att = y_att.view(y.shape[0],y.shape[2],y.shape[3],y.shape[4])
        # y_att = torch.mul(y.view(y.shape[0],y.shape[1],-1),(1+scores.unsqueeze(2))).view(y.shape)      # 8,12,*  x 8,12,1 = 8,12,*
        # y = y_att
        return self.net_2(y_att.view(y.shape[0],-1))

class MVCNN_attention_fc(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN_attention_fc, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        self.att1 = nn.Sequential(nn.Conv2d(512,128,1))
        self.att2 = nn.Sequential(nn.Linear(128*7*7,1024),nn.ReLU(),nn.Dropout(0.5),nn.Linear(1024,128),nn.ReLU(),nn.Dropout(0.5),nn.Linear(128,1),)

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)

        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 12,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 12
            scores[i] = temp2
        scores = nn.functional.softmax(scores,1)
        y_att = torch.bmm(y.view(y.shape[0],y.shape[1],-1).permute(0,2,1),scores.unsqueeze(2))
        y_att = y_att.view(y.shape[0],y.shape[2],y.shape[3],y.shape[4])
        # y_att = torch.mul(y.view(y.shape[0],y.shape[1],-1),(1+scores.unsqueeze(2))).view(y.shape)      # 8,12,*  x 8,12,1 = 8,12,*
        # y = y_att
        return self.net_2(y_att.view(y.shape[0],-1))


class MVCNN_attention_fc_sort(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN_attention_fc_sort, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
            self.net_2_2 = model.net_2

        self.att1 = nn.Sequential(nn.Conv2d(512,128,1))
        self.att2 = nn.Sequential(nn.Linear(128*7*7,1024),nn.ReLU(),nn.Dropout(0.5),nn.Linear(1024,128),nn.ReLU(),nn.Dropout(0.5),nn.Linear(128,1),)

        # branch 2 sort
        self.redu_conv = nn.Sequential(nn.Conv2d(512,64,1),nn.ReLU())
        self.clas_conv = nn.Sequential(nn.Conv2d(64*12,512,1),nn.ReLU())


    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)

        scores = y.new_zeros(y.shape[0],y.shape[1])
        y_redu = y.new_zeros(y.shape[0],y.shape[1],64,y.shape[3],y.shape[4])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 12,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 12
            scores[i] = temp2

            y_redu[i] = self.redu_conv(y[i])
        scores = nn.functional.softmax(scores,1)
        y_att = torch.bmm(y.view(y.shape[0],y.shape[1],-1).permute(0,2,1),scores.unsqueeze(2))
        y_att = y_att.view(y.shape[0],y.shape[2],y.shape[3],y.shape[4])

        # branch 2 sort
        _, idx = torch.sort(scores,dim=1,descending=True)
        y_sort = y.new_zeros(y.shape[0],64*12,y.shape[3],y.shape[4])        # B,64*12,7,7
        for i in range(0,y.shape[0]):
            for j in range(0,y.shape[1]):
                y_sort[i,j*64:(j+1)*64,:,:] = y_redu[i,idx[i,j],:,:,:]

        y_sort = self.clas_conv(y_sort)     # B,512,7,7


        return self.net_2(y_att.view(y.shape[0],-1)), self.net_2_2(y_sort.view(y.shape[0],-1))


class MVCNN_attention_fc_fourview(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN_attention_fc_fourview, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = 4
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        self.att1 = nn.Sequential(nn.Conv2d(512,128,1))
        self.att2 = nn.Sequential(nn.Linear(128*7*7,1024),nn.ReLU(),nn.Dropout(0.5),nn.Linear(1024,128),nn.ReLU(),nn.Dropout(0.5),nn.Linear(128,1),)

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)

        scores = y.new_zeros(y.shape[0],y.shape[1])
        for i in range(y.shape[0]):
            temp1 = self.att1(y[i])     # 12,128,7,7
            temp2 = self.att2(temp1.view(y.shape[1],-1)).squeeze()    # 12
            scores[i] = temp2
        scores = nn.functional.softmax(scores,1)
        y_att = torch.bmm(y.view(y.shape[0],y.shape[1],-1).permute(0,2,1),scores.unsqueeze(2))
        y_att = y_att.view(y.shape[0],y.shape[2],y.shape[3],y.shape[4])
        # y_att = torch.mul(y.view(y.shape[0],y.shape[1],-1),(1+scores.unsqueeze(2))).view(y.shape)      # 8,12,*  x 8,12,1 = 8,12,*
        # y = y_att
        return self.net_2(y_att.view(y.shape[0],-1))


class MVCNN_self_attention(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN_self_attention, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        self.att = Self_Attn_view(512)

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)

        y = self.att(y)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

class Self_Attn_view(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn_view,self).__init__()
        self.channel_in = in_dim

        #self.in_conv = nn.Sequential(nn.Conv2d(self.channel_in,self.channel_in/4,1),nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(256,128,2))   # 12,128,1,1
        self.query_conv = nn.Conv2d(self.channel_in,self.channel_in//4,1)
        self.key_conv = nn.Conv2d(self.channel_in,self.channel_in//4,1)
        self.value_conv = nn.Conv2d(self.channel_in,self.channel_in,1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, view_num, _,H,W = x.shape
        proj_query = x.new_zeros(B,view_num, self.channel_in//4*H*W)
        proj_key = x.new_zeros(B,view_num, self.channel_in//4*H*W)
        proj_value = x.new_zeros(x.shape)
        for i in range(B):
            proj_query[i] = self.query_conv(x[i]).view(view_num,-1)
            proj_key[i] = self.key_conv(x[i]).view(view_num,-1)
            proj_value[i] = self.value_conv(x[i])
        proj_key = proj_key.permute(0,2,1)  # B,C,12
        energy = torch.bmm(proj_query,proj_key)     # B,12,12
        attention = nn.functional.softmax(energy,-1)    # B,12,12
        proj_value = proj_value.view(B,view_num,-1)     # B,12,512*7*7
        out = torch.bmm(attention,proj_value)       # B,12,512*7*7
        out = out.view(B,view_num,self.channel_in,H,W)  # B,12,512,7,7
        out = self.gamma*out +x
        return out







# if __name__=='__main__':
#
#     from torch.autograd import Variable
#
#     torch.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     x = Variable(torch.randn(8*12, 3, 224, 224).cuda(), requires_grad=True)
#     print(x.shape)
#
#     cnet = SVCNN('MVCNN', nclasses=40, pretraining=True, cnn_name='vgg11')
#
#     cnet2 = MVCNN_attention('MVCNN',cnet, nclasses=40, cnn_name='vgg11', num_views=12)
#
#     out = cnet2(x)
#
#     print(out.shape)


#
# std=torch.tensor([0.229, 0.224, 0.225])
# std = std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
# std = std.expand(12,224,224,3)
#
# mean = torch.tensor([0.485, 0.456, 0.406])
# mean = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
# mean = mean.expand(12,224,224,3)
#
# x_new = x.view(x.shape[0]//12,12,x.shape[1],x.shape[2],x.shape[3])
# x_0 = x_new[0]
# x_0 = x_0.permute(0,2,3,1)
# x_0_temp = x_0.cpu()*std+mean
# x_0_temp = x_0_temp*255
# x_0_temp = x_0_temp.numpy()
# x_0_temp = np.round(x_0_temp)
# x_0_temp = np.uint8(x_0_temp)
# from PIL import Image
# #
# for i in range(0,12):
#     im = Image.fromarray(x_0_temp[i])
#     im.show()