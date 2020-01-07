import numpy as np
import torch
import torch.nn as nn
import math
import random
from scipy.stats import norm

# gamma initializaiton strategy
def compute_norm(center,m):
    init_center = int(m/2)
    num_list = np.linspace(-0.5,0.5,m,endpoint=True)
    offset = init_center - center
    num_list_roll = np.roll(num_list, offset)

    dist = norm.pdf(num_list_roll,loc=0,scale=0.15) #norm(0,1)
    dist_norm = dist/np.sum(dist)
    return torch.Tensor(dist_norm).unsqueeze(1)

def bin_rand(B,k,m):
    gamma = torch.zeros([B, k, m, 1])
    gamma_prior = torch.zeros([B, k, m, 1])
    list_view = [i for i in range(m)]
    random.shuffle(list_view)
    for i in range(k):
        gamma[:, i, list_view[i], :] = 1
        gamma_prior[:, i] = compute_norm(list_view[i],m)
    return gamma, gamma_prior

def bin_uniform(B,k,m):
    gamma = torch.zeros([B, k, m, 1])
    gamma_prior = torch.zeros([B, k, m, 1])
    stride = int(m / k)
    for i in range(k):
        gamma[:, i, 0 + i * stride, :] = 1
        gamma_prior[:,i] = compute_norm(0 + i * stride,m)
    return gamma, gamma_prior

def bin_search(B,k,m,x): # x:b,m,c,h,w
    x_reshape = x.view(B,m,-1)
    x_reshape_trans = torch.transpose(x_reshape,1,2)
    dist_matrix = torch.bmm(x_reshape,x_reshape_trans)

    gamma = torch.zeros([B, k, m, 1])
    stride = int(m / k)
    range = 1
    idx = [0]


class MyIdentity(nn.Module):
    '''
    A placeholder identity operator ;
    nn.Identity() is not surported in this version of pytorch yet
    '''
    def __init__(self):
        super(MyIdentity, self).__init__()

    def forward(self, input):
        return input

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if not param.grad is None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def compute_prior(type):
    #return: bernoulli prior (1,1,1,1,1,1) or (1,1,1,1) b,k,m,d
    if type == 'fc':
        prior = torch.zeros(1,1,1,1)
    else:
        prior = torch.zeros(1,1,1,1,1,1)
    return prior.cuda()



def gaussian_MSE_loss(mu, sigma, t):
   # -log(N(mu,sigma))
    intra_loss = (mu - t)**2 / (min(max(sigma**2, 1e-6),1e6)) + math.log(min(1e6,max(1e-6,sigma)))
    return intra_loss

def kl_loss_gaussian(prior_mu, mu,prior_sigma, sigma):
    inter_loss = math.log(min(max(prior_sigma/sigma, 1e-6),1e6)) + (sigma**2+(mu-prior_mu)**2)/(2*prior_sigma**2) - 0.5
    return inter_loss

def kl_loss_gamma(prior, pred):
    loss_f = nn.KLDivLoss(reduction='batchmean')
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = loss_f(logsoftmax(pred.view(-1,pred.shape[2])), prior.view(-1,prior.shape[2]))
    return loss

def gamma_entropy_loss(gamma):
    # B,K,M,1
    entropy = torch.sum(gamma * torch.log(gamma+1e-6))/gamma.shape[0]/gamma.shape[2] * -1.0
    sigma = torch.sum(torch.sqrt(torch.sum(((gamma - gamma.mean(2).unsqueeze(2))**2),2)))/gamma.shape[0]/gamma.shape[1]
    return entropy - sigma


def compute_outer_loss(mu, gamma, target_f, prior, stop_gamma_grad, gamma_prior, if_gamma_prior):
    loss_weight = [2,0.5] if if_gamma_prior == 0 else [2,1]#[2,1]
    intra_loss_p = gaussian_MSE_loss(mu,1,target_f)
    if if_gamma_prior == 0:
        inter_loss_p = kl_loss_gaussian(prior, mu, 1, 1)
    elif if_gamma_prior == 1:
        inter_loss_p = kl_loss_gamma(gamma_prior,gamma) #B,K,M,1
    elif if_gamma_prior == 2:
        inter_loss_p = gamma_entropy_loss(gamma) +  kl_loss_gamma(gamma_prior,gamma)

    # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
    # implemented as sum over all then divide by B
    B = mu.shape[0]
    M = mu.shape[2]
    intra_loss_reshape = intra_loss_p.view(intra_loss_p.shape[0], intra_loss_p.shape[1], intra_loss_p.shape[2],-1)
    if if_gamma_prior == 0:
        inter_loss_reshape = inter_loss_p.view(inter_loss_p.shape[0], inter_loss_p.shape[1], inter_loss_p.shape[2], -1)
    else:
        inter_loss_reshape = inter_loss_p

    if stop_gamma_grad == 0:
        intra_loss = torch.sum(intra_loss_reshape * gamma) / (B*M*intra_loss_reshape.shape[-1])
        inter_loss = torch.sum(inter_loss_reshape * (1. - gamma)) / (B * M * inter_loss_reshape.shape[-1]) if if_gamma_prior ==0 else inter_loss_reshape
    else:
        intra_loss = torch.sum(intra_loss_reshape * gamma.detach()) / (B * M * intra_loss_reshape.shape[-1])
        inter_loss = torch.sum(inter_loss_reshape * (1. - gamma.detach())) / (B * M * inter_loss_reshape.shape[-1]) if if_gamma_prior == 0 else inter_loss_reshape


    total_loss = intra_loss*loss_weight[0] + loss_weight[1] * inter_loss

    return total_loss, intra_loss, inter_loss
