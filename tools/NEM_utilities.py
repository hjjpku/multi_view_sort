import numpy as np
import torch
import math

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

def compute_outer_loss(mu, gamma, target_f, prior):
    loss_weight = [2.,0.5]
    intra_loss_p = gaussian_MSE_loss(mu,1,target_f)
    inter_loss_p = kl_loss_gaussian(prior, mu, 1, 1)

    # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
    # implemented as sum over all then divide by B
    B = mu.shape[0]
    M = mu.shape[2]
    intra_loss_reshape = intra_loss_p.view(intra_loss_p.shape[0], intra_loss_p.shape[1], intra_loss_p.shape[2],-1)
    intra_loss = torch.sum(intra_loss_reshape * gamma) / (B*M*intra_loss_reshape.shape[-1])
    inter_loss_reshape = inter_loss_p.view(inter_loss_p.shape[0],inter_loss_p.shape[1],inter_loss_p.shape[2],-1)
    inter_loss = torch.sum(inter_loss_reshape * (1. - gamma)) / (B*M*inter_loss_reshape.shape[-1])
    total_loss = intra_loss*loss_weight[0] + loss_weight[1] * inter_loss

    return total_loss, intra_loss, inter_loss
