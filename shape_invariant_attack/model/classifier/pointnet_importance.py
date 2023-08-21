#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    #idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    
    _, indices = torch.sort(pairwise_distance, dim=-1, descending=True)
    idx = indices[:,:,:k]
    idx = idx.to(torch.int64)
    
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

def my_entropy(x):
    
    import math
    eps = 1e-6
    x=x+eps
    x = x/torch.sum(x,dim=-1)
    
    B,N = x.shape
    ent = -torch.sum(x*torch.log2(x), dim=-1)/math.log2(N)
    return ent
    
def extract_importance(x, model, k):
    num_points = x.shape[-1]
    logits, x_f = model(x, False) #BxFxN
    imp = torch.max(x_f, dim=-1, keepdim=False)[1] #BxF
    imp2 = torch.zeros(imp.shape[0], num_points).to(imp.device)
    imp3 = torch.zeros(imp.shape[0], num_points).to(imp.device)
    for cur_b in range(imp.shape[0]):
        m_bincount = torch.bincount(imp[cur_b,:], minlength = num_points)
        bin_sorted = torch.argsort(m_bincount)
        imp2[cur_b,:] = bin_sorted
        imp3[cur_b,:] = m_bincount
    
    #tot_counter_sorted = torch.argsort(imp, dim=-1)
    importance_ppc = x
    #ent = my_entropy(imp3)
    #k = int(torch.floor(ent*imp2.shape[1]).item())        
    tot_counter_sorted_k = imp2[:,:k]
    B,K = tot_counter_sorted_k.shape 
    tot_counter_sorted_k = tot_counter_sorted_k.reshape(B,1,K).repeat(1,3,1).to(torch.int64)
    adaboost_ppc = torch.gather(x, index=tot_counter_sorted_k, dim=-1)
    #importance_ppc = adaboost_ppc.detach()
    return adaboost_ppc, imp3

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x, use_imp=True):
        if use_imp:
            x, x_f0 = extract_importance(x, self, 600)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        if not use_imp:
            return x, x_f
        else:
            return x


class get_model(nn.Module):
    def __init__(self, output_channels=40, normal_channel=False):
        super(get_model, self).__init__()
        self.emb_dims = 1024
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.linear1 = nn.Linear(self.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x, use_imp=True):
        batch_size = x.size(0)
        if use_imp:
            x, x_f0 = extract_importance(x, self, 600)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x_f = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x_f, 1).view(batch_size, -1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        if not use_imp:
            return x, x_f
        else:
            return x