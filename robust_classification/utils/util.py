"""
@Author: Meir Yossef Levi
@Contact: me.levi@campus.technion.ac.il
@File: util
@Time: 20/09/30
"""


import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        if isinstance(text, str):
            self.f.write(text+'\n')
        elif isinstance(text, dict):
            self.f.write(str(text))
            self.f.write(f'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        
def normalized_entropy(importance):
    """
    Input:
        importance: pointcloud, [B, N]
    Return:
        entropy: per-sample entropy, [B,]
    """
    import math
    B,N = importance.shape
    # normalize the importance
    eps = 1e-6
    importance=importance+eps
    normalized_importance = importance/torch.sum(importance,dim=-1)
    
    # calculate normalized entropy
    entropy = -torch.sum(normalized_importance*torch.log2(normalized_importance), dim=-1)/math.log2(N)
    return entropy
    
def extract_discrete_critical(x, model, k=None):
    """
    Input:
        x: pointcloud, [B, 3, N]
        model: model to use
        k: number of remains points, if not set it is selected adaptively (by entropy, see the paper)
    Return:
        critical_ppc: crop N-k input points with highest discrete importance [B,3,k]
        discrete_critical: per-point critical, [B, N]
    """
    with torch.no_grad():
        B, _, N = x.shape
        
        # extract X_f
        model.eval()
        logits, x_f = model(x) #BxFxN
        maximal_x_f_indices = torch.max(x_f, dim=-1, keepdim=False)[1] #BxF
        
        # count per-point maximal feature
        sorted_importance = torch.zeros(B, N).to(maximal_x_f_indices.device)
        discrete_importance = torch.zeros(B, N).to(maximal_x_f_indices.device)
        for cur_b in range(B):
            m_bincount = torch.bincount(maximal_x_f_indices[cur_b,:], minlength = N)
            bin_sorted = torch.argsort(m_bincount)
            sorted_importance[cur_b,:] = bin_sorted
            discrete_importance[cur_b,:] = m_bincount
        
        if k is None:
            # define adaptive threshold
            entropy = normalized_entropy(discrete_importance)
            k = int(torch.floor(entropy*N).item())        
        
        # crop the input point-cloud according to sorted importance, take only k input points
        critical_ppc = x
        k_sorted = sorted_importance[:,:k]
        B,K = k_sorted.shape 
        k_sorted = k_sorted.reshape(B,1,K).repeat(1,3,1).to(torch.int64)
        critical_ppc = torch.gather(x, index=k_sorted, dim=-1)
        critical_ppc = critical_ppc.detach()
    return critical_ppc, discrete_importance
    
    
def extract_soft_critical(x, model):
    """
    Input:
        x: pointcloud, [B, N, 3]
        model: model to use
    Return:
        soft_critical: per-point critical, [B, N]
    """
    with torch.no_grad():
        num_points = x.shape[-1]
        model.eval()
        logits, x_f = model(x) #BxFxN
        soft_critical = torch.sum(torch.abs(x_f), dim=1, keepdim=False)
    return soft_critical