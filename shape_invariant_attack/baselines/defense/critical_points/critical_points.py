import torch
import torch.nn as nn


def my_entropy(x):
    
    import math
    eps = 1e-6
    x=x+eps
    x = x/torch.sum(x,dim=-1)
    
    B,N = x.shape
    ent = -torch.sum(x*torch.log2(x), dim=-1)/math.log2(N)
    return ent
    
def extract_importance(x, model):
    num_points = x.shape[-1]
    logits, x_f = model(x, True) #BxFxN
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
    ent = my_entropy(imp3)
    k = int(torch.floor(ent*imp2.shape[1]).item())        
    
    tot_counter_sorted_k = imp2[:,:k]
    B,K = tot_counter_sorted_k.shape 
    tot_counter_sorted_k = tot_counter_sorted_k.reshape(B,1,K).repeat(1,3,1).to(torch.int64)
    adaboost_ppc = torch.gather(x, index=tot_counter_sorted_k, dim=-1)
    #importance_ppc = adaboost_ppc.detach()
    return importance_ppc, imp3

class critical_points(nn.Module):
    def __init__(self, model):
        super(critical_points, self).__init__()
        self.model = model

    def forward(self, x):
        with torch.enable_grad():
            importance_ppc, imp = extract_importance(x, self.model)
        return importance_ppc



