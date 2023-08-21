import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils.GDANet_util import local_operator, GDM, SGCAM

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
    ent = my_entropy(imp3)
    k = int(torch.floor(ent*imp2.shape[1]).item())        
    tot_counter_sorted_k = imp2[:,:k]
    B,K = tot_counter_sorted_k.shape 
    tot_counter_sorted_k = tot_counter_sorted_k.reshape(B,1,K).repeat(1,3,1).to(torch.int64)
    adaboost_ppc = torch.gather(x, index=tot_counter_sorted_k, dim=-1)
    #importance_ppc = adaboost_ppc.detach()
    return adaboost_ppc, imp3


class get_model(nn.Module):
    def __init__(self, output_channels=40, normal_channel=False):
        super(get_model, self).__init__()
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)

        self.linear1 = nn.Linear(1024, 512, bias=True)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256, bias=True)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(256, 40, bias=True)

    def forward(self, x):
        B, C, N = x.size()
        ###############
        """block 1"""
        # Local operator:
        x1 = local_operator(x, k=min(30, N))
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=min(128, N))

        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))
        z1 = torch.cat([y1s, y1g], 1)
        z1 = F.relu(self.conv12(z1))
        ###############
        """block 2"""
        x1t = torch.cat((x, z1), dim=1)
        x2 = local_operator(x1t, k=min(30, N))
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x2s, x2g = GDM(x2, M=min(128,N))

        y2s = self.SGCAM_2s(x2, x2s.transpose(2, 1))
        y2g = self.SGCAM_2g(x2, x2g.transpose(2, 1))
        z2 = torch.cat([y2s, y2g], 1)
        z2 = F.relu(self.conv22(z2))
        ###############
        x2t = torch.cat((x1t, z2), dim=1)
        x3 = local_operator(x2t, k=min(30, N))
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        x3 = x3.max(dim=-1, keepdim=False)[0]
        z3 = F.relu(self.conv32(x3))
        ###############
        x = torch.cat((z1, z2, z3), dim=1)
        x = F.relu(self.conv4(x))
        x11 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x22 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x11, x22), 1)

        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x