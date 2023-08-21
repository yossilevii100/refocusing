#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Meir Yossef Levi
@Contact: me.levi@campus.technion.ac.il
@File: main.py
@Time: 2023/08/21 10:39 PM
"""

from __future__ import print_function
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import ModelNet40

# import models
from models.dgcnn.model_dgcnn import DGCNN, PointNet
from models.gdanet.GDANet_cls import GDANET
from models.pct.model_pct import Pct, RPC

# import utils
from utils.util import *
from utils.epic_util import *
from utils.lpf_util import spherical_harmonics_defense
from modelnetc_utils import eval_corrupt_wrapper, ModelNetC

import sklearn.metrics as metrics
from augmentation.rsmix import provider, rsmix_provider
import random


def load_model(args, device):
    model = None
    if args.model == "gdanet":
        model = GDANET().to(device)
    elif args.model == "curvenet":
        model = CurveNet().to(device)
    elif args.model == "pct":
        model = Pct(args).to(device)
    elif args.model == "rpc":
        model = RPC(args).to(device)
    elif args.model == "dgcnn":
        model = DGCNN(args).to(device)
    elif args.model == "pointnet":
        model = PointNet(args).to(device)
    elif args.model == "paconv":
        model = PAConv(args).to(device)
    # placeholder to future researches to implant and examine their classification networks 
    elif args.model == 'custom_model':
        model = custom_model(args).to(device)
    return model
    
    
def apply_rsmix(args, data, device, label, label_b, lam, rsmix):
    if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (
            args.beta != 0.0):
        data = data.cpu().numpy()
    if args.rot:
        data = provider.rotate_point_cloud(data)
        data = provider.rotate_perturbation_point_cloud(data)
    if args.rdscale:
        tmp_data = provider.random_scale_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = tmp_data
    if args.shift:
        tmp_data = provider.shift_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = tmp_data
    if args.jitter:
        tmp_data = provider.jitter_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = tmp_data
    if args.rddrop:
        data = provider.random_point_dropout(data)
    if args.shuffle:
        data = provider.shuffle_points(data)
    r = np.random.rand(1)
    if args.beta > 0 and r < args.rsmix_prob:
        rsmix = True
        data, lam, label, label_b = rsmix_provider.rsmix(data, label, beta=args.beta, n_sample=args.nsample,
                                                         KNN=args.knn)
    if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (
            args.beta != 0.0):
        data = torch.FloatTensor(data)
    if rsmix:
        lam = torch.FloatTensor(lam)
        lam, label_b = lam.to(device), label_b.to(device).squeeze()
    else:
        lam = None
        label_b = None
    return data, label, label_b, lam, rsmix

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) * 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # train on ModelNet40 and test on ModelNet-C              
    train_loader = DataLoader(ModelNet40(args=args, partition='train'), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
        
    test_loader = DataLoader(ModelNetC(args=args, split="clean"),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=True)
    
    
    # init model
    device = torch.device("cuda" if args.cuda else "cpu")
    model_critical = load_model(args, device)
    if args.is_retrain:
        model_path_critical = os.path.join('checkpoints', args.exp_name, 'models',
                                                 f'{args.model}_critical{"_lpf" if args.is_lpf else ""}{"_wm" if args.use_wolfmix else ""}{"_pointguard" if args.is_pointguard else ""}.t7') if args.model_path_critical == "" else args.model_path_critical
        checkpoint = torch.load(model_path_critical, map_location='cpu')
        model_critical.load_state_dict(checkpoint)
    opt = optim.SGD(model_critical.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs)

    criterion = cal_loss
    best_test_acc = 0
    m_lpf = spherical_harmonics_defense()
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model_critical.train()
        train_pred = []
        train_true = []
        for m_iter, (data, label) in enumerate(train_loader):
            # wolfmix
            rsmix = False
            lam = False
            label_b = False
            if args.use_wolfmix:
                # RSMIX
                data, label, label_b, lam, rsmix = apply_rsmix(args, data, device, label, label_b, lam, rsmix)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            if args.is_pointguard:
                s = torch.randperm(data.shape[-1])
                ppc_critical = data[:,:,s[:16]]
            elif args.is_lpf:
                ppc_critical = m_lpf(data)
            else:
                # train on rand between 256 to num_points
                num_rand_points = torch.randint(256,data.shape[-1], (1,)).item()
                ppc_critical, _ = extract_discrete_critical(data, model_critical, num_rand_points)
                model_critical.train()
            logits,_ = model_critical(ppc_critical)
            batch_size = logits.shape[0]
            if not rsmix:
                loss = criterion(logits, label)
            else:
                loss = 0
                for i in range(batch_size):
                    loss_tmp = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - lam[i]) \
                               + criterion(logits[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                    loss += loss_tmp
                loss = loss / batch_size
            loss.backward()
            opt.step()
            count += batch_size
            train_true.append(label.cpu().numpy())
            preds = logits.max(dim=1)[1]
            train_loss += loss.item() * batch_size
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        
        outstr = 'critical: Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))

        io.cprint(outstr)
        with torch.no_grad():
            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model_critical.eval()
            test_pred = []
            test_true = []
            for m_iter, (data, label) in enumerate(test_loader):
                torch.cuda.empty_cache()
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                data = data[:,:,torch.randperm(data.shape[-1])]
                batch_size = data.size()[0] 
                if args.is_pointguard:
                    s = torch.randperm(data.shape[-1])
                    # PointGuard paper advocates 16 elements per sub-sample
                    ppc_critical = data[:,:,s[:16]]
                elif args.is_lpf:
                    ppc_critical = m_lpf(data)
                else:
                    num_rand_points = torch.randint(256,data.shape[-1], (1,)).item()
                    ppc_critical, _ = extract_discrete_critical(data, model_critical, num_rand_points)
                logits,_ = model_critical(ppc_critical)
                loss = criterion(logits, label)
                count += batch_size
                preds = logits.max(dim=1)[1]
                test_loss += loss.item() * batch_size
                test_pred.append(preds.detach().cpu().numpy())
                test_true.append(label.cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)                                                                                        
            
            test_acc = metrics.accuracy_score(test_true, test_pred)
            outstr = 'critical: Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                     test_loss * 1.0 / count,
                                                                                     test_acc,
                                                                                     metrics.balanced_accuracy_score(
                                                                                         test_true, test_pred))
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                model_path_critical = os.path.join('checkpoints', args.exp_name, 'models',
                                                 f'{args.model}_critical{"_lpf" if args.is_lpf else ""}{"_wm" if args.use_wolfmix else ""}{"_pointguard" if args.is_pointguard else ""}.t7') if args.model_path_critical == "" else args.model_path_critical
                best_test_acc = test_acc
                torch.save(model_critical.state_dict(), model_path_critical)
            io.cprint(f'critical: best test: {best_test_acc}')

def test_ensemble(args, io):
    device = torch.device("cuda" if args.cuda else "cpu")

    # load model patches
    model_patches = load_model(args, device)
    model_path_patches = os.path.join('pretrained',
                                      f'{args.model}_patches{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_patches == "" else args.model_path_patches
    checkpoint = torch.load(model_path_patches, map_location='cpu')
    model_patches.load_state_dict(checkpoint)

    # load model curves
    model_curves = load_model(args, device)
    model_path_curves = os.path.join('pretrained',
                                     f'{args.model}_curves{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_curves == "" else args.model_path_curves
    checkpoint = torch.load(model_path_curves, map_location='cpu')
    model_curves.load_state_dict(checkpoint)

    # load model random
    model_random = load_model(args, device)
    model_path_random = os.path.join('pretrained',
                                     f'{args.model}_random{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_random == "" else args.model_path_random
    checkpoint = torch.load(model_path_random, map_location='cpu')
    model_random.load_state_dict(checkpoint)
    
    # load model critical
    if args.use_ensemble_all:
        model_critical = load_model(args, device)
        model_path_critical = os.path.join('pretrained',
                                         f'{args.model}_critical_256_1024{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_critical == "" else args.model_path_critical
        checkpoint = torch.load(model_path_critical, map_location='cpu')
        model_critical.load_state_dict(checkpoint)

    model_patches.eval()
    model_curves.eval()
    model_random.eval()
    if args.use_ensemble_all:
        model_critical.eval()

    def test_corrupt(args, split, model):
        import time
        args.test_batch_size = 1
        with torch.no_grad():
            model_patches = model[0]
            model_curves = model[1]
            model_random = model[2]
            if args.use_ensemble_all:
                model_critical = model[3]
            test_loader = DataLoader(ModelNetC(args=args, split=split),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            test_true = []
            test_pred = []
            set_fixed_seed(args)
            start = time.time()
            for m_iter, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device).squeeze()  # TODO: #B,
                data = data.permute(0, 2, 1)  # TODO:B,3,N
                data = data[:,:,torch.randperm(data.shape[-1])]
                anchors = farthest_point_sample(data.transpose(1, 2), args.k_tilde)  # TODO: (B,k_tilde)
                for ii, cur_ppc in enumerate(range(args.k_tilde)):
                    random_ppc = extract_random(data, args.nr)  # B,3,nr Global partial point-cloud
                    patches_ppc = extract_patches(data, anchors[:, [cur_ppc]],
                                                  args.np)  # B,3,np Local partial point-cloud
                    curves_ppc = extract_curves(data, anchors[:, [cur_ppc]], args.m,
                                                args.nc)  # B,3,nc Local partial point-cloud
                    if args.use_ensemble_all and ii==0:
                        critical_ppc, _ = extract_discrete_critical(data, model_critical, args.na)  # B,3,na Global partial point-cloud
                    logits_random, _ = model_random(random_ppc)
                    logits_patches, _ = model_patches(patches_ppc)
                    logits_curves, _ = model_curves(curves_ppc)
                    if args.use_ensemble_all and ii==0:
                        logits_critical, _ = model_critical(critical_ppc)
                    if cur_ppc == 0:
                        tot_pred_curves = logits_curves.unsqueeze(-2)
                        tot_pred_patches = logits_patches.unsqueeze(-2)
                        tot_pred_random = logits_random.unsqueeze(-2)
                        if args.use_ensemble_all and ii==0:
                            tot_pred_critical = logits_critical.unsqueeze(-2)
                    else:
                        tot_pred_curves = torch.cat((tot_pred_curves, logits_curves.unsqueeze(-2)), dim=-2)  # BxTxC
                        tot_pred_patches = torch.cat((tot_pred_patches, logits_patches.unsqueeze(-2)), dim=-2)  # BxTxC
                        tot_pred_random = torch.cat((tot_pred_random, logits_random.unsqueeze(-2)), dim=-2)  # BxTxC
                        if args.use_ensemble_all and ii==0:
                            tot_pred_critical = torch.cat((tot_pred_critical, logits_critical.unsqueeze(-2)), dim=-2)  # BxTxC
                
                if args.use_ensemble_all:
                    tot_pred = torch.cat((tot_pred_curves, tot_pred_patches, tot_pred_random, tot_pred_critical), dim=-2)
                else:
                    tot_pred = torch.cat((tot_pred_curves, tot_pred_patches, tot_pred_random), dim=-2)
                logits = torch.mean(tot_pred, dim=-2)

                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            end = time.time()
            if split == "clean":
                io.cprint(f"Time(Total): {end-start}")
                io.cprint(f"Time(per sample): {(end-start)/m_iter}")
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}
    if args.use_ensemble_all:
        model = [model_patches, model_curves, model_random, model_critical]
    else:
        model = [model_patches, model_curves, model_random]
    eval_corrupt_wrapper(model, test_corrupt, {'args': args}, io)


def set_fixed_seed(args, fixed = -1):
    if fixed !=-1:
        args.seed = fixed 
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True

def test(args, io):
    import time
    # work on single sample per batch in order to adaptively select threshold per sample
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.use_ensemble:
        test_ensemble(args, io)
    else:
        args.test_batch_size = 1
        model = load_model(args, device)
        if args.is_vanilla:
            model = nn.DataParallel(model)
            model_path_critical = os.path.join('pretrained', f'{args.model}.t7')
        else:
            model_path_critical = os.path.join('pretrained',
                                                     f'{args.model}_importance_256_1024{"_wm" if args.use_wolfmix else ""}{"_pointguard" if args.is_pointguard else ""}.t7') if args.model_path_critical == "" else args.model_path_critical
        checkpoint = torch.load(model_path_critical, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        
        model.eval()
        def test_corrupt(args, split, model):
            with torch.no_grad():
                test_loader = DataLoader(ModelNetC(args=args, split=split),
                                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
                set_fixed_seed(args, 1)
                test_true = []
                test_pred = []
                start = time.time()
                for m_iter, (data, label) in enumerate(test_loader):
                    torch.cuda.empty_cache()
                    data, label = data.to(device), label.to(device).squeeze()
                    data = data.permute(0, 2, 1)
                    # rand the input, to ensure handling permutation of the points
                    data = data[:,:,torch.randperm(data.shape[-1])]
                    batch_size = data.size()[0]
                    num_points = data.size()[2]
                    if args.is_pointguard:
                        # PointGuard paper advocates 1000 ensemble members, each contains 16 elements
                        for ii in range(1000):
                            s = torch.randperm(data.shape[-1])
                            ppc_critical = data[:,:,s[:16]]
                            logits,_ = model(ppc_critical)
                            if ii == 0:
                                preds = logits.max(dim=1, keepdims=True)[1]
                            else:
                                preds = torch.cat((preds, logits.max(dim=1, keepdims=True)[1]), dim=1)
                        preds = torch.mode(preds, 1)[0]
                    elif args.is_vanilla:
                        logits,_ = model(data)
                        preds = logits.max(dim=1)[1]         
                    else:                
                        ppc_critical, m_critical = extract_discrete_critical(data, model)
                        logits,_ = model(ppc_critical)
                        preds = logits.max(dim=1)[1]
                    test_true.append(label.unsqueeze(0).cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                
                end = time.time()
                import scipy.io as sio
                if split == "clean":
                    io.cprint(f"Time(Total): {end-start}")
                    io.cprint(f"Time(per sample): {(end-start)/m_iter}")

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}
    
        eval_corrupt_wrapper(model, test_corrupt, {'args': args}, io)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='leanrd2', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['curvenet', 'dgcnn', 'dgcnn_v1', 'dgcnn_v2', 'dgcnn_v3', 'dgcnn_v5', 'gdanet', 'rpc',
                                 'pct', 'custom_model', 'pointnet', 'paconv'],
                        help='Model Architecture to use as a basic model for random, patches and curves')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--na', type=int, default=600, metavar='N', help='number of remaining points')
    parser.add_argument('--npoints', type=int, default=8, metavar='N', help='number of anchor points for EPiC')
    
    
    parser.add_argument('--model_path_patches', type=str, default='', metavar='N', help='Pretrained model path for patches testing')
    parser.add_argument('--model_path_random', type=str, default='', metavar='N', help='Pretrained model path for random testing')
    parser.add_argument('--model_path_curves', type=str, default='', metavar='N', help='Pretrained model path for curves testing')
    parser.add_argument('--model_path_critical', type=str, default='', metavar='N', help='Pretrained model path for critical testing')
    
    parser.add_argument('--use_ensemble', action='store_true', help='standalone or ensemble')
    parser.add_argument('--use_ensemble_all', action='store_true', help='ensemble using also critical')
    
    # EPiC parameters
    parser.add_argument('--nc', type=int, default=512, metavar='N', help='number of points per curve')
    parser.add_argument('--np', type=int, default=512, metavar='N', help='number of points per patch')
    parser.add_argument('--nr', type=int, default=128, metavar='N', help='number of points per random')
    parser.add_argument('--m', type=int, default=20, metavar='N', help='number of neighbors for curve random picking')
    parser.add_argument('--k_tilde', type=int, default=4, help='number of anchor points in the shape')
                        
    parser.add_argument('--train_random', action='store_true', help='enable for random ppc training')
    parser.add_argument('--train_patches', action='store_true', help='enable for patches ppc training')    
    parser.add_argument('--train_adaboost', action='store_true', help='enable for adaboost ppc training')  
                        
                        
    parser.add_argument('--is_retrain', action='store_true', help='if to retrain')
    parser.add_argument('--use_wolfmix', action='store_true', help='if to use wm models')
    parser.add_argument('--is_pointguard', action='store_true', help='if to train or test as pointguard')
    parser.add_argument('--is_lpf', action='store_true', help='if to train or test as low-pass filter')
    parser.add_argument('--is_vanilla', action='store_true', help='if to test as vanilla')
    # rsmix params
    parser.add_argument('--rdscale', action='store_true', help='random scaling data augmentation')
    parser.add_argument('--shift', action='store_true', help='random shift data augmentation')
    parser.add_argument('--shuffle', action='store_true', help='random shuffle data augmentation')
    parser.add_argument('--rot', action='store_true', help='random rotation augmentation')
    parser.add_argument('--jitter', action='store_true', help='jitter augmentation')
    parser.add_argument('--rddrop', action='store_true', help='random point drop data augmentation')
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=1.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512, help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--normal', action='store_true', help='use normal')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    # pointwolf params
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
    parser.add_argument('--w_sample_type', type=str, default='fps', help='Sampling method for anchor point, option : (fps, random)')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25, help='Maximum translation range of local transformation')                      
    args = parser.parse_args()

    _init_()
    io = IOStream(f'{"test" if args.eval else "train"}_{args.model}{"_wm" if args.use_wolfmix else ""}{"_pg" if args.is_pointguard else ""}{"_epic" if args.use_ensemble else ""}.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
