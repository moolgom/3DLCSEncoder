import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Models import HyperLCS
from VolumeDataset import TSDFVolumeDataset

if  torch.cuda.device_count() > 1:
    from pynvml import *
    # GPU auto allocation
    best_gpu = 0
    most_free_memory = 0
    nvmlInit()
    
    for gpu_index in range(3):
        h = nvmlDeviceGetHandleByIndex(gpu_index)
        info = nvmlDeviceGetMemoryInfo(h)
        free_memory = info.free
        if most_free_memory < free_memory:
            most_free_memory = free_memory
            best_gpu = gpu_index
            
    print('Best GPU: ', best_gpu)
    torch.cuda.set_device(best_gpu)
    torch.cuda.empty_cache() # Empty any cache, not sure this helps, we try waht we can 

if not os.path.exists('./TrainedModels'):
    os.mkdir('./TrainedModels')

if not os.path.exists('./TrainedModels/HyperLCS'):
    os.mkdir('./TrainedModels/HyperLCS')
    
    
###############################################################################
# Hyperparameters
##############################################################################

def get_RD_balance_Factor(p):
    # Deep Implicit Volume Compression (https://arxiv.org/abs/2005.08877)
    mu = (p / 10) * math.log10(200000)
    lmbd = 1.0 / (10 ** mu)
    return lmbd

def train_single_rate_point(rate_point):
    num_max_epochs = 5000
    batch_size = 256
    main_lr = 1e-4
    aux_lr = 1e-3 
    EPS = 2**-18
    M = 192
    N = 64
    
    RD_balance = get_RD_balance_Factor(rate_point)
    
    ''' Training Dataset '''
    tr_dataset = TSDFVolumeDataset(augment=True, dataset='train')
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    num_tr_batches = len(tr_dataloader)
    
    ''' Validation Dataset '''
    val_dataset = TSDFVolumeDataset(augment=False, dataset='val')
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    num_val_batches = len(val_dataloader)
    
    ''' Compression Model '''
    model = HyperLCS(M, N).to('cuda')
    
    ''' Optimizer '''
    main_parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters  = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    main_optimizer  = optim.Adam(main_parameters, lr=main_lr)
    aux_optimizer   = optim.Adam(aux_parameters, lr=aux_lr)
    main_scheduler  = optim.lr_scheduler.ReduceLROnPlateau(main_optimizer, patience=20, verbose='True')
    
    
    for e in range(num_max_epochs):
        ''' Training Phase '''
        model.train()
        avg_dist_x = torch.tensor([0.0]).cuda()
        avg_bpv_s  = torch.tensor([0.0]).cuda()
        avg_bpv_yz = torch.tensor([0.0]).cuda()
        avg_selection_ratio = torch.tensor([0.0]).cuda()
        for batch_idx, samples in enumerate(tr_dataloader):
            main_optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            tsdf, mask, sign, magn = samples
            tsdf = tsdf.cuda()
            mask = mask.cuda()
            sign = sign.cuda()
            magn = magn.cuda()
            B, _, H, W, L = tsdf.size()
            num_voxels = B * H * W * L
            
            sign_pred, magn_pred, y_likelihoods, z_likelihoods, impts_mask = model(tsdf)
        
            rate_y = (torch.log(y_likelihoods) * impts_mask).sum() / -math.log(2)
            rate_z = torch.log(z_likelihoods).sum() / -math.log(2)
            rate_s = (sign * torch.log(sign_pred + EPS) + (1.0 - sign) * torch.log((1.0 - sign_pred) + EPS)).sum() / (-math.log(2))
            dist_x = (mask * ((torch.abs(magn_pred) - magn) ** 2)).sum() 
            
            avg_dist_x          += dist_x / (num_tr_batches * num_voxels)
            avg_bpv_s           += rate_s / (num_tr_batches * num_voxels)
            avg_bpv_yz          += (rate_y + rate_z) / (num_tr_batches * num_voxels)
            avg_selection_ratio += (impts_mask.sum() / impts_mask.numel())  / num_tr_batches
            
            RD_loss  = RD_balance * (rate_y + rate_z + rate_s) + dist_x
            aux_loss = model.entropy_bottleneck.loss()
            
            RD_loss.backward()
            main_optimizer.step()
            
            aux_loss.backward()
            aux_optimizer.step()
            
        print('[TRAIN-p=%d, e:%04d] D: %10.8f,  R_yz: %10.8f,  R_s: %10.8f,  [LCS Ratio: %10.8f]' % (rate_point, e, avg_dist_x.item(), avg_bpv_yz.item(), avg_bpv_s.item(), avg_selection_ratio.item()), end='\t\t')
            
        ''' Valdiation Pahse '''
        model.eval()
        avg_dist_x = torch.tensor([0.0]).cuda()
        avg_bpv_s  = torch.tensor([0.0]).cuda()
        avg_bpv_yz = torch.tensor([0.0]).cuda()
        avg_selection_ratio = torch.tensor([0.0]).cuda()
        with torch.no_grad():
            for batch_idx, samples in enumerate(val_dataloader):
                tsdf, mask, sign, magn = samples
                tsdf = tsdf.cuda()
                mask = mask.cuda()
                sign = sign.cuda()
                magn = magn.cuda()
                B, _, H, W, L = tsdf.size()
                num_voxels = B * H * W * L 
        
                sign_pred, magn_pred, y_likelihoods, z_likelihoods, impts_mask = model(tsdf)
            
                rate_y = (torch.log(y_likelihoods) * impts_mask).sum() / -math.log(2)
                rate_z = torch.log(z_likelihoods).sum() / -math.log(2)
                rate_s = (sign * torch.log(sign_pred + EPS) + (1.0 - sign) * torch.log((1.0 - sign_pred) + EPS)).sum() / (-math.log(2))
                dist_x = (mask * ((torch.abs(magn_pred) - magn) ** 2)).sum() 
                
                avg_dist_x          += dist_x / (num_tr_batches * num_voxels)
                avg_bpv_s           += rate_s / (num_tr_batches * num_voxels)
                avg_bpv_yz          += (rate_y + rate_z) / (num_tr_batches * num_voxels)
                avg_selection_ratio += (impts_mask.sum() / impts_mask.numel())  / num_tr_batches
                
        avg_RD_loss = RD_balance * (avg_bpv_yz + avg_bpv_s) + avg_dist_x
        main_scheduler.step(avg_RD_loss)
            
        print('[VAL] D: %10.8f,  R_yz: %10.8f,  R_s: %10.8f,  [LCS Ratio: %10.8f]' % (avg_dist_x.item(), avg_bpv_yz.item(), avg_bpv_s.item(), avg_selection_ratio.item()))
                    
        if main_optimizer.param_groups[0]['lr'] <= 1e-06:
            break
        
    torch.save(model.state_dict(), './TrainedModels/HyperLCS/M%d_N%d_R%d.pth' % (M, N, RD_point))
            
            

###############################################################################   
###############################################################################   
###############################################################################   

def main():
    for rate_point in range(0, 10):
        train_single_rate_point(rate_point)

if __name__ == "__main__":
	main()
                
            
            
            
            
        
        
        
        
        
        
        
        
        
        
            
            