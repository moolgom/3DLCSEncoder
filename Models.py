import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

    
###############################################################################
# Fully Factorized Prior Model
# "Deep Implicit Volume Compression"
# https://arxiv.org/abs/2005.08877
###############################################################################

class FactorizedPriorModel(nn.Module):
    def __init__(self, M=192):
        super().__init__()
    
        # entropy bottleneck
        self.entropy_bottleneck = EntropyBottleneck(M)
        
        # encoder
        self.g_a = nn.Sequential(
            nn.Conv3d(1, M, kernel_size=3, stride=2, padding=1), # 8x8x8 -> 4x4x4
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(M, M, kernel_size=3, stride=2, padding=1), # 4x4x4 -> 2x2x2
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(M, M, kernel_size=3, stride=2, padding=1), # 2x2x2 -> 1x1x1
        )
        
        # decoder
        self.g_s = nn.Sequential(
            nn.ConvTranspose3d(M, M, kernel_size=3, stride=2, output_padding=1, padding=1), # 1x1x1 -> 2x2x2
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(M, M, kernel_size=3, stride=2, output_padding=1, padding=1), # 2x2x2 -> 4x4x4
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(M, M, kernel_size=3, stride=2, output_padding=1, padding=1), # 4x4x4 -> 8x8x8
            nn.LeakyReLU(inplace=True),
        )
        
        self.head_sign = nn.Conv3d(M, 1, kernel_size=3, stride=1, padding=1)
        self.head_magn = nn.Conv3d(M, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        y = self.g_a(x) # latent representations
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        x_hat = self.g_s(y_hat) # 복호화 
        sign = self.head_sign(x_hat).clamp( 0., 1.) # 부호 예측
        magn = self.head_magn(x_hat).clamp(-1., 1.) # 크기 예측 
        
        return sign, magn, y_likelihoods
    
    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        if len(y_hat.shape) == 4:
            y_hat = y_hat[:, :, :, :, None] # 
        x_hat = self.g_s(y_hat)
        sign = self.head_sign(x_hat).clamp( 0., 1.) # 부호 예측
        magn = self.head_magn(x_hat).clamp(-1., 1.) # 크기 예측 
        
        return sign, magn
    

# Zero-Mean Scale HyperPrior Model
class ScaleHyperPriorModel(FactorizedPriorModel):
    def __init__(self, M=192, N=64):
        super().__init__()
        
        # entropy bottleneck
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        # hyper-encoder
        self.h_a = nn.Sequential(
            nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, N, kernel_size=1, stride=1, padding=0),
        )
        
        # hyper-decoder
        self.h_s = nn.Sequential(
            nn.Conv3d(N, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        y = self.g_a(x) # latent representations
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        
        sign = self.head_sign(x_hat).clamp( 0., 1.) # 부호 예측
        magn = self.head_magn(x_hat).clamp(-1., 1.) # 크기 예측 
        
        return sign, magn, y_likelihoods, z_likelihoods
        

# Nonzero-Mean Scale HyperPrior Model
class MeanScaleHyperPriorModel(FactorizedPriorModel):
    def __init__(self, M=192, N=64):
        super().__init__()
        
        # entropy bottleneck
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        # hyper-encoder
        self.h_a = nn.Sequential(
            nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, N, kernel_size=1, stride=1, padding=0),
        )
        
        # hyper-decoder
        self.h_s = nn.Sequential(
            nn.Conv3d(N, M, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M, M * 3 // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(M * 3 // 2, M * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        y = self.g_a(x) # latent representations
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        
        sign = self.head_sign(x_hat).clamp( 0., 1.) # 부호 예측
        magn = self.head_magn(x_hat).clamp(-1., 1.) # 크기 예측 
        
        return sign, magn, y_likelihoods, z_likelihoods
        
    
# Zero-Mean Scale HyperPrior Model + Latent Code Selection
class HyperLCS(ScaleHyperPriorModel):
    def __init__(self, M=192, N=64):
        super().__init__()
    
        self.head_impm = nn.Conv3d(M, M, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        y = self.g_a(x) # latent representations
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        #scales_hat = self.h_s(z_hat)
        
        f0 = self.h_s[0](z_hat) # Conv3d
        f1 = self.h_s[1](f0)    # ReLU
        f2 = self.h_s[2](f1)    # Conv3d        
        f3 = self.h_s[3](f2)    # ReLU
        f4 = self.h_s[4](f3)    # Conv3d
        scales_hat = self.h_s[5](f4) # ReLU
        impts = self.head_impm(f3)
        
        impts = impts.clamp(-0.5, 0.5)
        impts_noise = torch.nn.init.uniform_(torch.zeros_like(impts), 0., 1.)
        impts_sample = impts + impts_noise
        # "straight-through" quantizer
        round_noise = (impts_sample - torch.round(impts_sample)).detach()
        impts_mask = impts_sample - round_noise
        
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat * impts_mask)
        
        sign = self.head_sign(x_hat).clamp( 0., 1.) # 부호 예측
        magn = self.head_magn(x_hat).clamp(-1., 1.) # 크기 예측 
        
        return sign, magn, y_likelihoods, z_likelihoods, impts_mask
    

# Nonzero-Mean Scale HyperPrior Model + Latent Code Selection
class HyperLCSMean(MeanScaleHyperPriorModel):
    def __init__(self, M=192, N=64):
        super().__init__()
    
        self.head_impm = nn.Conv3d(M * 3 // 2, M, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        y = self.g_a(x) # latent representations
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        #scales_hat = self.h_s(z_hat)
        
        f0 = self.h_s[0](z_hat) # Conv3d
        f1 = self.h_s[1](f0)    # ReLU
        f2 = self.h_s[2](f1)    # Conv3d        
        f3 = self.h_s[3](f2)    # ReLU
        f4 = self.h_s[4](f3)    # Conv3d
        gaussian_params = self.h_s[5](f4) # ReLU
        impts = self.head_impm(f3)
        
        impts = impts.clamp(-0.5, 0.5)
        impts_noise = torch.nn.init.uniform_(torch.zeros_like(impts), 0., 1.)
        impts_sample = impts + impts_noise
        # "straight-through" quantizer
        round_noise = (impts_sample - torch.round(impts_sample)).detach()
        impts_mask = impts_sample - round_noise
        
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat * impts_mask)
        
        sign = self.head_sign(x_hat).clamp( 0., 1.) # 부호 예측
        magn = self.head_magn(x_hat).clamp(-1., 1.) # 크기 예측 
        
        return sign, magn, y_likelihoods, z_likelihoods, impts_mask
