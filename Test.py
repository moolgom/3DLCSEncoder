import trimesh
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from sklearn.neighbors import NearestNeighbors

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchac import torchac

from Models import *
from VolumeDataset import TSDFVolumeDataset

    

MODEL_NAMES = [
    'FactorizedPriorModel',         # Fully Factorized Prior Model
    'ScaleHyperPriorModel',         # Zero-Mean Scale HyperPrior Model
    'MeanScaleHyperPriorModel',     # Nonzero-Mean Scale HyperPrior Model
    'UniScaleHyperPriorModel',      # Zero-Mean Scale HyperPrior Model + Latent Code Selection
    'UniMeanScaleHyperPriorModel',  # Nonzero-Mean Scale HyperPrior Model + Latent Code Selection
]


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
scale_table = torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))


test_dataset = TSDFVolumeDataset(augment=False, dataset='test')
num_blocks = len(test_dataset)
batch_size = num_blocks
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

M = 192
N = 64

for model_name in MODEL_NAMES:
    for RD_point in range(0, 12):
        
        tsdf, mask, sign, magn = next(iter(test_dataloader))
        tsdf = tsdf.cuda()
        mask = mask.cuda()
        sign = sign.cuda()
        magn = magn.cuda() # topology mask
            
        if model_name == 'FactorizedPriorModel':
            with torch.no_grad():
                model = FactorizedPriorModel(M).to('cuda')
                model.load_state_dict(torch.load('./TrainedModels/%s/M%d_R%2d.pth' % (model_name, M, RD_point),
                                                 map_location=torch.device('cuda')), strict=True)
                model.entropy_bottleneck.update()
                model.eval()
            
                # Compress (y)
                y = model.g_a(tsdf)
                y_enc = model.entropy_bottleneck.compress(y)
                
                # Decompress (y)
                y_hat = model.entropy_bottleneck.decompress(y_enc, y.size()[-2:])
                y_hat = y_hat.reshape(y.shape)
                x_hat = model.g_s(y_hat)
                sign_dec = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
                magn_dec = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
                
                # Compress (sign)
                symbols = sign.reshape(batch_size, 8*8*8, 1)
                symbols = symbols.type(torch.int16)
                probs = sign_dec.reshape(batch_size, 8*8*8, 1)
                cdf_float = torch.zeros((batch_size, 8*8*8, 1, 3), dtype=torch.float32)
                cdf_float[:, :, :, 0] = 0.0
                cdf_float[:, :, :, 1] = 1.0 - probs
                cdf_float[:, :, :, 2] = 1.0
                symbols = symbols.cpu()
                cdf_float = cdf_float.cpu()
                sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
                
                # Decompress (sign)
                symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
                #assert symbols_out.equal(symbols)
                sign_dec = symbols_out.reshape(batch_size, 1, 8, 8, 8)
                assert sign_dec.equal(sign.to(torch.int16).cpu())
                
                # Final Output
                tsdf_dec = (sign_dec * 2.0 - 1.0) * torch.abs(magn_dec.cpu())
                
        elif model_name == 'ScaleHyperPriorModel':
            with torch.no_grad():
                model = ScaleHyperPriorModel(M, N).to('cuda')
                model.load_state_dict(torch.load('./TrainedModels/%s/M%d_N%d_R%d.pth' % (model_name, M, N, RD_point),
                                                 map_location=torch.device('cuda')), strict=True)
                model.entropy_bottleneck.update()
                model.gaussian_conditional.update_scale_table(scale_table)
                model.eval()
            
                # Compress (z)
                y = model.g_a(tsdf)
                z = model.h_a(torch.abs(y))
                z_enc = model.entropy_bottleneck.compress(z)
            
                # Decompress (z)
                z_hat = model.entropy_bottleneck.decompress(z_enc, z.size()[-2:])
                z_hat = z_hat.reshape(z.shape)
                scales_hat = model.h_s(z_hat)
                indexes = model.gaussian_conditional.build_indexes(scales_hat)
                
                # Compress (y)
                y_enc = model.gaussian_conditional.compress(y, indexes)
                
                # Decompress (y)
                y_hat = model.gaussian_conditional.decompress(y_enc, indexes, torch.float32)
                x_hat = model.g_s(y_hat)
                sign_dec = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
                magn_dec = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
                
                # Compress (sign)
                symbols = sign.reshape(batch_size, 8*8*8, 1)
                symbols = symbols.type(torch.int16)
                probs = sign_dec.reshape(batch_size, 8*8*8, 1)
                cdf_float = torch.zeros((batch_size, 8*8*8, 1, 3), dtype=torch.float32)
                cdf_float[:, :, :, 0] = 0.0
                cdf_float[:, :, :, 1] = 1.0 - probs
                cdf_float[:, :, :, 2] = 1.0
                symbols = symbols.cpu()
                cdf_float = cdf_float.cpu()
                sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
                
                # Decompress (sign)
                symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
                #assert symbols_out.equal(symbols)
                sign_dec = symbols_out.reshape(batch_size, 1, 8, 8, 8)
                assert sign_dec.equal(sign.to(torch.int16).cpu())
                
                # Final Output
                tsdf_dec = (sign_dec * 2.0 - 1.0) * torch.abs(magn_dec.cpu())
            
        elif model_name == 'UniScaleHyperPriorModel':
            with torch.no_grad():
                model = UniScaleHyperPriorModel(M, N).to('cuda')
                model.load_state_dict(torch.load('./TrainedModels/%s/M%d_N%d_R%d.pth' % (model_name, M, N, RD_point),
                                             map_location=torch.device('cuda')), strict=True) # 구글 기학습 모델을 사용
                model.entropy_bottleneck.update()
                model.gaussian_conditional.update_scale_table(scale_table)
                model.eval()
                
                # Compress (z)
                y = model.g_a(tsdf)
                z = model.h_a(torch.abs(y))
                z_enc = model.entropy_bottleneck.compress(z)
            
                # Decompress (z)
                z_hat = model.entropy_bottleneck.decompress(z_enc, z.size()[-2:])
                z_hat = z_hat.reshape(z.shape)
                
                f0 = model.h_s[0](z_hat) # Conv3d
                f1 = model.h_s[1](f0)    # ReLU
                f2 = model.h_s[2](f1)    # Conv3d        
                f3 = model.h_s[3](f2)    # ReLU
                f4 = model.h_s[4](f3)    # Conv3d
                scales_hat = model.h_s[5](f4) # ReLU
                impts = model.head_impm(f3)
                
                impts_mask = torch.round(impts + 0.5)
                impts_mask = impts_mask.clamp(0.0, 1.0)
                impts_mask = impts_mask.bool() # Latent Code Selection Mask
                indexes = model.gaussian_conditional.build_indexes(scales_hat)
                
                # Compress (y)
                y_enc = model.gaussian_conditional.compress(torch.masked_select(y, impts_mask).reshape(1, -1),
                                                            torch.masked_select(indexes, impts_mask).reshape(1, -1))
                
                # Decompress (y)
                y_hat1 = model.gaussian_conditional.decompress(y_enc, 
                                                               torch.masked_select(indexes, impts_mask).reshape(1, -1), torch.float32)
                
                y_hat2 = torch.zeros_like(y)
                y_hat2[impts_mask] = y_hat1
                x_hat = model.g_s(y_hat2)
                sign_dec = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
                magn_dec = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
                
                # Compress (sign)
                symbols = sign.reshape(batch_size, 8*8*8, 1)
                symbols = symbols.type(torch.int16)
                probs = sign_dec.reshape(batch_size, 8*8*8, 1)
                cdf_float = torch.zeros((batch_size, 8*8*8, 1, 3), dtype=torch.float32)
                cdf_float[:, :, :, 0] = 0.0
                cdf_float[:, :, :, 1] = 1.0 - probs
                cdf_float[:, :, :, 2] = 1.0
                symbols = symbols.cpu()
                cdf_float = cdf_float.cpu()
                sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
                
                # Decompress (sign)
                symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
                #assert symbols_out.equal(symbols)
                sign_dec = symbols_out.reshape(batch_size, 1, 8, 8, 8)
                assert sign_dec.equal(sign.to(torch.int16).cpu())
                
                # Final Output
                tsdf_dec = (sign_dec * 2.0 - 1.0) * torch.abs(magn_dec.cpu())
                
        elif model_name == 'MeanScaleHyperPriorModel':
            with torch.no_grad():
                model = MeanScaleHyperPriorModel(M, N).to('cuda')
                model.load_state_dict(torch.load('./TrainedModels/%s/M%d_N%d_R%d.pth' % (model_name, M, N, RD_point),
                                             map_location=torch.device('cuda')), strict=True) # 구글 기학습 모델을 사용
                model.entropy_bottleneck.update()
                model.gaussian_conditional.update_scale_table(scale_table)
                model.eval()
                
                # Compress (z)
                y = model.g_a(tsdf)
                z = model.h_a(torch.abs(y))
                z_enc = model.entropy_bottleneck.compress(z)
            
                # Decompress (z)
                z_hat = model.entropy_bottleneck.decompress(z_enc, z.size()[-2:])
                z_hat = z_hat.reshape(z.shape)
                gaussian_params = model.h_s(z_hat)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = model.gaussian_conditional.build_indexes(scales_hat)
                
                # Compress (y)
                y_enc = model.gaussian_conditional.compress(y, indexes, means=means_hat)
                
                # Decompress (y)
                y_hat = model.gaussian_conditional.decompress(y_enc, indexes, dtype=torch.float32, means=means_hat)
                x_hat = model.g_s(y_hat)
                sign_dec = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
                magn_dec = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
                
                # Compress (sign)
                symbols = sign.reshape(batch_size, 8*8*8, 1)
                symbols = symbols.type(torch.int16)
                probs = sign_dec.reshape(batch_size, 8*8*8, 1)
                cdf_float = torch.zeros((batch_size, 8*8*8, 1, 3), dtype=torch.float32)
                cdf_float[:, :, :, 0] = 0.0
                cdf_float[:, :, :, 1] = 1.0 - probs
                cdf_float[:, :, :, 2] = 1.0
                symbols = symbols.cpu()
                cdf_float = cdf_float.cpu()
                sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
                
                # Decompress (sign)
                symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
                #assert symbols_out.equal(symbols)
                sign_dec = symbols_out.reshape(batch_size, 1, 8, 8, 8)
                assert sign_dec.equal(sign.to(torch.int16).cpu())
                
                # Final Output
                tsdf_dec = (sign_dec * 2.0 - 1.0) * torch.abs(magn_dec.cpu())
                
        elif model_name == 'UniMeanScaleHyperPriorModel':
            with torch.no_grad():
                model = UniMeanScaleHyperPriorModel(M, N).to('cuda')
                model.load_state_dict(torch.load('./TrainedModels/%s/M%d_N%d_R%d.pth' % (model_name, M, N, RD_point),
                                             map_location=torch.device('cuda')), strict=True) # 구글 기학습 모델을 사용
                model.entropy_bottleneck.update()
                model.gaussian_conditional.update_scale_table(scale_table)
                model.eval()
                
                # Compress (z)
                y = model.g_a(tsdf)
                z = model.h_a(torch.abs(y))
                z_enc = model.entropy_bottleneck.compress(z)
            
                # Decompress (z)
                z_hat = model.entropy_bottleneck.decompress(z_enc, z.size()[-2:])
                z_hat = z_hat.reshape(z.shape)
                
                f0 = model.h_s[0](z_hat) # Conv3d
                f1 = model.h_s[1](f0)    # ReLU
                f2 = model.h_s[2](f1)    # Conv3d        
                f3 = model.h_s[3](f2)    # ReLU
                f4 = model.h_s[4](f3)    # Conv3d
                gaussian_params = model.h_s[5](f4) # ReLU
                impts = model.head_impm(f3)
                
                impts_mask = torch.round(impts + 0.5)
                impts_mask = impts_mask.clamp(0.0, 1.0)
                impts_mask = impts_mask.bool()
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = model.gaussian_conditional.build_indexes(scales_hat)
                
                # Compress (y)
                y_enc = model.gaussian_conditional.compress(torch.masked_select(y, impts_mask).reshape(1, -1),
                                                            torch.masked_select(indexes, impts_mask).reshape(1, -1),
                                                            means=torch.masked_select(means_hat, impts_mask).reshape(1, -1))
                
                # Decompress (y)
                y_hat1 = model.gaussian_conditional.decompress(y_enc, 
                                                               torch.masked_select(indexes, impts_mask).reshape(1, -1), 
                                                               dtype=torch.float32,
                                                               means=torch.masked_select(means_hat, impts_mask).reshape(1, -1))
                
                y_hat2 = torch.zeros_like(y)
                y_hat2[impts_mask] = y_hat1
                x_hat = model.g_s(y_hat2)
                sign_dec = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
                magn_dec = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
                
                # Compress (sign)
                symbols = sign.reshape(batch_size, 8*8*8, 1)
                symbols = symbols.type(torch.int16)
                probs = sign_dec.reshape(batch_size, 8*8*8, 1)
                cdf_float = torch.zeros((batch_size, 8*8*8, 1, 3), dtype=torch.float32)
                cdf_float[:, :, :, 0] = 0.0
                cdf_float[:, :, :, 1] = 1.0 - probs
                cdf_float[:, :, :, 2] = 1.0
                symbols = symbols.cpu()
                cdf_float = cdf_float.cpu()
                sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
                
                # Decompress (sign)
                symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
                #assert symbols_out.equal(symbols)
                sign_dec = symbols_out.reshape(batch_size, 1, 8, 8, 8)
                assert sign_dec.equal(sign.to(torch.int16).cpu())
                
                # Final Output
                tsdf_dec = (sign_dec * 2.0 - 1.0) * torch.abs(magn_dec.cpu())
                       
        # Reconstructing Voxel Blocks into a Volume
        volume_dim  = round(num_blocks ** (1/3) * 8)
        volume = np.zeros((volume_dim, volume_dim, volume_dim), dtype=np.float32)
        index = 0
        for x in range(0, volume_dim, 8):
            for y in range(0, volume_dim, 8):
                for z in range(0, volume_dim, 8):
                    volume[x:x+8, y:y+8, z:z+8] = tsdf_dec[index, :, :, :]
                    index += 1
        
        # Use marching cubes to obtain the surface mesh 
        verts_dec, faces_dec, normals_dec, values_dec = measure.marching_cubes(volume, 0)
        mesh = trimesh.Trimesh(vertices=verts_dec, faces=faces_dec, vertex_normals=normals_dec)
        _ = mesh.export('Buddha_%s_%d.ply' % (model_name, RD_point))
