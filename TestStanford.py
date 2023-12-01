import os
import time
import zlib
import trimesh
import numpy as np
from pysdf import SDF
from numba import njit
from skimage import measure

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchac import torchac

from Models import *

def mesh_to_TSDFVolume(mesh_path, voxel_size):
    print("Converting meth to TSDF volume...", end='')
    mesh = trimesh.load(mesh_path)
    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    min_bound = verts.min(0) - voxel_size * 4
    max_bound = verts.max(0)

    # Initial grid size
    grid_size = np.ceil((max_bound - min_bound) / voxel_size)

    def make_divisible_by_8(n):
        """ Make the given number a multiple of eight """
        return n + (8 - n % 8) if n % 8 != 0 else n
    
    # Adjust the grid size to be a multiple of 8
    grid_size = np.array([make_divisible_by_8(n) for n in grid_size])
    grid_size = grid_size.astype(np.uint32)

    # Calculate query points 
    x_range = min_bound[0] + np.arange(grid_size[0]) * voxel_size
    y_range = min_bound[1] + np.arange(grid_size[1]) * voxel_size
    z_range = min_bound[2] + np.arange(grid_size[2]) * voxel_size
    query_points = np.stack(np.meshgrid(x_range, y_range, z_range, indexing='ij'), -1)
    query_points_list = query_points.reshape(-1, 3).tolist()
    
    f = SDF(verts, faces);
    sdf_values_list = f(query_points_list)
    sdf_values = np.array(sdf_values_list).reshape(grid_size[0], grid_size[1], grid_size[2])
    
    sdf_trunc = voxel_size * 10.0
    tsdf_values = np.minimum(1.0, sdf_values / sdf_trunc)
    tsdf_values = np.maximum(-1.0, tsdf_values)
    print("Done.")
    return tsdf_values, min_bound, grid_size


@njit
def create_topology_mask(tsdf_volume):
    L, M, N = tsdf_volume.shape
    sign_volume = (tsdf_volume >= 0.0)
    mask_x = np.zeros_like(tsdf_volume, dtype=np.uint8)
    mask_y = np.zeros_like(tsdf_volume, dtype=np.uint8)
    mask_z = np.zeros_like(tsdf_volume, dtype=np.uint8)

    # Iterate through each voxel (excluding the borders)
    for x in range(0, L-1):
        for y in range(0, M-1):
            for z in range(0, N-1):
                current_voxel   = sign_volume[x,   y,   z]
                right_voxel     = sign_volume[x+1, y,   z]
                top_voxel       = sign_volume[x,   y+1, z]
                back_voxel      = sign_volume[x,   y,   z+1]
                
                if current_voxel != right_voxel:
                    mask_x[x,   y, z] = 1
                    mask_x[x+1, y, z] = 1
                    
                if current_voxel != top_voxel:
                    mask_y[x, y,   z] = 1
                    mask_y[x, y+1, z] = 1
                    
                if current_voxel != back_voxel:
                    mask_z[x, y, z] = 1
                    mask_z[x, y, z+1] = 1
    
    
    mask_cube = np.zeros_like(tsdf_volume, dtype=np.uint8)
    
    # Iterate through each voxel (excluding the borders)
    for x in range(0, L-1):
        for y in range(0, M-1):
            for z in range(0, N-1):
                current_cube = sign_volume[x:x+2, y:y+2, z:z+2]
                if current_cube.sum() != 8 and current_cube.sum() != 0:
                    mask_cube[x:x+2, y:y+2, z:z+2] = 1
                
    return mask_x, mask_y, mask_z, mask_cube
                    
                
                
                

def reshape_volume_to_blocks(volume, block_size=(8, 8, 8)):
    L, M, N = volume.shape
    small_L, small_M, small_N = block_size
    
    # Check if the TSDF block dimensions are divisible by the block size
    if L % small_L != 0 or M % small_M != 0 or N % small_N != 0:
        raise ValueError("TSDF block dimensions must be divisible by the block size.")
        
    # Reshape the array
    blocks = volume.reshape(L // small_L, small_L, M // small_M, small_M, N // small_N, small_N)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5)
    blocks = blocks.reshape(-1, 1, small_L, small_M, small_N)

    return blocks

def merge_blocks_to_volume(blocks, original_shape, block_size=(8, 8, 8)):
    L, M, N = original_shape
    small_L, small_M, small_N = block_size

    # Check if the original TSDF block dimensions are divisible by the block size
    if L % small_L != 0 or M % small_M != 0 or N % small_N != 0:
        raise ValueError("Original TSDF block dimensions must be divisible by the block size.")

    # Reshape and transpose the array back to the original format
    volume = blocks.reshape(L // small_L, M // small_M, N // small_N, small_L, small_M, small_N)
    volume = volume.transpose(0, 3, 1, 4, 2, 5)
    volume = volume.reshape(L, M, N)

    return volume


def Test_Factorizerd(input_mesh, rate_point, tsdf_volume, volume_origin, voxel_size, batch_size=512):
    model = FactorizedPriorModel(192).to('cuda')
    model.load_state_dict(torch.load('./TrainedModels/FactorizedPriorModel/M%d_R%2d.pth' % (192, rate_point), map_location=torch.device('cuda')), strict=True)
    model.entropy_bottleneck.update()
    model.eval()
    
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    num_blocks = tsdf_blocks.shape[0]
    
    print("Compression using Factorized Model p=%d" % rate_point)
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_blocks, batch_size):
            
            if (i+batch_size) <= num_blocks:
                x = tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = tsdf_blocks[i::, :, :, :, :]
            current_batck_size = x.shape[0]
            
            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (y)
            y = model.g_a(x)
            y_enc = model.entropy_bottleneck.compress(y)
            
            # Decompress (y)
            y_hat = model.entropy_bottleneck.decompress(y_enc, y.size()[-2:])
            y_hat = y_hat.reshape(y.shape)
            x_hat = model.g_s(y_hat)
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batck_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batck_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batck_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batck_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_blocks:
                tsdf_blocks_out[i:i+batch_size, :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[i::, :, :, :, :] = xhat.detach().cpu().numpy()
    end_time = time.time() 
    duration = end_time - start_time  
    print(f"Duration: {duration} seconds")

    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_Factorized_%d.ply' % (input_mesh, rate_point))
           
    
def Test_Hyperprior(input_mesh, rate_point, tsdf_volume, volume_origin, voxel_size, batch_size=512):
    # From Balle's tensorflow compression examples
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64
    scale_table = torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))
    
    model = ScaleHyperPriorModel(192, 64).to('cuda')
    model.load_state_dict(torch.load('./TrainedModels/ScaleHyperPriorModel/M%d_N%d_R%d.pth' % (192, 64, rate_point),
                                     map_location=torch.device('cuda')), strict=True)
    model.entropy_bottleneck.update()
    model.gaussian_conditional.update_scale_table(scale_table)
    model.eval()
    
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    num_blocks = tsdf_blocks.shape[0]
    
    print("Compression using Hyperprior Model p=%d" % rate_point)
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_blocks, batch_size):
            
            if (i+batch_size) <= num_blocks:
                x = tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = tsdf_blocks[i::, :, :, :, :]
            current_batck_size = x.shape[0]
            
            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (z)
            y = model.g_a(x)
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
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batck_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batck_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batck_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batck_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_blocks:
                tsdf_blocks_out[i:i+batch_size, :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[i::, :, :, :, :] = xhat.detach().cpu().numpy()
    end_time = time.time() 
    duration = end_time - start_time  
    print(f"Duration: {duration} seconds")

    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_Hyperprior_%d.ply' % (input_mesh, rate_point))

'''
def Test_HyperLCS(input_mesh, rate_point, tsdf_volume, volume_origin, voxel_size, batch_size=512):
    # From Balle's tensorflow compression examples
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64
    scale_table = torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))
    
    model = HyperLCS(192, 64).to('cuda')
    model.load_state_dict(torch.load('./TrainedModels/HyperLCS/M%d_N%d_R%d.pth' % (192, 64, rate_point), map_location=torch.device('cuda')), strict=True)
    model.entropy_bottleneck.update()
    model.gaussian_conditional.update_scale_table(scale_table)
    model.eval()
    
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    num_blocks = tsdf_blocks.shape[0]
    
    print("Compression using HyperLCS Model p=%d" % rate_point)
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_blocks, batch_size):
            
            if (i+batch_size) <= num_blocks:
                x = tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = tsdf_blocks[i::, :, :, :, :]
            current_batck_size = x.shape[0]
            
            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (z)
            y = model.g_a(x)
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
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batck_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batck_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batck_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batck_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_blocks:
                tsdf_blocks_out[i:i+batch_size, :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[i::, :, :, :, :] = xhat.detach().cpu().numpy()
    end_time = time.time() 
    duration = end_time - start_time  
    print(f"Duration: {duration} seconds")
    
    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_HyperLCS_%d.ply' % (input_mesh, rate_point))
'''


def Test_HyperLCS(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size, batch_size=512):
    # From Balle's tensorflow compression examples
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64
    scale_table = torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))
    
    model = HyperLCS(192, 64).to('cuda')
    model.load_state_dict(torch.load('./TrainedModels/HyperLCS/M%d_N%d_R%d.pth' % (192, 64, rate_point), map_location=torch.device('cuda')), strict=True)
    model.entropy_bottleneck.update()
    model.gaussian_conditional.update_scale_table(scale_table)
    model.eval()
    
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    num_blocks = tsdf_blocks.shape[0]
    
    print("Compression using HyperLCS Model p=%d" % rate_point)
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_blocks, batch_size):
            
            if (i+batch_size) <= num_blocks:
                x = tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = tsdf_blocks[i::, :, :, :, :]
            current_batck_size = x.shape[0]
            
            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (z)
            y = model.g_a(x)
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
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batck_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batck_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batck_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batck_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_blocks:
                tsdf_blocks_out[i:i+batch_size, :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[i::, :, :, :, :] = xhat.detach().cpu().numpy()
    end_time = time.time() 
    duration = end_time - start_time  
    print(f"Duration: {duration} seconds")
    
    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0, mask=mask_volume)
    #verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_HyperLCS_%d.ply' % (input_mesh, rate_point))



input_mesh = 'Armadillo'
voxel_size = 0.8
rate_point = 4


tsdf_volume, volume_origin, volume_dim = mesh_to_TSDFVolume('./Stanford3D/%s.ply' % input_mesh, voxel_size)
_, _, _, mask_cube = create_topology_mask(tsdf_volume)
mask_volume = mask_cube.astype(bool)

Test_HyperLCS(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size)
#Test_HyperLCS(input_mesh, rate_point, tsdf_volume, volume_origin, voxel_size)
#Test_Factorizerd(input_mesh, rate_point, tsdf_volume, volume_origin, voxel_size)
#Test_Hyperprior(input_mesh, rate_point, tsdf_volume, volume_origin, voxel_size)

mask_blocks = reshape_volume_to_blocks(mask_volume, (8, 8, 8))










