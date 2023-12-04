import os
import time
import zlib
import numpy as np
from skimage import measure

import torch
from torchac import torchac

from Models import *
from Utils import *

###############################################################################
def Test_Factorizerd(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size, batch_size=512):
    model = FactorizedPriorModel(192).to('cuda')
    model.load_state_dict(torch.load('./TrainedModels/FactorizedPriorModel/M%d_R%2d.pth' % (192, rate_point), map_location=torch.device('cuda')), strict=True)
    model.entropy_bottleneck.update()
    model.eval()
    
    # Only blocks spanning the zero-crossing surface will be compressed.
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    mask_blocks = reshape_volume_to_blocks(mask_volume, (8, 8, 8))
    valids = np.where(mask_blocks.sum(axis=(1,2,3,4)) != 0)[0] # Blocks where a sign change occurs at least once between adjacent voxels.
    valid_tsdf_blocks = tsdf_blocks[valids, ...]
    num_valid_blocks = valid_tsdf_blocks.shape[0]
    
    print("Compression using Factorized Model p=%d" % rate_point)
    y_rate = 0.0
    s_rate = 0.0   
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_valid_blocks, batch_size):
            
            if (i+batch_size) <= num_valid_blocks:
                x = valid_tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = valid_tsdf_blocks[i::, :, :, :, :]
            current_batch_size = x.shape[0]
            
            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (y)
            y = model.g_a(x)
            y_enc = model.entropy_bottleneck.compress(y)
            
            y_string = b''.join(y_enc)        
            y_rate += len(zlib.compress(y_string)) * 8
            
            # Decompress (y)
            y_hat = model.entropy_bottleneck.decompress(y_enc, y.size()[-2:])
            y_hat = y_hat.reshape(y.shape)
            x_hat = model.g_s(y_hat)
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batch_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batch_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batch_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
            
            s_rate += len(zlib.compress(sign_enc)) * 8
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batch_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_valid_blocks:
                tsdf_blocks_out[valids[i:i+batch_size], :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[valids[i::], :, :, :, :] = xhat.detach().cpu().numpy()
                
    end_time = time.time() 
    duration = end_time - start_time  
    print(f" - Duration: {duration} seconds")
    total_bits = (y_rate + s_rate) / 1000.0
    print(f" - Compressed Size: {total_bits} Kbit")
    
    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0, mask=mask_volume)
    #verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_Factorized_%d.ply' % (input_mesh, rate_point))
    
    

###############################################################################
def Test_Hyperprior(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size, batch_size=512):
    # From Balle's tensorflow compression examples
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64
    scale_table = torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))
    
    model = ScaleHyperPriorModel(192, 64).to('cuda')
    model.load_state_dict(torch.load('./TrainedModels/ScaleHyperPriorModel/M%d_N%d_R%d.pth' % (192, 64, rate_point), map_location=torch.device('cuda')), strict=True)
    model.entropy_bottleneck.update()
    model.gaussian_conditional.update_scale_table(scale_table)
    model.eval()
    
    # Only blocks spanning the zero-crossing surface will be compressed.
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    mask_blocks = reshape_volume_to_blocks(mask_volume, (8, 8, 8))
    valids = np.where(mask_blocks.sum(axis=(1,2,3,4)) != 0)[0] # Blocks where a sign change occurs at least once between adjacent voxels.
    valid_tsdf_blocks = tsdf_blocks[valids, ...]
    num_valid_blocks = valid_tsdf_blocks.shape[0]
    
    print("Compression using Hyperprior Model p=%d" % rate_point)
    y_rate = 0.0
    z_rate = 0.0
    s_rate = 0.0   
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_valid_blocks, batch_size):
            
            if (i+batch_size) <= num_valid_blocks:
                x = valid_tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = valid_tsdf_blocks[i::, :, :, :, :]
            current_batch_size = x.shape[0]

            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (z)
            y = model.g_a(x)
            z = model.h_a(torch.abs(y))
            z_enc = model.entropy_bottleneck.compress(z)
            
            z_string = b''.join(z_enc)        
            z_rate += len(zlib.compress(z_string)) * 8
            
            # Decompress (z)
            z_hat = model.entropy_bottleneck.decompress(z_enc, z.size()[-2:])
            z_hat = z_hat.reshape(z.shape)
            scales_hat = model.h_s(z_hat)
            indexes = model.gaussian_conditional.build_indexes(scales_hat)
            
            # Compress (y)
            y_enc = model.gaussian_conditional.compress(y, indexes)
            
            y_string = b''.join(y_enc)        
            y_rate += len(zlib.compress(y_string)) * 8
            
            # Decompress (y)
            y_hat = model.gaussian_conditional.decompress(y_enc, indexes, torch.float32)
            x_hat = model.g_s(y_hat)
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batch_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batch_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batch_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
            
            s_rate += len(zlib.compress(sign_enc)) * 8
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batch_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_valid_blocks:
                tsdf_blocks_out[valids[i:i+batch_size], :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[valids[i::], :, :, :, :] = xhat.detach().cpu().numpy()
                
    end_time = time.time() 
    duration = end_time - start_time  
    print(f" - Duration: {duration} seconds")
    total_bits = (y_rate + z_rate + s_rate) / 1000.0
    print(f" - Compressed Size: {total_bits} Kbit")
    
    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0, mask=mask_volume)
    #verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_Hyperprior_%d.ply' % (input_mesh, rate_point))
    
    

###############################################################################
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
    
    # Only blocks spanning the zero-crossing surface will be compressed.
    tsdf_blocks = reshape_volume_to_blocks(tsdf_volume, (8, 8, 8))
    tsdf_blocks_out = np.zeros_like(tsdf_blocks)
    mask_blocks = reshape_volume_to_blocks(mask_volume, (8, 8, 8))
    valids = np.where(mask_blocks.sum(axis=(1,2,3,4)) != 0)[0] # Blocks where a sign change occurs at least once between adjacent voxels.
    valid_tsdf_blocks = tsdf_blocks[valids, ...]
    num_valid_blocks = valid_tsdf_blocks.shape[0]
    
    print("Compression using HyperLCS Model p=%d" % rate_point)
    y_rate = 0.0
    z_rate = 0.0
    s_rate = 0.0   
    start_time = time.time()
    with torch.no_grad():    
        for i in range(0, num_valid_blocks, batch_size):
            
            if (i+batch_size) <= num_valid_blocks:
                x = valid_tsdf_blocks[i:i+batch_size, :, :, :, :]
            else:
                x = valid_tsdf_blocks[i::, :, :, :, :]
            current_batch_size = x.shape[0]
            
            s = (x >= 0).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            s = torch.from_numpy(s).cuda()
            
            # Compress (z)
            y = model.g_a(x)
            z = model.h_a(torch.abs(y))
            z_enc = model.entropy_bottleneck.compress(z)
            
            z_string = b''.join(z_enc)        
            z_rate += len(zlib.compress(z_string)) * 8
        
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
            
            y_string = b''.join(y_enc)        
            y_rate += len(zlib.compress(y_string)) * 8
            
            # Decompress (y)
            y_hat1 = model.gaussian_conditional.decompress(y_enc, 
                                                           torch.masked_select(indexes, impts_mask).reshape(1, -1), torch.float32)
            
            y_hat2 = torch.zeros_like(y)
            y_hat2[impts_mask] = y_hat1
            x_hat = model.g_s(y_hat2)
            shat = model.head_sign(x_hat).clamp( 0., 1.) # sign prediction
            xhat_mgn = model.head_magn(x_hat).clamp(-1., 1.) # magn prediction
            
            # Compress (sign)
            symbols = s.reshape(current_batch_size, 8*8*8, 1)
            symbols = symbols.type(torch.int16)
            probs = shat.reshape(current_batch_size, 8*8*8, 1)
            cdf_float = torch.zeros((current_batch_size, 8*8*8, 1, 3), dtype=torch.float32)
            cdf_float[:, :, :, 0] = 0.0
            cdf_float[:, :, :, 1] = 1.0 - probs
            cdf_float[:, :, :, 2] = 1.0
            symbols = symbols.cpu()
            cdf_float = cdf_float.cpu()
            sign_enc = torchac.encode_float_cdf(cdf_float, symbols, needs_normalization=True, check_input_bounds=True)
           
            s_rate += len(zlib.compress(sign_enc)) * 8
            
            # Decompress (sign)
            symbols_out = torchac.decode_float_cdf(cdf_float, sign_enc, needs_normalization=True)
            #assert symbols_out.equal(symbols)
            sign_dec = symbols_out.reshape(current_batch_size, 1, 8, 8, 8)
            assert sign_dec.equal(s.to(torch.int16).cpu())
            
            # Final Output
            xhat = (sign_dec * 2.0 - 1.0) * torch.abs(xhat_mgn.cpu())
                    
            if (i+batch_size) <= num_valid_blocks:
                tsdf_blocks_out[valids[i:i+batch_size], :, :, :, :] = xhat.detach().cpu().numpy()
            else:
                tsdf_blocks_out[valids[i::], :, :, :, :] = xhat.detach().cpu().numpy()
                
    end_time = time.time() 
    duration = end_time - start_time  
    print(f" - Duration: {duration} seconds")
    total_bits = (y_rate + z_rate + s_rate) / 1000.0
    print(f" - Compressed Size: {total_bits} Kbit")
    
    tsdf_volume_out = merge_blocks_to_volume(tsdf_blocks_out, tsdf_volume.shape, (8, 8, 8))    
    
    # Use marching cubes to obtain the surface mesh 
    verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0, mask=mask_volume)
    #verts_out, faces_out, normals_out, values_out = measure.marching_cubes(tsdf_volume_out, 0)
    verts_out = verts_out * voxel_size + volume_origin
    faces_out = np.flip(faces_out, axis=1)
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_normals=normals_out)
    _ = mesh_out.export('%s_HyperLCS_%d.ply' % (input_mesh, rate_point))



###############################################################################
def main():
    input_mesh = 'Armadillo' 
    voxel_size = 0.8 
    rate_point = 4
    
    # Converting a mesh PLY file to a TSDF volume
    tsdf_volume, volume_origin, volume_dim = mesh_to_TSDFVolume('./Stanford3D/%s.ply' % input_mesh, voxel_size)
    _, _, _, mask_cube = create_topology_mask(tsdf_volume)
    mask_volume = mask_cube.astype(bool)
    
    # Compression of the same TSDF volume using three different models
    Test_Factorizerd(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size)
    Test_Hyperprior(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size)
    Test_HyperLCS(input_mesh, rate_point, tsdf_volume, mask_volume, volume_origin, voxel_size)


if __name__ == "__main__":
	main()
