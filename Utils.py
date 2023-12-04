import trimesh
import numpy as np
from pysdf import SDF
from numba import njit


def mesh_to_TSDFVolume(mesh_path, voxel_size):
    print("Converting mesh to TSDF volume...", end='')
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

