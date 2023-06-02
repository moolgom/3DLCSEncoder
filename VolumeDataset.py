import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TSDFVolumeDataset(Dataset):
    def __init__(self, block_dim=[8, 8, 8], augment=True, dataset='train'):
        self.block_dim = block_dim
        self.augment = augment
        self.num_blocks = 0
        self.dataset = dataset
        
        self.TSDF = np.zeros((0, self.block_dim[0], self.block_dim[1], self.block_dim[2]), dtype=np.float32)
        self.MASK = np.zeros((0, self.block_dim[0], self.block_dim[1], self.block_dim[2]), dtype=np.float32)
        self.SIGN = np.zeros((0, self.block_dim[0], self.block_dim[1], self.block_dim[2]), dtype=np.float32)
        self.MAGN = np.zeros((0, self.block_dim[0], self.block_dim[1], self.block_dim[2]), dtype=np.float32)
        
        self._loadDataset()
        
    def _loadDataset(self):
        if self.dataset == 'val':
            npzfiles = glob('./Dataset/Val/*%dx%dx%d.npz' % (self.block_dim[0], self.block_dim[1], self.block_dim[2]))
        elif self.dataset == 'train':
            npzfiles = glob('./Dataset/Train/*%dx%dx%d.npz' % (self.block_dim[0], self.block_dim[1], self.block_dim[2]))
        elif self.dataset =='test':
            npzfiles = glob('./Dataset/Test/*%dx%dx%d.npz' % (self.block_dim[0], self.block_dim[1], self.block_dim[2]))
            
        for npzfile in npzfiles:
            ''' print('Loading %s...' % npzfile) '''
            npzdata = np.load(npzfile)
            _TSDF = npzdata['TSDF']
            _MASK = npzdata['MASK']
            _SIGN = npzdata['SIGN']
            _MAGN = npzdata['MAGN']
            
            print('%d TSDF blocks are loaded from %s' % (_TSDF.shape[0], npzfile))
            
            self.TSDF = np.vstack((self.TSDF, _TSDF))
            self.MASK = np.vstack((self.MASK, _MASK))
            self.SIGN = np.vstack((self.SIGN, _SIGN))
            self.MAGN = np.vstack((self.MAGN, _MAGN))
            
        self.num_blocks = self.TSDF.shape[0]
        
    def __len__(self):
        return self.num_blocks
    
    
    def __getitem__(self, idx):
        tsdf = self.TSDF[idx, :, :, :]
        mask = self.MASK[idx, :, :, :]
        sign = self.SIGN[idx, :, :, :]
        magn = self.MAGN[idx, :, :, :]
        
        tsdf = tsdf.reshape((1, self.block_dim[0], self.block_dim[1], self.block_dim[2]))
        mask = mask.reshape((1, self.block_dim[0], self.block_dim[1], self.block_dim[2]))
        sign = sign.reshape((1, self.block_dim[0], self.block_dim[1], self.block_dim[2]))
        magn = magn.reshape((1, self.block_dim[0], self.block_dim[1], self.block_dim[2]))
        
        if self.augment:
            # augmentation (random rotation)
            axes = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
            r0 = random.randint(0, 3)
            r1 = random.randint(0, 5)
            
            tsdf = np.rot90(tsdf, k=r0, axes=axes[r1])
            mask = np.rot90(mask, k=r0, axes=axes[r1])
            sign = np.rot90(sign, k=r0, axes=axes[r1])
            magn = np.rot90(magn, k=r0, axes=axes[r1])
        
        return tsdf.copy(), mask.copy(), sign.copy(), magn.copy()
    
    
    
'''
dataset = TSDFVolumeDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_idx, samples in enumerate(dataloader):
    
    tsdf = samples[0]
    mask = samples[1]
    sign = samples[2]
    magn = samples[3]
    
    break


import trimesh
from skimage import measure
vol = tsdf.detach().numpy()
vol = vol[0, :, :, :]
vol = vol.squeeze()
verts, faces, normals, values = measure.marching_cubes(vol, 0)
mesh = trimesh.Trimesh(verts, faces)
_ = mesh.export('dec1.ply')   
'''