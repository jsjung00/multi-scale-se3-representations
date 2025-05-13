import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch

SPLIT_MAP = {
    'OBJ': '_objectdataset.h5',
    'OBJ-BG': '_objectdataset.h5',
    'PB': '_objectdataset_augmentedrot_scale75.h5',
}

class ScanObjectNN(Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.subset = subset
        self.root = config.ROOT
        self.split = config.SPLIT
        self.sample_points_num = config.SAMPLE_POINTS
        self.npoints = config.N_POINTS
        self.permutation = np.arange(self.npoints)
        self.sampling = config.SAMPLING
        self.gauss_sigma = config.GAUSS_SIGMA 
        self.norm = config.NORM
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training' + SPLIT_MAP[self.split]), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test' + SPLIT_MAP[self.split]), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        self.torch_gen = torch.Generator().manual_seed(42)
        self.np_rng   = np.random.default_rng(42)

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def random_sample(self, pc, num):
        self.np_rng.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def farthest_point_sampling(self, points, seed=None):
        '''
        Return a subsampled version using FPS
        '''
        N,dim = points.shape 
        k = self.sample_points_num
        assert k <= N, "sampling k must be smaller than number of points"
        
        # array of selected indices
        sampled_idxs = np.zeros(k, dtype=int)
        sampled_idxs[0] = self.np_rng.integers(low=0, high=N, size=None)

        # keep track of min distance to the set of selected points
        min_dists = np.full(N, np.inf)
        for i in range(1,k):
            # get distance from all points to last added point
            last_pt = points[sampled_idxs[i-1]]
            diff = points - last_pt 
            d2 = np.linalg.norm(diff, axis=1)

            min_dists = np.minimum(min_dists, d2) #update the minimum dist to any selected point 

            sampled_idxs[i] = np.argmax(min_dists)
        
        return points[sampled_idxs]

    def __getitem__(self, idx):
        # Returns de-meaned and denormed point cloud 
        pt_idxs = np.arange(0, self.points.shape[1])   
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()

        if self.sampling == "FPS":
            current_points = self.farthest_point_sampling(current_points, self.sample_points_num)
        else:
            current_points = self.random_sample(current_points, self.sample_points_num)
        
        if self.norm:
            current_points = self.pc_norm(current_points)

        if self.gauss_sigma > 0:
            # add noise and then re-normalize 
            current_points = current_points + self.gauss_sigma * np.random.randn(*current_points.shape) 
            if self.norm:
                current_points = self.pc_norm(current_points)
        

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]