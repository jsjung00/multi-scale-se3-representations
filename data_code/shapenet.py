import torch.utils.data as data
import os 
import torch 
import numpy as np 
from .io import IO 

class ShapeNet(data.Dataset):
    def __init__(self, config, subset):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = subset
        self.npoints = config.N_POINTS
        self.sampling = config.SAMPLING 
        self.gauss_sigma = config.GAUSS_SIGMA 
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.SAMPLE_POINTS
        self.whole = config.get('whole')

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print(f'[DATASET] Open file {test_data_list_file}')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def farthest_point_sampling(self, points, seed=None):
        '''
        Return a subsampled version using FPS
        '''
        N,dim = points.shape 
        k = self.sample_points_num
        assert k <= N, "sampling k must be smaller than number of points"

        if seed is not None:
            np.random.seed(seed)
        
        # array of selected indices
        sampled_idxs = np.zeros(k, dtype=int)
        sampled_idxs[0] = np.random.randint(N)

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
        # Returns de-meaned and de-normed point cloud
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        if self.sampling == "FPS":
            data = self.farthest_point_sampling(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        

        data = self.pc_norm(data)
        if self.gauss_sigma > 0:
            # add noise and then re-normalize 
            data = data + self.gauss_sigma * np.random.randn(*data.shape) 
            data = self.pc_norm(data)

        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)