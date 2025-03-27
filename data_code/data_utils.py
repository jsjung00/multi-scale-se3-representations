from .shapenet import ShapeNet
from torch.utils.data import random_split 
import numpy as np 
import torch 

class PointCloudScaleAndTranslate(object):
    '''
    Augments point cloud by randomly scaling and translating. This means point clouds
        may not have max norm 1 
    '''
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

def get_pointcloud_datasets(dataset_cfg, split_ratios=[0.8, 0.1,0.1]):
    # Returns train, val, test datasets or train/test based on the given split_ratios 
    dataset_name = dataset_cfg.NAME 
    if dataset_name == 'ShapeNet':
        full_dataset = ShapeNet(dataset_cfg)
    else:
        raise ValueError("Currently only loads ShapeNet. TODO: add in ModelNet40 and ScanObject")

    split_datasets = random_split(full_dataset, split_ratios) 
    return split_datasets