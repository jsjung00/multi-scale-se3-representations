from .shapenet import ShapeNet
from .ScanObjectNNDataset import ScanObjectNN
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

def get_pointcloud_datasets(dataset_cfg, val_ratio=None):
    '''
    Returns train, val (optional), test datasets

    val_ratio: (float) Percentage of training data
    '''
    
    dataset_name = dataset_cfg.NAME 
    if dataset_name == 'ShapeNet':
        train_dataset = ShapeNet(dataset_cfg, "train")
        if val_ratio:
            train_dataset, val_dataset = random_split(train_dataset, [1-val_ratio, val_ratio]) 
        test_dataset = ShapeNet(dataset_cfg, "test")
    elif dataset_name == "ScanObjectNN":
        train_dataset = ScanObjectNN(dataset_cfg, "train")
        if val_ratio:
            train_dataset, val_dataset = random_split(train_dataset, [1-val_ratio, val_ratio]) 
        test_dataset = ScanObjectNN(dataset_cfg, "test")
    else:
        raise ValueError("Currently only loads ShapeNet and ScanObject. TODO: add modelnet40")

    if val_ratio:
        return train_dataset, val_dataset, test_dataset 
    
    return train_dataset, test_dataset