from .shapenet import ShapeNet
from .ScanObjectNNDataset import ScanObjectNN
from .ModelNetDataset import ModelNet 
from torch.utils.data import random_split 
import numpy as np 
import torch 

class GraphScale(object):
    '''
    The same as PointCloudScaleAndTranslate but takes in as input (feats, coords, mask) or (feats, coords, mask, label)    
    '''
    def __init__(self):
        pass

    def __call__(self, graph_tuple):
        if len(graph_tuple) == 3:
            feats, coords, mask = graph_tuple 
        elif len(graph_tuple) == 4:
            feats, coords, mask, label = graph_tuple
        else: 
            raise ValueError("Objet should be either (feats, coords, mask) or (feats, coords, mask, label)") 
        
        # NOTE: this assumes that feature is still just the norm of the point cloud coordinate
        scaler = PointCloudScale()
        if len(coords.shape) == 2:
            new_coords = scaler(coords.unsqueeze(dim=0)).squeeze() #(N,3)
        elif len(coords.shape) == 3:
            new_coords = scaler(coords)
        else:
            raise ValueError("Coordinates shoudl be (B,N,3) or (N,3)")

        new_feats = torch.norm(new_coords, dim=-1, keepdim=True)

        if len(graph_tuple) == 3:
            return new_feats, new_coords, mask 
        
        return new_feats, new_coords, mask, label

class PointCloudScale(object):
    '''
    Randomly scales the point cloud.... but NO translation
    '''
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        pc_transformed = pc.clone()
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc_transformed[i, :, 0:3] = torch.mul(pc_transformed[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc_transformed.device))
            
        return pc_transformed


class PointCloudScaleAndTranslate(object):
    '''
    Augments point cloud by randomly scaling and translating. This means point clouds
        may not have max norm 1. This is used by PointM2AE. NOTE: We don't do this
    '''
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        pc_transformed = pc.clone()
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc_transformed[i, :, 0:3] = torch.mul(pc_transformed[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc_transformed

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
    elif dataset_name == "ModelNet":
        train_dataset = ModelNet(dataset_cfg, "train")
        if val_ratio:
            train_dataset, val_dataset = random_split(train_dataset, [1-val_ratio, val_ratio]) 
        test_dataset = ModelNet(dataset_cfg, "test")

    else:
        raise ValueError("Currently only loads ShapeNet and ScanObject. TODO: add modelnet40")

    if val_ratio:
        return train_dataset, val_dataset, test_dataset 
    
    return train_dataset, test_dataset