'''Various utility functions from dataloaders, ect.'''
from fmVAE.datasets.multiZarrPointCloud import PointCloudDataset
from fmVAE.datasets.splitting import split_directory
import torch
from torch import nn, optim 
from torch.utils.data import Subset
import torch.nn.functional as F 
from torchvision.transforms.v2 import GaussianNoise
import sys
import os
import pdb
from torch_geometric.data import Data
#import pytorch3d 
import numpy as np 
#from torch_geometric.loader import Dataloader
sys.path.append(os.path.abspath(".."))
import math 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def get_max_num_nodes(ds):
    '''
    Given a pointcloud dataset (i.e each pointcloud is a tensor [N,3]) returns max number of nodes for any given pointcloud
    
    ds: (PointCloudDataset)
    '''
   
    max_num = 0 
    for i in range(0, len(ds)):
        max_num = max(max_num, ds[i].shape[0])
    return max_num 

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
        weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
            eta=eta, weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def adjust_learning_rate(training_config, optimizer, loader, step):
    # Adjust learning rate as in barlowtwins 
    max_steps = training_config.num_epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = training_config.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * training_config.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * training_config.learning_rate_biases

def off_diagonal(x):
    # x: (n, n) matrix
    # Used in barlow loss: returns a flattened view of off diagonla elements of square matrix
    n,m = x.shape 
    assert n == m 
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()



class PairedDataset(torch.utils.data.Dataset):
    '''
    Given a dataset and some transformations, returns a pair (x,x') where x' is a transformed pair.
        Applies one transformation which is sampled based on probability weights.

    transform: List[str] Names of the transforms that should be applied
    transform_p: List[float] Probability of transform being applied to create pair (one transform per pair) 
    
    ds: (dataset) PaddedMatrixDataset whose get element returns (feats, coors, mask)
    '''
    def __init__(self, ds, transform=['noise'], transform_p=[1], noise_sigma_low=1e-2, noise_sigma_high=1e-1):
        self.ds = ds 
        self.transform = transform 
        self.transform_p = transform_p 
        self.noise_sigma_low = noise_sigma_low
        self.noise_sigma_high = noise_sigma_high
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        assert len(self.transform) == len(self.transform_p)
        feats, coors, mask = self.ds[idx]

        chosen_transform = np.random.choice(self.transform, p=self.transform_p)
        if chosen_transform == 'rotation':
            raise ValueError("Not implemented. also don't need rotation for EGNN")
                
        if chosen_transform == "noise":
            # uniformly sample sigma from [low, high] and noises coordinates
            sigma = np.random.uniform(low=self.noise_sigma_low, high=self.noise_sigma_high)
            gaussian_noise = torch.normal(mean=torch.zeros_like(coors), std=sigma)
            coors_prime = coors + gaussian_noise
            # calculate new features based on noised coordinates
            feats_prime = torch.norm(coors_prime, dim=-1, keepdim=True)
        else:
            raise ValueError("Must have some transformation")

        return feats, feats_prime, coors, coors_prime, mask   


class PointToGraphDataset(torch.utils.data.Dataset):
    '''
    Retursn a dataset that is used for EGNN. Specifically handles point cloud dataset such as Shapenet and ScanObjectNN.
        Assumes that each point cloud has same number of points

    contains_label: (bool)If true, then the dataset has as third object tuple (pc, label). If false, dataset has as third object pc 
    '''
    def __init__(self, ds, contains_label, transform=None):
        self.ds = ds 
        self.contains_label = contains_label
        self.transform = transform 
        #num_nodes = [len(self.ds[i]) for i in range(0, len(self.ds))]
        #assert len(set(num_nodes)) == 1
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        # if contains_label, return (feats, coors, mask, label)
        # else return (feats, coors, mask)

        if self.contains_label: 
            coords, label = self.ds[idx][2]
        else:
            coords = self.ds[idx][2]

        mask = torch.ones(len(coords), dtype=torch.int).bool()
        feats = torch.norm(coords, dim=-1, keepdim=True)

        if self.transform:
            (feats, coords, mask) = self.transform((feats, coords, mask))

        if self.contains_label:
            return feats, coords, mask, label 

        return feats, coords, mask    

class PaddedMatrixDataset(torch.utils.data.Dataset):
    '''
    Returns a dataset that is used for EGNN. 
    Pads graph with zero nodes such that each graph has the same number of nodes. Supplies corresponding mask

    Input: 
    ds (PointCloudDataset)
    max_num_nodes (int | None)

    For each index return (feats, coords, mask) where
        feat: (Tensor) of shape (N,1) which is the norm of each point. (Important! This is now E(N) invariant)
        coords: (Tensor) of shape (N,3)
        mask: (Tensor) of shape (N,) where 0 where element should be ignored and 1 is actual graph node  
    '''
    def __init__(self, ds, max_num_nodes=None):
        self.ds = ds
    
        if max_num_nodes is None:
            # define max number of nodes as the maximum num nodes in given dataset
            print("Warning: The maximum number of nodes for the input ds should be same for train + test!")
            num_nodes = [len(self.ds[i]) for i in range(0, len(self.ds))]
            self.max_num_nodes = max(num_nodes)
        else:
            self.max_num_nodes = max_num_nodes
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # return (feat, coors, mask) where feat is norm of coords
        
        # pad coords
        coords = self.ds[idx] #(N,3)
        num_null_nodes = self.max_num_nodes - len(coords)
        assert num_null_nodes >= 0, "max_num_nodes must be geq maximum number of nodes any graph in dataset"
        pad = (0, 0, 0, num_null_nodes)
        padded_coords = F.pad(coords, pad, "constant", 0)
        
        assert padded_coords.shape == (self.max_num_nodes, 3) 
        mask = torch.zeros(self.max_num_nodes, dtype=torch.int)
        mask[: len(coords)] = 1
        mask = mask.bool()


        # define features as the norm of each point
        feats = torch.norm(padded_coords, dim=-1, keepdim=True)

        return feats, padded_coords, mask  

def get_pointcloud_datasets(train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15, seed=42, directory=None, train_zarr=None,
                      val_zarr=None, test_zarr=None):
    '''
    Returns train/val/test datasets of pointclouds (Tensor, (N,3)) 

    Params: 
    directory: (str | None) Directory that contains folders of zarr files
    seed: (int | None) If not None, set the seed. This allows for us to prevent test leakage later

    train_zarr: (str | None) Zarr file path
    '''
    assert train_zarr is None and val_zarr is None and test_zarr is None, "TODO: add different datasets based on files"

    if train_zarr is None and val_zarr is None and test_zarr is None:
        assert directory, "Must give a directory if don't give specific files"
    
    if seed is not None:
        # set seed so we can do valid test!
        torch.manual_seed(seed)

    complete_dataset = PointCloudDataset(directory)
    total_size = len(complete_dataset)
    shuffled_indices = torch.randperm(total_size)
    train_indices = shuffled_indices[:int(train_ratio * total_size)]
    val_indices = shuffled_indices[int(train_ratio * total_size): int((train_ratio + val_ratio) * total_size)]
    test_indices = shuffled_indices[int((train_ratio + val_ratio) * total_size): ]

    train_dataset = Subset(complete_dataset, train_indices)
    val_dataset = Subset(complete_dataset, val_indices)
    test_dataset = Subset(complete_dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset 


def plot_3d_point_clouds(cloud1, cloud2=None, point_size=20):
    '''
    Plots a (batch of) point cloud or a (batch of) pair of point clouds next to each other 

    cloud1: Tensor/numpy of first point cloud, either (B,N,3) or (N,3) 
    cloud2: (None or tensors/numpy) of second point cloud, either (B,N,3) or (N,3) 
    '''
    def prepare_cloud(cloud):
        # convert to squeezed numpy 
        if cloud is not None:
            if torch.is_tensor(cloud):
                cloud = cloud.detach().cpu().numpy()
            return cloud.squeeze() 
        return 

    cloud1, cloud2 = prepare_cloud(cloud1), prepare_cloud(cloud2)

    if cloud1.ndim == 3:
        batch_size = cloud1.shape[0]
        fig = plt.figure(figsize=(6*batch_size, 10))

        for i in range(batch_size):
            ax = fig.add_subplot(1, batch_size, i+1, projection='3d')
            ax.scatter(cloud1[i][:, 0], cloud1[i][:, 1], cloud1[i][:, 2],
                    s=point_size, c='b', marker='o', label='Cloud1')
            if cloud2 is not None:
                ax.scatter(cloud2[i][:, 0], cloud2[i][:, 1], cloud2[i][:, 2],
                        s=point_size, c='r', marker='o', label='Cloud2')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title(f"Pair: {i+1}")
        plt.tight_layout()
        plt.show()
    elif cloud1.ndim == 2:
        # Single pair
        fig = plt.figure(figsize=(6, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2],
                s=point_size, c='b', marker='o', label='Cloud1')
        if cloud2 is not None:
            ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2],
                    s=point_size, c='r', marker='o', label='Cloud2')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title("3D Point Clouds")
        plt.show()
    else: 
        raise ValueError("Incorrect data given, either batch of point clouds or one point cloud")


def get_ptcloud_img(ptcloud, file_path="/mnt/justin/multi-scale-se3-representations/misc/point_cloud.png"):
    if torch.is_tensor(ptcloud):
        ptcloud = ptcloud.detach().cpu().numpy()

    if ptcloud.ndim == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        # ax.axis('scaled')
        ax.view_init(30, 45)

        max, min = np.max(ptcloud), np.min(ptcloud)
        ax.set_xbound(min, max)
        ax.set_ybound(min, max)
        ax.set_zbound(min, max)
        
        x, z, y = ptcloud.transpose(1, 0)
        ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
    elif ptcloud.ndim == 3:
        batch_size = ptcloud.shape[0]
        fig = plt.figure(figsize=(8*batch_size, 8))
        for i in range(batch_size):
            ax = fig.add_subplot(1, batch_size, i+1, projection='3d')
            ax.axis('off')
            ax.view_init(30, 45)

            single_point_cloud = ptcloud[i]
            max, min = np.max(single_point_cloud), np.min(single_point_cloud)
            ax.set_xbound(min, max)
            ax.set_ybound(min, max)
            ax.set_zbound(min, max)
            
            x, z, y = single_point_cloud.transpose(1, 0)
            ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
    else:
        raise ValueError("Must give ptcloud either a batch of point clouds dim 3 or a single point cloud dim2")


    fig.canvas.draw()
    fig.savefig(file_path, format="png")
    
    #TODO: get some object that I can return 
    
    #renderer = fig.canvas.get_renderer()
    #img = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
    #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #return img

def get_feat_mask(point_cloud, orig_mask):
    '''
    For a point cloud, returns features vector and mask 
    '''
    # TODO: change it so it doesn't just return ones vector  
    mask = torch.ones(len(point_cloud), dtype=torch.int)
    mask = mask.bool()

    # define features as the norm of each point
    feat = torch.norm(point_cloud, dim=-1, keepdim=True)
    return feat, mask 
    


if __name__ == "__main__":
    print("Hello world")
    #get_geometric_dl('')
