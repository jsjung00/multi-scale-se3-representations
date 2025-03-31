import sys
import os 
import torch 
import numpy as np 
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))

from data_code.data_utils import get_pointcloud_datasets
from data_code.graph_utils import SpectralPointCloud
import yaml
from munch import Munch
from utils import plot_3d_point_clouds, get_ptcloud_img
import numpy as np 




def test_low_pass():
    torch.manual_seed(0)
    np.random.seed(0)

    config_path = '/mnt/justin/multi-scale-se3-representations/conf/data/shapenet.yaml'

    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    cfg = Munch(config_dict)
    pc_train_dataset, pc_val_dataset, pc_test_dataset = get_pointcloud_datasets(cfg)
    point_cloud = pc_train_dataset[7][2]
   
    Spec = SpectralPointCloud(point_cloud, eps_ball=True, eps_radius=0.1, k=None)

    point_cloud_batch = Spec.get_lower_resolution_batch(4, point_cloud, [1024, 500, 100, 10])

    get_ptcloud_img(point_cloud_batch, "composition.png")


if __name__ == "__main__":
    test_low_pass()