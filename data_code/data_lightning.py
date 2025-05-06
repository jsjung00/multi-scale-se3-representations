import lightning as L 
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from data_code.data_utils import get_pointcloud_datasets, GraphScale
from torch.utils.data import DataLoader
from utils import PaddedMatrixDataset, PairedDataset, LARS, get_max_num_nodes, PointToGraphDataset
from torchvision import transforms 

'''
Data module returns multiple dataloaders for train, validation and test (because we do SVM!)
'''


class PointCloudDataModule(L.LightningDataModule):
    '''
    Returns the training point cloud dataset in addition to train_svm and test_svm for svm evaluation!
    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.cfg_training = cfg.training 
        self.cfg_training_data = cfg.training_data 
        self.cfg_svm_data = cfg.svm_data

    def setup(self, stage: str):
        #Initialize the training (ShapeNet) Dataset and Add Augmentations
        train_transforms = transforms.Compose([GraphScale()])
        self.train_transforms = train_transforms
        pc_train_dataset, pc_val_dataset, pc_test_dataset = get_pointcloud_datasets(self.cfg_training_data, val_ratio=0.1)
        train_ds, val_ds, test_ds = PointToGraphDataset(pc_train_dataset, False, self.train_transforms), PointToGraphDataset(pc_val_dataset, False), PointToGraphDataset(pc_test_dataset, False)
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds 

        # Initialize SVM evaluation dataset
        svm_train_ds, svm_test_ds = get_pointcloud_datasets(self.cfg_svm_data, val_ratio=None)
        svm_train_ds, svm_test_ds = PointToGraphDataset(svm_train_ds, True), PointToGraphDataset(svm_test_ds, True) 
        self.svm_train_ds, self.svm_test_ds = svm_train_ds, svm_test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg_training.batch_size, num_workers=8, drop_last=True)
        # create a combined dataloader '
        '''
        iterables = {'training_pointclouds': DataLoader(self.train_ds, batch_size=self.cfg_training.batch_size),\
                     'svm_train': DataLoader(self.svm_train_ds, batch_size=self.cfg_training.svm_batch_size),\
                     'svm_test': DataLoader(self.svm_test_ds, batch_size=self.cfg_training.svm_batch_size)}
        combined_loader = CombinedLoader(iterables, 'max_size') 

        return combined_loader
        '''

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg_training.batch_size, num_workers=8, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.cfg_training.batch_size, num_workers=8, drop_last=True)

