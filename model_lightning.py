import lightning as L 
from lightning.pytorch.callbacks import Callback 
import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math 
from sklearn.svm import SVC
import numpy as np 
from tqdm import tqdm 

class SVMScoreCallback(L.Callback):
    def __init__(self, train_svm_dl, test_svm_dl):
        self.train_svm_dl = train_svm_dl
        self.test_svm_dl = test_svm_dl 

    def on_train_epoch_end(self, trainer, pl_module):
        train_output_dicts = [] 
        test_output_dicts = []

        # get features from frozen encoder 
        for batch in tqdm(self.train_svm_dl):
            output_dict = pl_module.get_svm_feat_labels(batch, is_test=False)
            train_output_dicts.append(output_dict)

        for batch in tqdm(self.test_svm_dl):
            output_dict = pl_module.get_svm_feat_labels(batch, is_test=True)
            test_output_dicts.append(output_dict)
        
        train_rep_dict, test_rep_dict = {}, {}
        for pool_type in ['max', 'mean', 'sum', 'cat']:
            train_rep_dict[pool_type] = np.concatenate([output_dict[f"{pool_type}_rep"].detach().cpu().numpy() for output_dict in train_output_dicts], axis=0) 
            test_rep_dict[pool_type] = np.concatenate([output_dict[f"test_{pool_type}_rep"].detach().cpu().numpy() for output_dict in test_output_dicts], axis=0)

        train_labels = np.concatenate([output_dict['labels'] for output_dict in train_output_dicts], axis=0)
        test_labels = np.concatenate([output_dict['test_labels'] for output_dict in test_output_dicts], axis=0)
        
        for pool_type in ['max', 'mean', 'sum', 'cat']:
            model = SVC(C = 0.01, kernel='linear')
            model.fit(train_rep_dict[pool_type], train_labels)
            test_accuracy = model.score(test_rep_dict[pool_type], test_labels)
            print(f"Pooling type {pool_type} model Linear accuracy: {test_accuracy}")
            if pl_module.cfg.strategy != "auto":
                pl_module.log(f"{pool_type}_pool_test_accuracy", test_accuracy, sync_dist=True)
            else:
                pl_module.log(f"{pool_type}_pool_test_accuracy", test_accuracy, sync_dist=False)
            






