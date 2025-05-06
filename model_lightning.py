import lightning as L 
from lightning.pytorch.callbacks import Callback 
import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math 
from sklearn.svm import SVC, LinearSVC
import numpy as np 
from tqdm import tqdm 
from time import time 
from torch import distributed as dist
import threadpoolctl 


class SVMScoreCallback(Callback):
    def __init__(self, train_svm_dl, test_svm_dl, every_n_epoch=1):
        self.train_svm_dl = train_svm_dl
        self.test_svm_dl = test_svm_dl 
        self.every_n_epoch = every_n_epoch 

    def on_train_epoch_end(self, trainer, pl_module):
        '''
        Only runs on rank 0 GPU to prevent hanging
        '''
        torch.cuda.synchronize()

        # align all GPUs
        if dist.is_initialized():
            dist.barrier()      
       
        if not (trainer.current_epoch + 1) % self.every_n_epoch == 0:
            return 
        
        # only run on rank-zero GPU
        if trainer.is_global_zero:
            train_output_dicts, test_output_dicts = [], [] 
            # get features from frozen encoder 
            pl_module.eval()
            with torch.inference_mode():
                for batch in tqdm(self.train_svm_dl):
                    train_output_dicts.append(pl_module.get_svm_feat_labels(batch, is_test=False))
                for batch in tqdm(self.test_svm_dl):
                    test_output_dicts.append(pl_module.get_svm_feat_labels(batch, is_test=True))
        
            train_rep_dict, test_rep_dict = {}, {}
            for pool_type in ['sum', 'cat']:
                train_rep_dict[pool_type] = np.concatenate([output_dict[f"{pool_type}_rep"] for output_dict in train_output_dicts], axis=0) 
                test_rep_dict[pool_type] = np.concatenate([output_dict[f"test_{pool_type}_rep"] for output_dict in test_output_dicts], axis=0)

            train_labels = np.concatenate([output_dict['labels'] for output_dict in train_output_dicts], axis=0)
            test_labels = np.concatenate([output_dict['test_labels'] for output_dict in test_output_dicts], axis=0)
            
            for pool_type in ['sum', 'cat']:
                model = LinearSVC(C = 0.01)
                model.fit(train_rep_dict[pool_type], train_labels)
                test_accuracy = model.score(test_rep_dict[pool_type], test_labels)

                pl_module.log(f"{pool_type}_pool_test_accuracy", test_accuracy, sync_dist=False, rank_zero_only=True)
                print(f"Pooling type {pool_type} model Linear accuracy: {test_accuracy}")
        
        if dist.is_initialized():
            dist.barrier()






