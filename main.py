'''
Pytorch lightning driver code to set up trainer and train
'''
from egnn_lightning import LitEGNNConsistent
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint 
from utils import PaddedMatrixDataset, PairedDataset, LARS, get_max_num_nodes, PointToGraphDataset
from data_code.data_utils import get_pointcloud_datasets, PointCloudScaleAndTranslate
from data_code.data_lightning import PointCloudDataModule

from torch.utils.data import DataLoader
import hydra 
from omegaconf import DictConfig, OmegaConf 
import mlflow 
import yaml
from dotmap import DotMap
from torchvision import transforms 
import multiprocessing


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Dataset Paths
    training_dir = cfg.training.training_dir
    
    # Initialize the Data module
    data_module = PointCloudDataModule(cfg)

    if cfg.training.pair_loss:
        raise ValueError("Pair loss should be turned off for these point cloud experiments.")
        #train_ds, val_ds, test_ds = PairedDataset(train_ds), PairedDataset(val_ds), PairedDataset(test_ds)

    model = LitEGNNConsistent(cfg)
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs_consistency", tracking_uri="http://127.0.0.1:5000", save_dir="./mlruns")
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    #TODO: change min delta based on experiment!
    checkpoint_callback = ModelCheckpoint(monitor="svm_test_accuracy", mode="max")

    trainer = Trainer(accelerator=cfg.accelerator, devices=cfg.devices, logger=mlf_logger,\
        callbacks=[checkpoint_callback, EarlyStopping(monitor="svm_test_accuracy", mode="max", patience=20, min_delta=0.002)],\
            max_epochs=cfg.training.num_epochs)
    
    if cfg.training.test:
        assert cfg.training.model_checkpoint is not None, "Need to specify which checkpoint to test"
        model = LitEGNNConsistent.load_from_checkpoint(checkpoint_path=cfg.training.model_checkpoint)
        
        test_results = trainer.test(model, datamodule=data_module)
        print("Test results", test_results)
    elif cfg.training.cont and cfg.training.model_checkpoint is not None:
        print("Starting from pre-trained checkpoint...\n")
        trainer.fit(model, datamodule=data_module, ckpt_path=cfg.training.model_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    #multiprocessing.set_start_method('spawn', force=True)
    print("\n <<Don't forget to change the loss weighting scalars!>>\n")
    main()