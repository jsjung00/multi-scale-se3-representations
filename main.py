'''
Pytorch lightning driver code to set up trainer and train
'''
from egnn_lightning import LitEGNNConsistent
from lightning.pytorch import Trainer
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint 
from utils import PaddedMatrixDataset, PairedDataset, LARS, get_max_num_nodes, PointToGraphDataset
from data_code.data_utils import get_pointcloud_datasets, PointCloudScaleAndTranslate
from data_code.data_lightning import PointCloudDataModule
from model_lightning import SVMScoreCallback

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
    checkpoint_callback = ModelCheckpoint(monitor="log_val_loss", mode="min")

    # Initialize SVM evaluation dataset and SVM callback
    svm_train_ds, svm_test_ds = get_pointcloud_datasets(cfg.svm_data, val_ratio=None)
    svm_train_ds, svm_test_ds = PointToGraphDataset(svm_train_ds, True), PointToGraphDataset(svm_test_ds, True) 
    svm_train_dl, svm_test_dl = DataLoader(svm_train_ds, batch_size=cfg.training.svm_batch_size), DataLoader(svm_test_ds, batch_size=cfg.training.svm_batch_size)
    SVM_callback =  SVMScoreCallback(svm_train_dl, svm_test_dl)

    trainer = Trainer(accelerator=cfg.accelerator, devices=cfg.devices, strategy=cfg.strategy, logger=mlf_logger,\
        callbacks=[checkpoint_callback, SVM_callback, EarlyStopping(monitor="log_val_loss", mode="min", patience=80, min_delta=0.002)],\
            max_epochs=cfg.training.num_epochs, val_check_interval=1.0)
    
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
    print("\n <<Don't forget to change the loss weighting scalars!>>\n")
    main()