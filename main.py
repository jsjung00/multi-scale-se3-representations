'''
Pytorch lightning driver code to set up trainer and train
'''
from egnn_lightning import LitEGNNConsistent
from lightning.pytorch import Trainer
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils import PaddedMatrixDataset, PairedDataset, LARS, get_max_num_nodes, PointToGraphDataset
from data_code.data_utils import get_pointcloud_datasets, PointCloudScaleAndTranslate
from data_code.data_lightning import PointCloudDataModule
from model_lightning import SVMScoreCallback
from utils_lightning import SimulatedErrorCallback
import torch 
from datetime import timedelta
from torch.utils.data import DataLoader
import torch.distributed as dist 
import hydra 
from omegaconf import DictConfig, OmegaConf 
import mlflow 
import os 
import yaml
from dotmap import DotMap
from torchvision import transforms 
import multiprocessing
import wandb
from dotenv import load_dotenv
load_dotenv()

# Ensure that we fail immediately with any blockers
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    wandb.login(key=os.getenv('WANDB_API_KEY'), relogin=True)

    # reduce timeout
    #if cfg.devices > 1:
    #    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=3), )

    print(OmegaConf.to_yaml(cfg))

    # Dataset Paths
    training_dir = cfg.training.training_dir
    
    # Initialize the Data module
    data_module = PointCloudDataModule(cfg)

    if cfg.training.pair_loss:
        raise ValueError("Pair loss should be turned off for these point cloud experiments.")
        #train_ds, val_ds, test_ds = PairedDataset(train_ds), PairedDataset(val_ds), PairedDataset(test_ds)

    model = LitEGNNConsistent(cfg)
    #model = torch.compile(model)  #torch compile has slight speedup
    
    # Use wandb logger
    logger = WandbLogger(name=cfg.training.experiment_name, save_dir="./mlruns",project="consistency_profiling")
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    #TODO: change min delta based on experiment!
    checkpoint_callback = ModelCheckpoint(monitor="log_val_loss", mode="min",every_n_epochs=1,\
        dirpath=cfg.training.checkpoint_dir, filename='{epoch}-{step}')
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "__"

    # Initialize SVM evaluation dataset and SVM callback
    svm_train_ds, svm_test_ds = get_pointcloud_datasets(cfg.svm_data, val_ratio=None)
    svm_train_ds, svm_test_ds = PointToGraphDataset(svm_train_ds, True), PointToGraphDataset(svm_test_ds, True) 
    svm_train_dl, svm_test_dl = DataLoader(svm_train_ds, batch_size=cfg.training.svm_batch_size, num_workers=2, drop_last=True), DataLoader(svm_test_ds, batch_size=cfg.training.svm_batch_size, num_workers=2, drop_last=True)
    SVM_callback =  SVMScoreCallback(svm_train_dl, svm_test_dl, every_n_epoch=1)

    callbacks = [checkpoint_callback, SVM_callback]
    if cfg.training.early_stopping:
        callbacks.append(EarlyStopping(monitor="log_val_loss", mode="min", patience=5000, min_delta=0.01, check_on_train_epoch_end=False))

    trainer = Trainer(accelerator=cfg.accelerator, devices=cfg.devices, strategy=cfg.strategy, logger=logger,\
        callbacks=callbacks,\
            max_epochs=cfg.training.num_epochs, val_check_interval=1.0, gradient_clip_val=5.0, enable_progress_bar=cfg.training.enable_tqdm,\
            profiler="simple", precision='bf16-mixed')
    
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