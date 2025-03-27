'''
Pytorch lightning driver code to set up trainer and train
'''
from egnn_lightning import LitEGNNPointNet
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint 
from utils import PaddedMatrixDataset, PairedDataset, LARS, get_max_num_nodes
from data_code.data_utils import get_pointcloud_datasets, PointCloudScaleAndTranslate
from torch.utils.data import DataLoader
import hydra 
from omegaconf import DictConfig, OmegaConf 
import mlflow 
import yaml
from dotmap import DotMap
from torchvision import transforms 


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Dataset Paths
    training_dir = cfg.training.training_dir
    
    # Initialize the Dataset and Add Augmentations
    train_transforms = transforms.Compose([PointCloudScaleAndTranslate()])

    pc_train_dataset, pc_val_dataset, pc_test_dataset = get_pointcloud_datasets(cfg.data)
    breakpoint()


    # get max number of nodes 
    max_num_nodes = max(get_max_num_nodes(pc_train_dataset), get_max_num_nodes(pc_val_dataset), get_max_num_nodes(pc_test_dataset))
    
    train_ds, val_ds, test_ds = PaddedMatrixDataset(pc_train_dataset, max_num_nodes),\
        PaddedMatrixDataset(pc_val_dataset, max_num_nodes), PaddedMatrixDataset(pc_test_dataset, max_num_nodes)

    if cfg.training.pair_loss:
        train_ds, val_ds, test_ds = PairedDataset(train_ds), PairedDataset(val_ds), PairedDataset(test_ds)
    print(f"Number of training samples: {len(train_ds)}, val samples: {len(val_ds)}")
    
    # Initialize the DataLoader
    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, num_workers=127)
    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=127)
    test_dl = DataLoader(test_ds, batch_size=cfg.training.batch_size, num_workers=127)
    
    model = LitEGNNPointNet(cfg)
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs_ae", tracking_uri="http://127.0.0.1:5000", save_dir="./mlruns")
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    #TODO: change min delta based on experiment!
    checkpoint_callback = ModelCheckpoint(monitor="log_val_loss", mode="min")

    trainer = Trainer(accelerator=cfg.accelerator, devices=cfg.devices, logger=mlf_logger,\
        callbacks=[checkpoint_callback, EarlyStopping(monitor="log_val_loss", mode="min", patience=50, min_delta=0.05)],\
            max_epochs=cfg.training.num_epochs)
    
    if cfg.training.test:
        assert cfg.training.model_checkpoint is not None, "Need to specify which checkpoint to test"
        model = LitEGNNPointNet.load_from_checkpoint(checkpoint_path=cfg.training.model_checkpoint)
        
        test_results = trainer.test(model, test_dl)
        print("Test results", test_results)
    elif cfg.training.cont and cfg.training.model_checkpoint is not None:
        print("Starting from pre-trained checkpoint...\n")
        trainer.fit(model, train_dl, val_dl, ckpt_path=cfg.training.model_checkpoint)
    else:
        trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    print("\n <<Don't forget to change the loss weighting scalars!>>\n")
    main()