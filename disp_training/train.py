import os
import sys
sys.path.append('../')
import logging
import hydra
import pytorch_lightning as pl
import torch

from utils.utils import *


logger = logging.getLogger(__name__)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


def propagate_keys_disp(cfg, testing=False):
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.data.patch_size = cfg.patch_size
        cfg.data.min_track_length = cfg.min_track_length
        cfg.data.min_tracks_per_sample = cfg.min_tracks_per_sample
        cfg.data.max_tracks_per_sample = cfg.max_tracks_per_sample
        cfg.data.disp_patch_range = cfg.disp_patch_range
        cfg.data.augment = cfg.augment

        cfg.model.patch_size = cfg.patch_size
        cfg.model.min_track_length = cfg.min_track_length
        cfg.model.disp_patch_range = cfg.disp_patch_range

        if not testing:
            cfg.model.n_vis = cfg.n_vis
            cfg.model.debug = cfg.debug


@hydra.main(config_path="disp_configs", config_name="m3ed_train")
def train(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys_disp(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Instantiate model and dataloaders
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    if cfg.checkpoint_path.lower() != 'none':
        # Load weights
        model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)

    data_module = hydra.utils.instantiate(cfg.data)

    # Logging
    if cfg.logging:
        training_logger = pl.loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)
    else:
        training_logger = None

    # Training schedule
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
                 pl.callbacks.ModelCheckpoint(save_top_k=-1)]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        devices=[0],
        accelerator='gpu',
        callbacks=callbacks,
        logger=training_logger
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    train()
