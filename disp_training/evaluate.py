import logging

import os
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import sys
import tqdm

sys.path.append('../')

from utils.utils import *
from disp_dataloader.m3ed_loader import M3EDTestDataModule


logger = logging.getLogger(__name__)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


def propagate_keys_disp(cfg):
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.data.patch_size = cfg.patch_size
        cfg.data.min_track_length = cfg.min_track_length
        cfg.data.tracks_per_sample = cfg.tracks_per_sample
        cfg.data.disp_patch_range = cfg.disp_patch_range

        cfg.model.patch_size = cfg.patch_size
        cfg.model.min_track_length = cfg.min_track_length
        cfg.model.tracks_per_sample = cfg.tracks_per_sample
        cfg.model.disp_patch_range = cfg.disp_patch_range


def create_attn_mask(seq_frame_idx, device):
    attn_mask = torch.from_numpy(seq_frame_idx[:, None] == seq_frame_idx[None, :]).to(device)
    attn_mask = torch.logical_not(attn_mask).bool()

    return attn_mask


def test_run(model, dataloader, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    model = model.to(device)

    # Create attention mask
    n_samples = cfg.data.batch_size * cfg.tracks_per_sample
    attn_mask = None

    # Iterate over test dataset
    list_disp_pred, list_disp_gt, list_seq_names, list_img_points = [], [], [], []
    for batch_sample in tqdm(dataloader):
        seq_frame_patches, seq_event_patches, seq_y_gt_disp_samples, seq_frame_idx, seq_names, img_points  = batch_sample
        seq_frame_patches = torch.from_numpy(seq_frame_patches).permute([0, 1, 4, 2, 3]).to(device)
        seq_event_patches = torch.from_numpy(seq_event_patches).permute([0, 1, 4, 2, 3]).to(device)
        n_samples = seq_frame_patches.shape[0]

        assert cfg.min_track_length == seq_frame_patches.shape[1]
        y_pred_disp_samples = np.zeros([n_samples, cfg.min_track_length])

        if attn_mask is None or seq_frame_patches.shape[0] != attn_mask.shape[0]:
            attn_mask = create_attn_mask(seq_frame_idx, device)

        model.reset(None)
        for i_unroll in range(cfg.min_track_length):
            frame_patches = seq_frame_patches[:, i_unroll, :, :, :]
            event_patches = seq_event_patches[:, i_unroll, :, :, :]

            # Inference
            y_disp_pred = model.forward(frame_patches, event_patches, attn_mask)
            y_pred_disp_samples[:, i_unroll] = y_disp_pred[:, 1].detach().cpu().numpy()

        list_disp_pred.append(y_pred_disp_samples)
        list_disp_gt.append(seq_y_gt_disp_samples)
        list_seq_names.append(seq_names)
        list_img_points.append(img_points)

    # Save results
    np.savez_compressed('results.npz',
                        disparity_pred=np.concatenate(list_disp_pred, axis=0),
                        seq_names=np.concatenate(list_seq_names, axis=0).flatten())

    np.savez_compressed('ground_truth.npz',
                        disparity_gt=np.concatenate(list_disp_gt, axis=0),
                        image_points=np.concatenate(list_img_points, axis=0),
                        seq_names=np.concatenate(list_seq_names, axis=0).flatten())


@hydra.main(config_path="disp_configs", config_name="m3ed_test")
def test(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys_disp(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    with open('test_config.yaml', 'w') as outfile:
        OmegaConf.save(cfg, outfile)

    # Instantiate model
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    if cfg.checkpoint_path.lower() == 'none':
        print("Provide Checkpoints")

    # Load weights
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    data_module = M3EDTestDataModule(**cfg.data)
    data_module.setup()
    dataloader = data_module.test_dataloader()

    with torch.no_grad():
        test_run(model, dataloader, cfg)


if __name__ == '__main__':
    test()
