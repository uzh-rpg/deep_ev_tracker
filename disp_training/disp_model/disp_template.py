import hydra
import torch.optim.lr_scheduler
from pytorch_lightning import LightningModule
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.losses import *


class DistTemplate(LightningModule):
    def __init__(self, min_track_length, n_vis=8, patch_size=31, debug=True, **kwargs):
        super(DistTemplate, self).__init__()
        self.save_hyperparameters()

        # High level model config
        self.patch_size = patch_size
        self.min_track_length = min_track_length
        self.debug = debug

        # Determine num channels from representation name
        self.channels_in_per_patch = 1

        # Loss Function
        self.loss = None

        # Training variables
        self.n_vis = n_vis
        self.colormap = cm.get_cmap('inferno')
        self.graymap = cm.get_cmap('gray')

    def create_attention_mask(self, seq_frame_idx):
        attn_mask = torch.from_numpy(seq_frame_idx[:, None] == seq_frame_idx[None, :]).to(self.device)
        attn_mask = torch.logical_not(attn_mask).bool()

        return attn_mask

    def configure_optimizers(self):
        if not self.debug:
            opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
            return {'optimizer': opt,
                    'lr_scheduler': {
                        "scheduler": torch.optim.lr_scheduler.OneCycleLR(opt, self.hparams.optimizer.lr,
                                                                         total_steps=1000000,
                                                                         pct_start=0.002),
                        "interval": "step",
                        "frequency": 1,
                        "strict": True,
                        "name": "lr"}
                    }
        else:
            return hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())

    def forward(self, frame_patches, event_patches, attn_mask=None):
        return None

    def on_train_epoch_end(self, *args):
        return

    def training_step(self, batch_sample, batch_nb):
        # Get data
        seq_frame_patches, seq_event_patches, seq_y_disps, seq_frame_idx = batch_sample
        n_samples = seq_frame_patches.shape[0]

        # Create attention mask for frame attention module
        attn_mask = self.create_attention_mask(seq_frame_idx)

        seq_frame_patches = torch.from_numpy(seq_frame_patches).permute([0, 1, 4, 2, 3]).to(self.device)
        seq_event_patches = torch.from_numpy(seq_event_patches).permute([0, 1, 4, 2, 3]).to(self.device)
        seq_y_disps = torch.from_numpy(seq_y_disps).to(self.device)

        # Unroll network
        loss_total = torch.zeros(n_samples, dtype=torch.float32, device=self.device)
        self.reset(n_samples)

        for i_unroll in range(self.min_track_length):
            # Get data
            frame_patches = seq_frame_patches[:, i_unroll, :, :, :]
            event_patches = seq_event_patches[:, i_unroll, :, :, :]
            y_disps = seq_y_disps[:, i_unroll]

            # Inference
            y_disp_pred = self.forward(frame_patches, event_patches, attn_mask)

            # Accumulate losses
            loss = self.loss(y_disps, y_disp_pred[:, 1])
            loss_total += loss

        loss_total = loss_total.mean() / self.min_track_length

        self.log("loss/train", loss_total, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        return loss_total

    def on_validation_epoch_start(self):
        self.metrics = {'disp_error': []}

    def validation_step(self, batch_sample, batch_nb):
        # Get data
        seq_frame_patches, seq_event_patches, seq_y_disps, seq_frame_idx = batch_sample
        n_samples = seq_frame_patches.shape[0]

        seq_frame_patches = torch.from_numpy(seq_frame_patches).permute([0, 1, 4, 2, 3]).to(self.device)
        seq_event_patches = torch.from_numpy(seq_event_patches).permute([0, 1, 4, 2, 3]).to(self.device)
        seq_y_disps = torch.from_numpy(seq_y_disps).to(self.device)

        # Flow history visualization for first batch
        if batch_nb == 0:
            x_hat_hist = []
            x_ref_hist = []

        # Unroll network
        loss_total = torch.zeros(n_samples, dtype=torch.float32, device=self.device)
        self.reset(n_samples)

        # Create attention mask for frame attention module
        attn_mask = self.create_attention_mask(seq_frame_idx)

        # Rollout
        seq_disp_error = np.zeros([n_samples, self.min_track_length])
        for i_unroll in range(self.min_track_length):
            # Construct x and y
            # Get data
            frame_patches = seq_frame_patches[:, i_unroll, :, :, :]
            event_patches = seq_event_patches[:, i_unroll, :, :, :]
            y_disps = seq_y_disps[:, i_unroll]

            # Inference
            y_disp_pred = self.forward(frame_patches, event_patches, attn_mask)

            loss = self.loss(y_disps, y_disp_pred[:, 1])
            loss_total += loss

            # Patch visualizations for first batch
            if batch_nb == 0:
                x_hat_hist.append(torch.max(event_patches[0, :, :, :], dim=0, keepdim=True)[0].detach().clone())
                x_ref_hist.append(frame_patches[0, 0, None, :, :].detach().clone())

            seq_disp_error[:, i_unroll] = torch.abs(y_disps - y_disp_pred[:, 1]).detach().cpu().numpy()

        self.metrics['disp_error'].append(seq_disp_error)

        loss_total = loss_total.mean() / self.min_track_length
        # Log loss for both training modes
        self.log("loss/val", loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

        # Log predicted patches for both training modes
        if batch_nb == 0:
            for i_vis in range(self.n_vis):
                # Patches
                ev_patch = x_hat_hist[i_vis].cpu().squeeze(0).numpy()
                ev_patch = torch.from_numpy(self.colormap(ev_patch)[:, :, :3])
                self.logger.experiment.add_image(f'input/event_patch_{i_vis}',
                                                 ev_patch,
                                                 self.global_step, dataformats='HWC')

                # Reference
                img_patch = x_ref_hist[i_vis].cpu().squeeze(0).numpy()
                img_patch = torch.from_numpy(self.graymap(img_patch)[:, :, :3])
                self.logger.experiment.add_image(f'input/frame_patch_{i_vis}',
                                                 img_patch,
                                                 self.global_step, dataformats='HWC')

        return loss_total

    def on_validation_epoch_end(self):
        # Disparity error visualization
        disp_errors = np.concatenate(self.metrics['disp_error'], axis=0)

        # Cumulative Error Plot
        with plt.style.context('ggplot'):
            fig = plt.figure()
            x, counts = np.unique(disp_errors[:, -1], return_counts=True)
            y = np.cumsum(counts) / np.sum(counts)
            ax = fig.add_subplot()
            ax.plot(x, y)
            ax.set_xlabel('EPE (px)')
            ax.set_ylabel('Proportion')
            self.logger.experiment.add_figure("cumulative_error/val", fig, self.global_step)
            plt.close("all")

        # Mean Error Plot
        with plt.style.context('ggplot'):
            fig = plt.figure()
            mean_error = disp_errors.mean(axis=0)

            y = mean_error
            x = np.arange(mean_error.shape[0])
            ax = fig.add_subplot()
            ax.plot(x, y)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Error')
            self.logger.experiment.add_figure("mean_error_seq/val", fig, self.global_step)
            plt.close("all")

        self.log("mean_epe/val", np.mean(disp_errors[:, -1]))
        self.log("var_epe/val", np.var(disp_errors[:, -1]))
        self.log("mean_error/val", np.mean(disp_errors))
        self.log("var_error/val", np.var(disp_errors))
