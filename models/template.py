import cv2
import hydra
import matplotlib
import numpy as np
import torch.optim.lr_scheduler
from pytorch_lightning import LightningModule

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.losses import *


class Template(LightningModule):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        patch_size=31,
        init_unrolls=4,
        pose_mode=False,
        debug=True,
        **kwargs,
    ):
        super(Template, self).__init__()
        self.save_hyperparameters()

        # High level model config
        self.representation = representation
        self.patch_size = patch_size
        self.debug = debug
        self.model_type = "non_global"
        self.pose_mode = pose_mode

        # Determine num channels from representation name
        if "grayscale" in representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(representation[-1])

            if "v2" in self.representation:
                self.channels_in_per_patch *= (
                    2  # V2 representations have separate channels for each polarity
                )

        # Loss Function
        self.loss = None
        self.loss_reproj = ReprojectionError(threshold=self.patch_size / 2)

        # Training variables
        self.unrolls = init_unrolls
        self.max_unrolls = max_unrolls
        self.n_vis = n_vis
        self.colormap = cm.get_cmap("inferno")
        self.graymap = cm.get_cmap("gray")

        # Validation variables
        self.epe_l2_hist = []
        self.l2 = L2Distance()

    def configure_optimizers(self):
        if not self.debug:
            opt = hydra.utils.instantiate(
                self.hparams.optimizer, params=self.parameters()
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        self.hparams.optimizer.lr,
                        total_steps=1000000,
                        pct_start=0.002,
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                    "name": "lr",
                },
            }
        else:
            return hydra.utils.instantiate(
                self.hparams.optimizer, params=self.parameters()
            )

    def forward(self, x, attn_mask=None):
        return None

    def on_train_epoch_end(self, *args):
        return

    def training_step(self, batch_dataloaders, batch_nb):
        if self.pose_mode:
            # Freeze batchnorm running values for fine-tuning
            self.reference_encoder = self.reference_encoder.eval()
            self.target_encoder = self.target_encoder.eval()
            self.reference_redir = self.reference_redir.eval()
            self.target_redir = self.target_redir.eval()
            self.joint_encoder = self.joint_encoder.eval()
            self.predictor = self.predictor.eval()

        # Determine number of tracks in batch
        nb = len(batch_dataloaders)
        if self.pose_mode:
            nt = 0
            for bl in batch_dataloaders:
                nt += bl.n_tracks
        else:
            nt = len(batch_dataloaders)

        # Preparation
        if not self.pose_mode:
            for bl in batch_dataloaders:
                bl.auto_update_center = False
        else:
            u_centers_init = []
            for bl in batch_dataloaders:
                u_centers_init.append(bl.u_centers)
            u_centers_init = (
                torch.cat(u_centers_init, dim=0).to(self.device).unsqueeze(1)
            )
            u_centers_hist = [u_centers_init]
            projection_matrices_hist = [
                torch.cat(
                    [
                        torch.from_numpy(
                            batch_dataloaders[0].camera_matrix.astype(np.float32)
                        ),
                        torch.zeros((3, 1), dtype=torch.float32),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(u_centers_init.size(0), 1, 1, 1)
                .to(self.device)
            ]

        # Unroll network
        loss_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        loss_mask_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        self.reset(nt)

        if self.pose_mode:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            i_src = 0
            for bl_src in batch_dataloaders:
                n_src_tracks = bl_src.n_tracks
                attn_mask[
                    i_src : i_src + n_src_tracks, i_src : i_src + n_src_tracks
                ] = 1
                i_src += n_src_tracks
        else:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            for i_src in range(nt):
                src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                for i_target in range(nt):
                    attn_mask[i_src, i_target] = (
                        src_path
                        == batch_dataloaders[i_target].track_path.split("/")[-3]
                    )
        attn_mask = (1 - attn_mask).bool()

        for i_unroll in range(self.unrolls):
            # Construct batched x and y for current timestep
            x, y = [], []
            for bl in batch_dataloaders:
                x_j, y_j = bl.get_next()
                x.append(x_j)
                y.append(y_j)
            x = torch.cat(x, dim=0).to(self.device)
            y = torch.cat(y, dim=0).to(self.device)

            # Inference
            y_hat = self.forward(x, attn_mask)

            # Accumulate losses
            if self.pose_mode:
                u_centers = []
                for bl in batch_dataloaders:
                    u_centers.append(bl.u_centers)
                u_centers = torch.cat(u_centers, dim=0).to(self.device)

                # Reprojection Loss
                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat.unsqueeze(1)
                )
                projection_matrices_hist.append(y.unsqueeze(1).to(self.device))

            else:
                loss, loss_mask = self.loss(y, y_hat)
                loss_total += loss
                loss_mask_total += loss_mask

            # Pass predicted flow to dataloader
            if self.pose_mode:
                idx_acc = 0
                for j in range(nb):
                    n_tracks = batch_dataloaders[j].n_tracks
                    batch_dataloaders[j].accumulate_y_hat(
                        y_hat[idx_acc : idx_acc + n_tracks, :]
                    )
                    idx_acc += n_tracks
            else:
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat[j, :])

        # Average out losses (ignoring the masked out steps)
        if not self.pose_mode:
            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()
        else:
            u_centers_hist = torch.cat(u_centers_hist, dim=1)
            projection_matrices_hist = torch.cat(projection_matrices_hist, dim=1)
            loss_total, loss_mask_total = self.loss_reproj.forward(
                projection_matrices_hist, u_centers_hist
            )

            loss_total = loss_total.sum(1)
            loss_mask_total = loss_mask_total.sum(1)

            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()

        self.log(
            "loss/train",
            loss_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        return loss_total

    def on_validation_epoch_start(self):
        # Reset distribution monitors
        self.epe_l2_hist = []
        self.track_error_hist = []
        self.feature_age_hist = []

    def validation_step(self, batch_dataloaders, batch_nb):
        # Determine number of tracks in batch
        nb = len(batch_dataloaders)
        if self.pose_mode:
            nt = 0
            for bl in batch_dataloaders:
                nt += bl.n_tracks
        else:
            nt = nb

        # Flow history visualization for first batch
        if batch_nb == 0:
            x_hat_hist = []
            x_ref_hist = []
            if not self.pose_mode:
                y_hat_total_hist = [
                    torch.zeros((nt, 1, 2), dtype=torch.float32, device="cpu")
                ]
                y_total_hist = [
                    torch.zeros((nt, 1, 2), dtype=torch.float32, device="cpu")
                ]
                x_hist = []
            else:
                loss_hist = []

        # Validation Metrics
        if not self.pose_mode:
            metrics = {
                "feature_age": torch.zeros(nb, dtype=torch.float32, device="cpu"),
                "tracking_error": [[] for _ in range(nb)],
            }

            for bl in batch_dataloaders:
                bl.auto_update_center = False
        else:
            u_centers_init = []
            for bl in batch_dataloaders:
                u_centers_init.append(bl.u_centers)
            u_centers_init = (
                torch.cat(u_centers_init, dim=0).to(self.device).unsqueeze(1)
            )
            u_centers_hist = [u_centers_init]
            projection_matrices_hist = [
                torch.cat(
                    [
                        torch.from_numpy(
                            batch_dataloaders[0].camera_matrix.astype(np.float32)
                        ),
                        torch.zeros((3, 1), dtype=torch.float32),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(u_centers_init.size(0), 1, 1, 1)
                .to(self.device)
            ]

        # Unroll network
        loss_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        loss_mask_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        self.reset(nt)

        if self.pose_mode:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            i_src = 0
            for bl_src in batch_dataloaders:
                n_src_tracks = bl_src.n_tracks
                attn_mask[
                    i_src : i_src + n_src_tracks, i_src : i_src + n_src_tracks
                ] = 1
                i_src += n_src_tracks
        else:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            for i_src in range(nt):
                src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                for i_target in range(nt):
                    attn_mask[i_src, i_target] = (
                        src_path
                        == batch_dataloaders[i_target].track_path.split("/")[-3]
                    )
        attn_mask = (1 - attn_mask).bool()

        for i_unroll in range(self.unrolls):
            # Construct x and y
            x, y = [], []
            for bl in batch_dataloaders:
                x_j, y_j = bl.get_next()
                x.append(x_j)
                y.append(y_j)
            x = torch.cat(x, dim=0).to(self.device)
            y = torch.cat(y, dim=0).to(self.device)

            # Inference
            y_hat = self.forward(x, attn_mask)

            if self.pose_mode:
                u_centers = []
                for j in range(nb):
                    u_centers.append(batch_dataloaders[j].u_centers)
                u_centers = torch.cat(u_centers, dim=0).to(self.device)

                # Reproj Loss
                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat.unsqueeze(1)
                )
                projection_matrices_hist.append(y.unsqueeze(1).to(self.device))

            else:
                loss, loss_mask = self.loss(y, y_hat)
                loss_total += loss
                loss_mask_total += loss_mask

            # Pass predicted flow to dataloader
            if self.pose_mode:
                idx_acc = 0
                for j in range(nb):
                    n_tracks = batch_dataloaders[j].n_tracks
                    batch_dataloaders[j].accumulate_y_hat(
                        y_hat[idx_acc : idx_acc + n_tracks, :]
                    )
                    idx_acc += n_tracks
            else:
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat[j, :])

            # Patch visualizations for first batch
            if batch_nb == 0:
                x_hat_hist.append(
                    torch.max(x[:, :-1, :, :], dim=1, keepdim=True)[0].detach().clone()
                )
                x_ref_hist.append(x[:, -1, :, :].unsqueeze(1).detach().clone())

            # Metrics
            if self.pose_mode is False:
                dist, y_hat_total, y_total = [], [], []
                for j in range(nb):
                    y_hat_total.append(
                        batch_dataloaders[j].u_center
                        - batch_dataloaders[j].u_center_init
                    )
                    y_total.append(
                        batch_dataloaders[j].u_center_gt
                        - batch_dataloaders[j].u_center_init
                    )
                    dist.append(
                        np.linalg.norm(
                            batch_dataloaders[j].u_center_gt
                            - batch_dataloaders[j].u_center
                        )
                    )
                y_total = torch.from_numpy(np.array(y_total))
                y_hat_total = torch.from_numpy(np.array(y_hat_total))
                dist = torch.from_numpy(np.array(dist))

                # Update feature ages
                live_track_idxs = torch.nonzero(dist < self.patch_size)
                for i in live_track_idxs:
                    metrics["feature_age"][i] = (i_unroll + 1) * 0.01
                    if self.representation == "grayscale":
                        metrics["feature_age"] *= 5
                    metrics["tracking_error"][i].append(dist[i].item())

                # Flow history visualization for first batch
                if batch_nb == 0:
                    y_total_hist.append(
                        y_total.detach().unsqueeze(1).clone()
                    )  # Introduce time axis
                    y_hat_total_hist.append(y_hat_total.detach().unsqueeze(1).clone())

        # Log loss for both training modes
        if not self.pose_mode:
            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()
        else:
            u_centers_hist = torch.cat(u_centers_hist, dim=1)
            projection_matrices_hist = torch.cat(projection_matrices_hist, dim=1)
            loss_total, loss_mask_total, u_centers_reproj = self.loss_reproj.forward(
                projection_matrices_hist, u_centers_hist, training=False
            )
            loss_hist = loss_total.clone()
            loss_total = loss_total.sum(1)
            loss_mask_total = loss_mask_total.sum(1)
            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()

        self.log(
            "loss/val",
            loss_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        # Log predicted patches for both training modes
        if batch_nb == 0:
            x_hat_hist = torch.cat(x_hat_hist, dim=1)
            x_ref_hist = torch.cat(x_ref_hist, dim=1)

            with plt.style.context("ggplot"):
                for i_vis in range(self.n_vis):
                    # Patches
                    patch_traj_hat = x_hat_hist[i_vis, :, :, :].cpu().squeeze(0)
                    patch_traj_hat_list = list(patch_traj_hat)

                    # Reference
                    patch_ref = x_ref_hist[i_vis, :, :, :].cpu().squeeze(0)
                    patch_ref_list = list(patch_ref)
                    img_patch_ref = torch.cat(patch_ref_list, dim=1)
                    img_patch_ref = img_patch_ref.cpu().numpy()
                    img_patch_ref = torch.from_numpy(
                        self.graymap(img_patch_ref)[:, :, :3]
                    )
                    self.logger.experiment.add_image(
                        f"time_surface_ref/patch_{i_vis}",
                        img_patch_ref,
                        self.global_step,
                        dataformats="HWC",
                    )

                    # Predicted
                    if not self.pose_mode:
                        img_patch_traj_hat = torch.cat(patch_traj_hat_list, dim=1)
                        img_patch_traj_hat = img_patch_traj_hat.cpu().numpy()
                        img_patch_traj_hat = torch.from_numpy(
                            self.colormap(img_patch_traj_hat)[:, :, :3]
                        )

                    else:
                        img_arr, err_arr = [], []
                        for i_unroll in range(len(patch_traj_hat_list)):
                            patch_traj_hat = self.colormap(
                                patch_traj_hat_list[i_unroll].cpu().numpy()
                            )[:, :, :3]

                            # Reproj Loss
                            u_center = (
                                u_centers_hist[i_vis, 1 + i_unroll, :]
                                .view(2)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            u_center_reproj = (
                                u_centers_reproj[i_vis, 1 + i_unroll, :]
                                .view(2)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            x = int(
                                round(
                                    u_center_reproj[0]
                                    - u_center[0]
                                    + self.patch_size // 2
                                )
                            )
                            y = int(
                                round(
                                    u_center_reproj[1]
                                    - u_center[1]
                                    + self.patch_size // 2
                                )
                            )
                            err_arr.append(
                                loss_hist[i_vis, 1 + i_unroll].detach().cpu().numpy()
                            )

                            r_vis = 1

                            if 0 <= x < self.patch_size and 0 <= y < self.patch_size:
                                patch_traj_hat[
                                    y - r_vis : y + r_vis + 1,
                                    x - r_vis : x + r_vis + 1,
                                    0,
                                ] = 0
                                patch_traj_hat[
                                    y - r_vis : y + r_vis + 1,
                                    x - r_vis : x + r_vis + 1,
                                    1,
                                ] = 1
                                patch_traj_hat[
                                    y - r_vis : y + r_vis + 1,
                                    x - r_vis : x + r_vis + 1,
                                    2,
                                ] = 0

                            # Highlight the Center
                            patch_traj_hat[
                                self.patch_size // 2
                                - r_vis : self.patch_size // 2
                                + r_vis
                                + 1,
                                self.patch_size // 2
                                - r_vis : self.patch_size // 2
                                + r_vis
                                + 1,
                                0,
                            ] = 1
                            patch_traj_hat[
                                self.patch_size // 2
                                - r_vis : self.patch_size // 2
                                + r_vis
                                + 1,
                                self.patch_size // 2
                                - r_vis : self.patch_size // 2
                                + r_vis
                                + 1,
                                1,
                            ] = 0
                            patch_traj_hat[
                                self.patch_size // 2
                                - r_vis : self.patch_size // 2
                                + r_vis
                                + 1,
                                self.patch_size // 2
                                - r_vis : self.patch_size // 2
                                + r_vis
                                + 1,
                                2,
                            ] = 0

                            img_arr.append(patch_traj_hat)

                        img_patch_traj_hat = np.concatenate(img_arr, axis=1)
                        img_patch_traj_hat = cv2.resize(
                            img_patch_traj_hat, fx=3, fy=3, dsize=(-1, -1)
                        )
                        img_patch_traj_hat = torch.from_numpy(img_patch_traj_hat)

                        with plt.style.context("ggplot"):
                            fig = plt.figure()
                            ax = fig.add_subplot()
                            ax.plot(np.arange(0, len(err_arr)), err_arr)
                            ax.set_xlabel("Unroll Step")

                            # Reproj Error
                            ax.set_ylabel("Reprojection Error (px)")
                            self.logger.experiment.add_figure(
                                f"reproj_error/patch_{i_vis}", fig, self.global_step
                            )

                            plt.close("all")

                    self.logger.experiment.add_image(
                        f"time_surface_traj_hat/patch_{i_vis}",
                        img_patch_traj_hat,
                        self.global_step,
                        dataformats="HWC",
                    )

        # Log GT patch visualizations and track metrics for GT supervision
        if not self.pose_mode:
            # Log scalars
            for j in range(nb):
                if len(metrics["tracking_error"][j]):
                    self.track_error_hist.append(np.mean(metrics["tracking_error"][j]))

            self.feature_age_hist += (
                metrics["feature_age"]
                .numpy()
                .reshape(
                    -1,
                )
                .tolist()
            )

            dist = self.l2(y_total, y_hat_total)
            self.epe_l2_hist += (
                dist.detach()
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
                .tolist()
            )

            # Visualize some predicted patch trajectories
            if batch_nb == 0:
                # Get gt patches
                for j in range(nb):
                    batch_dataloaders[j].reset()
                    batch_dataloaders[j].auto_update_center = True

                for i_unroll in range(self.unrolls + 1):
                    x = []
                    for j in range(nb):
                        x_j, _ = batch_dataloaders[j].get_next()
                        x.append(x_j)
                    x = torch.cat(x, dim=0).to(self.device)
                    x_hist.append(
                        torch.max(x[:, :-1, :, :], dim=1, keepdim=True)[0]
                        .detach()
                        .clone()
                    )

                # Concatenate along time axis
                y_total_hist = torch.cat(y_total_hist, dim=1)
                y_hat_total_hist = torch.cat(y_hat_total_hist, dim=1)
                x_hist = torch.cat(x_hist, dim=1)

                with plt.style.context("ggplot"):
                    for i_vis in range(self.n_vis):
                        # Flow histories
                        fig = plt.figure()
                        ax = fig.add_subplot()

                        traj = y_total_hist[i_vis, :, :].cpu().numpy()
                        traj_hat = y_hat_total_hist[i_vis, :, :].cpu().numpy()

                        ax.plot(traj[:, 0], traj[:, 1], color="g")
                        ax.plot(traj_hat[:, 0], traj_hat[:, 1], color="b")

                        plt_lims = ax.get_xlim()
                        plt_lims += ax.get_ylim()
                        plt_lims = max([abs(x) for x in plt_lims])
                        ax.set_xlim([-plt_lims, plt_lims])
                        ax.set_ylim([-plt_lims, plt_lims])
                        ax.set_xticks(
                            np.linspace(np.floor(-plt_lims), np.ceil(plt_lims), 10)
                        )
                        ax.set_yticks(
                            np.linspace(np.floor(-plt_lims), np.ceil(plt_lims), 10)
                        )

                        ax.set_aspect("equal")
                        ax.set_title(f"Val Batch 0 - Patch {i_vis}")
                        self.logger.experiment.add_figure(
                            f"cumulative_flow/patch_{i_vis}", fig, self.global_step
                        )

                        # Patches
                        patch_traj = x_hist[i_vis, :, :, :].cpu().squeeze(0)
                        img_patch_traj = torch.cat(list(patch_traj), dim=1)

                        img_patch_traj = img_patch_traj.cpu().numpy()
                        img_patch_traj = torch.from_numpy(
                            self.colormap(img_patch_traj)[:, :, :3]
                        )
                        self.logger.experiment.add_image(
                            f"time_surface_traj/patch_{i_vis}",
                            img_patch_traj,
                            self.global_step,
                            dataformats="HWC",
                        )

        return loss_total

    def on_validation_epoch_end(self):
        if self.pose_mode is False:
            # L2 error cumsum
            with plt.style.context("ggplot"):
                fig = plt.figure()
                x, counts = np.unique(self.epe_l2_hist, return_counts=True)
                y = np.cumsum(counts) / np.sum(counts)
                ax = fig.add_subplot()
                ax.plot(x, y)
                ax.set_xlabel("EPE (px)")
                ax.set_ylabel("Proportion")
                self.logger.experiment.add_figure(
                    "l2_cumsum/val", fig, self.global_step
                )
                plt.close("all")

            self.logger.experiment.add_histogram(
                "EPE_hist/val", np.array(self.epe_l2_hist), self.global_step
            )

            self.log("EPE_median/val", np.median(self.epe_l2_hist))
            self.log("TE_median/val", np.median(self.track_error_hist))
            self.log("TE_mean/val", np.mean(self.track_error_hist))
            self.log("EPE_mean/val", np.mean(self.epe_l2_hist))
            self.log("TE_std/val", np.std(self.track_error_hist))
            self.log("EPE_std/val", np.std(self.epe_l2_hist))
            self.log(f"FA_median/val", np.median(self.feature_age_hist))
            self.log(f"FA_mean/val", np.mean(self.feature_age_hist))
