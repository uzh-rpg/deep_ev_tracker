import torch
import torch.nn as nn


class L1Truncated(nn.Module):
    """
    L1 Loss, but zero if label is outside the patch
    """

    def __init__(self, patch_size=31):
        super(L1Truncated, self).__init__()
        self.patch_size = patch_size
        self.L1 = nn.L1Loss(reduction="none")

    def forward(self, y, y_hat):
        self.mask = (
            (torch.abs(y) <= self.patch_size / 2.0)
            .all(dim=1)
            .float()
            .detach()
            .requires_grad_(True)
        )
        loss = self.L1(y, y_hat).sum(1)
        loss *= self.mask
        return loss, self.mask


class ReprojectionError:
    def __init__(self, threshold=15):
        self.threshold = threshold

    def forward(self, projection_matrices, u_centers_hat, training=True):
        """
        :param projection_matrices: (B, T, 3, 4)
        :param u_centers_hat: (B, T, 2)
        :return: (N, T) re-projection errors, (N, T) masks
        """
        e_reproj, masks, u_centers_reproj = [], [], []

        for idx_track in range(u_centers_hat.size(0)):
            A_rows = []

            # Triangulate
            for idx_obs in range(u_centers_hat.size(1)):
                A_rows.append(
                    u_centers_hat[idx_track, idx_obs, 0]
                    * projection_matrices[idx_track, idx_obs, 2:3, :]
                    - projection_matrices[idx_track, idx_obs, 0:1, :]
                )
                A_rows.append(
                    u_centers_hat[idx_track, idx_obs, 1]
                    * projection_matrices[idx_track, idx_obs, 2:3, :]
                    - projection_matrices[idx_track, idx_obs, 1:2, :]
                )
            A = torch.cat(A_rows, dim=0)
            _, s, vh = torch.linalg.svd(A)
            X_init = vh[-1, :].view(4, 1)
            X_init = X_init / X_init[3, 0]

            # Re-project
            (
                e_reproj_track,
                mask_track,
                x_proj_track,
            ) = (
                [],
                [],
                [],
            )
            for idx_obs in range(u_centers_hat.size(1)):
                x_proj = torch.matmul(
                    projection_matrices[idx_track, idx_obs, :, :], X_init
                )
                x_proj = x_proj / x_proj[2, 0]
                x_proj_track.append(x_proj[:2, :].detach().view(1, 1, 2))
                err = torch.linalg.norm(
                    x_proj[:2, 0].view(1, 2).detach()
                    - u_centers_hat[idx_track, idx_obs, :].view(1, 2),
                    dim=1,
                )
                e_reproj_track.append(err.view(1, 1))
                mask_track.append((err < self.threshold).view(1, 1))
            e_reproj.append(torch.cat(e_reproj_track, dim=1))
            u_centers_reproj.append(torch.cat(x_proj_track, dim=1))

            mask_track = torch.cat(mask_track, dim=1)
            # if X_init[2, 0] < 0 or s[-1] > 20:
            # if s[-1] > 20:
            #     mask_track = torch.zeros_like(mask_track)
            masks.append(mask_track)

        e_reproj = torch.cat(e_reproj, dim=0)
        masks = torch.cat(masks, dim=0).detach()

        e_reproj *= masks

        if training:
            return e_reproj, masks
        else:
            u_centers_reproj = torch.cat(u_centers_reproj, dim=0)
            return e_reproj, masks, u_centers_reproj


class L2Distance(nn.Module):
    def __init__(self):
        super(L2Distance, self).__init__()

    def forward(self, y, y_hat):
        diff = y - y_hat
        diff = diff**2
        return torch.sqrt(torch.sum(diff, dim=list(range(1, len(y.size())))))
