import numpy as np


def get_patches(representation, u_center, patch_size):
    center = np.rint(u_center).astype(int)
    h, w = representation.shape[:2]

    patch_uv = np.stack(np.meshgrid(np.arange(patch_size), np.arange(patch_size)), axis=2)
    patch_uv = patch_uv - patch_size // 2
    grid_coords = center[:, None, None, :] + patch_uv[None, :, :, :]

    assert grid_coords.min() >= 0
    assert grid_coords[:, :, :, 0].max() < w
    assert grid_coords[:, :, :, 1].max() < h

    if representation.ndim == 2:
        patches = representation[grid_coords[:, :, :, 1], grid_coords[:, :, :, 0]]
    elif representation.ndim == 3:
        patches = representation[grid_coords[:, :, :, 1], grid_coords[:, :, :, 0], :]

    return patches


def get_event_patches(representation, u_center, patch_size, disp_patch_range):
    center = np.rint(u_center).astype(int)
    h, w = representation.shape[:2]

    patch_uv = np.stack(np.meshgrid(np.arange(patch_size), np.arange(-(disp_patch_range-1), 1)), axis=2)
    patch_uv[:, :, 0] = patch_uv[:, :, 0] - patch_size // 2
    patch_uv[:, :, 1] = patch_uv[:, :, 1] + patch_size // 2
    grid_coords = center[:, None, None, :] + patch_uv[None, :, :, :]

    grid_coords[:, :, :, 1] = np.clip(grid_coords[:, :, :, 1], 0, h-1)

    assert grid_coords.min() >= 0
    assert grid_coords[:, :, :, 0].max() < w
    assert grid_coords[:, :, :, 1].max() < h

    if representation.ndim == 2:
        patches = representation[grid_coords[:, :, :, 1], grid_coords[:, :, :, 0]]
    elif representation.ndim == 3:
        patches = representation[grid_coords[:, :, :, 1], grid_coords[:, :, :, 0], :]
        
    return patches

