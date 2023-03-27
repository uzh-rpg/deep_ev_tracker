import numpy as np
import torch


def array_to_tensor(array):
    # Get patch inputs
    array = np.array(array)
    if len(array.shape) == 2:
        array = np.expand_dims(array, 0)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def get_patch(time_surface, u_center, patch_size):
    center = np.rint(u_center).astype(int)
    h, w = time_surface.shape
    c = 1

    # Check out-of-bounds
    if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
        return torch.zeros((c, patch_size, patch_size), dtype=torch.float32)

    r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

    x = np.array(time_surface[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]).astype(
        np.float32
    )
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    x = torch.nn.functional.pad(x, pad_lrud)
    return x


def compute_padding(center, patch_size, resolution):
    """
    Return patch crop area and required padding
    :param center: Integer center coordinates of desired patch crop
    :param resolution: Image res (w, h)
    :return:
    """
    w, h = resolution

    # Crop around the patch
    r_min = int(max(0, center[1] - patch_size // 2))
    r_max = int(min(h - 1, center[1] + patch_size // 2 + 1))
    c_min = int(max(0, center[0] - patch_size // 2))
    c_max = int(min(w - 1, center[0] + patch_size // 2 + 1))

    # Determine padding
    pad_l, pad_r, pad_u, pad_d = 0, 0, 0, 0
    if center[1] - patch_size // 2 < 0:
        pad_u = abs(center[1] - patch_size // 2)
    if center[1] + patch_size // 2 + 1 > h - 1:
        pad_d = center[1] + patch_size // 2 + 1 - (h - 1)
    if center[0] - patch_size // 2 < 0:
        pad_l = abs(center[0] - patch_size // 2)
    if center[0] + patch_size // 2 + 1 > w - 1:
        pad_r = center[0] + patch_size // 2 + 1 - (w - 1)

    return (
        (r_min, r_max),
        (c_min, c_max),
        (int(pad_l), int(pad_r), int(pad_u), int(pad_d)),
    )


def get_patch_tensor(input_tensor, center, patch_size):
    """

    :param input_tensor: (1, c, h, w)
    :param u_center:
    :param patch_size:
    :return:
    """
    # center = np.rint(u_center).astype(int)
    _, c, h, w = input_tensor.shape

    # Check out-of-bounds
    if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
        return torch.zeros(
            (1, c, patch_size, patch_size),
            dtype=torch.float32,
            device=input_tensor.device,
        )

    r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

    x = input_tensor[0:1, :, r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]
    x = torch.nn.functional.pad(x, pad_lrud)
    return x


def get_patch_voxel(voxel_grid, u_center, patch_size):
    center = np.rint(u_center).astype(int).reshape((2,))
    if len(voxel_grid.shape) == 2:
        c = 1
        h, w = voxel_grid.shape
    else:
        h, w, c = voxel_grid.shape

    # Check out-of-bounds
    if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
        return torch.zeros((c, patch_size, patch_size), dtype=torch.float32)

    r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

    if len(voxel_grid.shape) == 2:
        x = np.array(voxel_grid[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]).astype(
            np.float32
        )
        x = np.expand_dims(x, axis=2)
    else:
        x = np.array(
            voxel_grid[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1], :]
        ).astype(np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    x = torch.nn.functional.pad(x, pad_lrud)
    return x


def get_patch_voxel2(voxel_grid, u_center, patch_size, padding=10):
    """
    get_patch_voxel but using extract glimpse (no roundidng of center coords)
    :param voxel_grid:
    :param u_center:
    :param patch_size:
    :return: (C, P, P)
    """
    # Extract expanded patches from the h5 file
    u_center = u_center.reshape((2,))
    u_center_rounded = np.rint(u_center)

    u_center_offset = u_center - u_center_rounded + ((patch_size + padding) // 2.0)
    x_patch_expanded = get_patch_voxel(
        voxel_grid, u_center_rounded.reshape((-1,)), patch_size + padding
    ).unsqueeze(0)
    return extract_glimpse(
        x_patch_expanded,
        (patch_size, patch_size),
        torch.from_numpy(u_center_offset.astype(np.float32)).view((1, 2)) + 0.5,
        mode="bilinear",
    ).squeeze(0)


def get_patch_voxel_pairs(voxel_grid_0, voxel_grid_1, u_center, patch_size):
    center = np.rint(u_center).astype(int)
    if len(voxel_grid_0.shape) == 2:
        c = 1
        h, w = voxel_grid_0.shape
    else:
        h, w, c = voxel_grid_0.shape

    # Check out-of-bounds
    if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
        return torch.zeros((c * 2, patch_size, patch_size), dtype=torch.float32)

    r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

    if len(voxel_grid_0.shape) == 2:
        x0 = np.array(
            voxel_grid_0[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]
        ).astype(np.float32)
        x1 = np.array(
            voxel_grid_1[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]
        ).astype(np.float32)
        x0 = np.expand_dims(x0, axis=2)
        x1 = np.expand_dims(x1, axis=2)
    else:
        x0 = np.array(
            voxel_grid_0[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1], :]
        ).astype(np.float32)
        x1 = np.array(
            voxel_grid_1[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1], :]
        ).astype(np.float32)
    x = np.concatenate([x0, x1], axis=2)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    x = torch.nn.functional.pad(x, pad_lrud)
    return x


def get_patch_pairs(time_surface_0, time_surface_1, u_center, patch_size):
    center = np.rint(u_center).astype(int)
    h, w = time_surface_0.shape
    c = 1

    # Check out-of-bounds
    if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
        return torch.zeros((c * 2, patch_size, patch_size), dtype=torch.float32)

    r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

    x0 = np.array(time_surface_0[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]).astype(
        np.float32
    )
    x1 = np.array(time_surface_1[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]).astype(
        np.float32
    )
    x0 = np.expand_dims(x0, axis=0)
    x1 = np.expand_dims(x1, axis=0)
    x = np.concatenate([x0, x1], axis=0)
    x = torch.from_numpy(x)
    x = torch.nn.functional.pad(x, pad_lrud)
    return x


def extract_glimpse(
    input,
    size,
    offsets,
    centered=False,
    normalized=False,
    mode="nearest",
    padding_mode="zeros",
):
    """Returns a set of windows called glimpses extracted at location offsets
    from the input tensor. If the windows only partially overlaps the inputs,
    the non-overlapping areas are handled as defined by :attr:`padding_mode`.
    Options of :attr:`padding_mode` refers to `torch.grid_sample`'s document.
    The result is a 4-D tensor of shape [N, C, h, w].  The channels and batch
    dimensions are the same as that of the input tensor.  The height and width
    of the output windows are specified in the size parameter.
    The argument normalized and centered controls how the windows are built:
        * If the coordinates are normalized but not centered, 0.0 and 1.0 correspond
          to the minimum and maximum of each height and width dimension.
        * If the coordinates are both normalized and centered, they range from
          -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper left
          corner, the lower right corner is located at (1.0, 1.0) and the center
          is at (0, 0).
        * If the coordinates are not normalized they are interpreted as numbers
          of pixels.
    Args:
        input (Tensor): A Tensor of type float32. A 4-D float tensor of shape
            [N, C, H, W].
        size (tuple): 2-element integer tuple specified the
            output glimpses' size. The glimpse height must be specified first,
            following by the glimpse width.
        offsets (Tensor): A Tensor of type float32. A 2-D integer tensor of
            shape [N, 2]  containing the x, y locations of the center
            of each window.
        centered (bool, optional): An optional bool. Defaults to True. indicates
            if the offset coordinates are centered relative to the image, in
            which case the (0, 0) offset is relative to the center of the input
            images. If false, the (0,0) offset corresponds to the upper left
            corner of the input images.
        normalized (bool, optional): An optional bool. Defaults to True. indicates
            if the offset coordinates are normalized.
        mode (str, optional): Interpolation mode to calculate output values.
            Defaults to 'bilinear'.
        padding_mode (str, optional): padding mode for values outside the input.
    Raises:
        ValueError: When normalized set False but centered set True
    Returns:
        output (Tensor): A Tensor of same type with input.
    """
    W, H = input.size(-1), input.size(-2)

    if normalized and centered:
        offsets = (offsets + 1) * offsets.new_tensor([W / 2, H / 2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W, H])
    elif centered:
        raise ValueError("Invalid parameter that offsets centered but not normlized")

    h, w = size
    xs = torch.arange(0, w, dtype=input.dtype, device=input.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=input.dtype, device=input.device) - (h - 1) / 2.0

    # vy, vx = torch.meshgrid(ys, xs)
    vy, vx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])
    ) / offsets_grid.new_tensor([W / 2, H / 2])

    return torch.nn.functional.grid_sample(
        input, offsets_grid, mode=mode, align_corners=True, padding_mode=padding_mode
    )
