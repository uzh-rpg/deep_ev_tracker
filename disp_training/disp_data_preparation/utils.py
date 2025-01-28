import cv2
import numpy as np
import matplotlib


def compute_rectify_map(target_group, source_group):
    """Adapted from: https://github.com/daniilidis-group/m3ed/blob/main/build_system/semantics/internimage.py"""
    target_T_to_prophesee_left = target_group['T_to_prophesee_left'][...]
    source_T_to_prophesee_left = source_group['T_to_prophesee_left'][...]

    source_T_target = source_T_to_prophesee_left @ np.linalg.inv( target_T_to_prophesee_left )
    target_dist_coeffs = target_group['distortion_coeffs'][...]
    target_intrinsics = target_group['intrinsics'][...]
    target_res = target_group['resolution'][...]
    target_Size = target_res

    target_K = np.eye(3)
    target_K[0,0] = target_intrinsics[0]
    target_K[1,1] = target_intrinsics[1]
    target_K[0,2] = target_intrinsics[2]
    target_K[1,2] = target_intrinsics[3]

    target_P = np.zeros((3,4))
    target_P[:3,:3] = target_K

    source_dist_coeffs = source_group['distortion_coeffs'][...]
    source_intrinsics = source_group['intrinsics'][...]
    source_res = source_group['resolution'][...]
    source_Size = source_res

    source_K = np.eye(3)
    source_K[0,0] = source_intrinsics[0]
    source_K[1,1] = source_intrinsics[1]
    source_K[0,2] = source_intrinsics[2]
    source_K[1,2] = source_intrinsics[3]

    # Image is already undistorted, this only works for the M3ED loading
    target_dist_coeffs *= 0
    out = cv2.stereoRectify(cameraMatrix1=source_K,
                            distCoeffs1=source_dist_coeffs,
                            cameraMatrix2=target_K,
                            distCoeffs2=target_dist_coeffs,
                            imageSize=target_Size,
                            newImageSize=target_Size,
                            T=source_T_target[:3, 3],
                            R=source_T_target[:3, :3],
                            alpha=0,
                            )
    rot_source, rot_target, proj_source, proj_target, Q, validPixROI1, validPixROI2 = out
    map_target = np.stack(cv2.initUndistortRectifyMap(target_K, target_dist_coeffs, rot_target, proj_target, source_Size, cv2.CV_32FC1), axis=-1)
    map_source = np.stack(cv2.initUndistortRectifyMap(source_K, source_dist_coeffs, rot_source, proj_source, source_Size, cv2.CV_32FC1), axis=-1)

    inv_map_target = np.stack(cv2.initInverseRectificationMap(target_K, target_dist_coeffs, rot_target, proj_target, target_Size, cv2.CV_32FC1), axis=-1)
    inv_map_source = invert_map(map_source)

    return map_target, map_source, inv_map_target, inv_map_source, proj_target, proj_source, rot_target, rot_source, Q


def invert_map(F):
    # shape is (h, w, 2), a "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P


def reproject_points(u, v, depth_image, T_C_W, K, z=None):
    if z is None:
        z = depth_image[v.astype('int'), u.astype('int')]
    valid_mask = z != 0

    u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
    p = np.stack([u, v, np.ones([u.shape[0]])], axis=1)
    P_C = np.matmul(np.linalg.inv(K)[None, :, :], p[:, :, None]) * z[:, None, None]
    P_W = np.matmul(np.linalg.inv(T_C_W),
                    np.concatenate([P_C, np.ones([P_C.shape[0], 1, 1])], axis=1))

    return P_W, valid_mask


def project_points(P_W, T_C_W, K):
    assert T_C_W.ndim == 2
    P_C = np.matmul(T_C_W[:3, :], P_W[:, :, None])
    in_front_camera_mask = P_C[:, 2, 0] > 0
    P_C = P_C[in_front_camera_mask, :, :]

    points = np.matmul(K[:, :], P_C).squeeze(-1)
    points = points / points[:, 2, None]

    return points, in_front_camera_mask


def visualize_depth_image(depth_image, rgb_image, file_path):
    rgb_image = rgb_image.copy()
    cmap = matplotlib.colormaps.get_cmap('gist_ncar')
    v, u = np.nonzero(depth_image != 0)
    img_depth = depth_image[v, u]
    depth_colors = cmap((img_depth - img_depth.min()) / (img_depth.max() - img_depth.min()))[:, :3]
    rgb_image[v, u, :] = depth_colors * 255
    cv2.imwrite(file_path, rgb_image)


def remap_events(events, map, rotate, shape=None, valid_region=None):
    mx, my = map
    x, y = mx[events['y'], events['x']], my[events['y'], events['x']]
    p = np.array(events['p'])
    t = np.array(events['t'])

    if rotate:
        target_width, target_height = shape
        x = target_width - 1 - x
        y = target_height - 1 - y

    if valid_region is not None:
        mask = ((x >= valid_region[0]) & (x < valid_region[2]) &
                (y >= valid_region[1]) & (y < valid_region[3]))
        x = x - valid_region[0]
        y = y - valid_region[1]
    else:
        target_width, target_height = shape
        mask = (x >= 0) & (x <= target_width - 1) & (y >= 0) & (y <= target_height - 1)

    return {'x': x[mask], 'y': y[mask], 't': t[mask], 'p': p[mask]}
