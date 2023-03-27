"""
EXTRACT:
Convert colmap's images.txt file to a stamped_groundtruth poses file
colmap entries are:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)

stamped_ground_truth pose entries are:
#timestamp[seconds] px py pz qx qy qz qw

GENERATE:
Create the images.txt file for colmap
Create an empty points.txt file
Create the cameras.txt file with a single camera formatted like :
    (1 PINHOLE 640 480 766.536025127154 767.5749459126396 291.0503512057777 227.4060484950132)
"""


import os
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation

from utils.dataset import ECPoseSegmentDataset, EDSPoseSegmentDataset


def read_colmap_data(colmap_data_path, skip_header=4):
    """
    Returns the colmap data from images.txt file
    :param colmap_data_path: path to images.txt
    :param skip_header: how many columns to skip
    :return: image_indices, poses (x, y, z, qx, qy, qz, qw)
    """
    pose_data = []
    with open(colmap_data_path, "r") as colmap_data_f:
        for i_row, data_row in enumerate(colmap_data_f):
            if (i_row > skip_header) and (i_row - skip_header) % 2 == 0:
                data_row = data_row.split(" ")
                pose_data.append([data_row[i] for i in [-1, 5, 6, 7, 2, 3, 4, 1]])
    pose_data = sorted(pose_data, key=lambda x: x[0])
    image_idxs = [
        int(pose[0].replace("frame_", "").replace(".png\n", "")) for pose in pose_data
    ]
    pose_data = np.array([pose[1:] for pose in pose_data]).astype(np.float32)
    return image_idxs, pose_data


def extract(sequence_dir, dataset_type):
    assert dataset_type in ["EC", "EDS"], "Dataset type must be one of EC, EDS"
    if dataset_type == "EC":
        dataset_class = ECPoseSegmentDataset
    else:
        dataset_class = EDSPoseSegmentDataset
    sequence_dir = Path(sequence_dir)

    # Read image timestamps
    image_ts = dataset_class.get_frame_timestamps(sequence_dir).reshape((-1, 1))

    # Read colmap poses
    colmap_data_path = sequence_dir / "colmap" / "images.txt"
    image_idxs, colmap_data = read_colmap_data(str(colmap_data_path))

    # Invert the poses bc colmap transforms are world->camera instead of camera->world
    inverted_poses = []
    for pose in colmap_data:
        T_C_W = np.eye(4)
        T_C_W[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        T_C_W[0, 3] = pose[0]
        T_C_W[1, 3] = pose[1]
        T_C_W[2, 3] = pose[2]
        T_W_C = np.linalg.inv(T_C_W)
        quat = Rotation.from_matrix(T_W_C[:3, :3]).as_quat()
        inverted_poses.append(
            [T_W_C[0, 3], T_W_C[1, 3], T_W_C[2, 3], quat[0], quat[1], quat[2], quat[3]]
        )
    inverted_poses = np.array(inverted_poses)

    colmap_poses = np.concatenate([image_ts[image_idxs, 0:1], inverted_poses], axis=1)

    output_path = sequence_dir / "colmap" / "stamped_groundtruth.txt"
    np.savetxt(
        str(output_path),
        colmap_poses,
        header="#timestamp[seconds] px py pz qx qy qz qw",
    )


def generate(sequence_dir, dataset_type):
    assert dataset_type in ["EC", "EDS"], "Dataset type must be one of EC, EDS"
    if dataset_type == "EC":
        dataset_class = ECPoseSegmentDataset
    else:
        dataset_class = EDSPoseSegmentDataset
    seq_dir = Path(sequence_dir)

    # Read poses and image names
    pose_interpolator = dataset_class.get_pose_interpolator(seq_dir)

    image_dir = seq_dir / "images_corrected"
    if not image_dir.exists():
        print("Rectified image directory not found")
        exit()
    else:
        image_paths = dataset_class.get_frame_paths(seq_dir)
        image_names = [os.path.split(image_path)[1] for image_path in image_paths]

    image_timestamps = dataset_class.get_frame_timestamps(seq_dir)

    # Write formatted poses to images.txt
    colmap_dir = seq_dir / "colmap"
    if not colmap_dir.exists():
        colmap_dir.mkdir()

    colmap_poses_path = colmap_dir / "images.txt"
    with open(colmap_poses_path, "w") as colmap_poses_f:
        for image_idx, (image_ts, image_name) in enumerate(
            zip(image_timestamps, image_names)
        ):
            image_pose = pose_interpolator.interpolate_colmap(image_ts)
            colmap_poses_f.write(
                f"{image_idx+1} {image_pose[6]} {image_pose[3]} {image_pose[4]} {image_pose[5]} {image_pose[0]} {image_pose[1]} {image_pose[2]} 1 {image_name}\n\n"
            )

    # Write empty points file
    with open(str(colmap_dir / "points3D.txt"), "w") as _:
        pass

    # Write cameras
    camera_matrix, _, _ = dataset_class.get_calibration(seq_dir)
    with open(str(colmap_dir / "cameras.txt"), "w") as colmap_cameras_f:
        colmap_cameras_f.write(
            f"1 PINHOLE {dataset_class.resolution[0]} {dataset_class.resolution[1]} "
            f"{camera_matrix[0, 0]} {camera_matrix[1, 1]} {camera_matrix[0, 2]} {camera_matrix[1, 2]}"
        )


if __name__ == "__main__":
    fire.Fire({"generate": generate, "extract": extract})
