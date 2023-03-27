"""
Prepare data for a subset of an Event Camera Dataset sequence
- Undistort images and events
- Create time surfaces
- Create an output directory with undistorted images, undistorted event txt, and time surfaces
"""
import os
import shutil
from glob import glob
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
from fire import Fire
from matplotlib import pyplot as plt
from pandas import read_csv
from tqdm import tqdm

from utils.utils import blosc_opts


def prepare_data(root_dir, sequence_name, start_idx, end_idx):
    sequence_dir = Path(root_dir) / sequence_name
    if not sequence_dir.exists():
        print(f"Sequence directory does not exist for {sequence_name}")
        exit()

    # Read calib
    calib_data = np.genfromtxt(str(sequence_dir / "calib.txt"))
    camera_matrix = calib_data[:4]
    distortion_coeffs = calib_data[4:]
    camera_matrix = np.array(
        [
            [camera_matrix[0], 0, camera_matrix[2]],
            [0, camera_matrix[1], camera_matrix[3]],
            [0, 0, 1],
        ]
    )
    print("Calibration loaded")

    # Create output directory
    subseq_dir = Path(root_dir) / f"{sequence_name}_{start_idx}_{end_idx}"
    subseq_dir.mkdir(exist_ok=True)

    # Undistort images
    images_dir = sequence_dir / "images_corrected"
    if not images_dir.exists():
        images_dir.mkdir()
        for img_idx, img_path in enumerate(
            tqdm(
                sorted(glob(os.path.join(str(sequence_dir / "images" / "*.png")))),
                desc="Undistorting images...",
            )
        ):
            img = cv2.imread(img_path)
            img = cv2.undistort(
                img, cameraMatrix=camera_matrix, distCoeffs=distortion_coeffs
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filename = f"frame_{str(img_idx).zfill(8)}.png"
            cv2.imwrite(os.path.join(str(images_dir / filename)), img)
    img_tmp = cv2.imread(str(images_dir / "frame_00000000.png"))
    H_img, W_img = img_tmp.shape[:2]

    # Remove first entry in image timestamps
    image_timestamps = np.genfromtxt(str(sequence_dir / "images.txt"), usecols=[0])
    image_timestamps = image_timestamps[1:]
    np.savetxt(str(sequence_dir / "images.txt"), image_timestamps)

    # Undistort events
    events_corrected_path = sequence_dir / "events_corrected.txt"
    if not events_corrected_path.exists():
        events = read_csv(
            str(sequence_dir / "events.txt"), header=None, delimiter=" "
        ).to_numpy()
        print("Raw events loaded")

        events[:, 1:3] = cv2.undistortPoints(
            events[:, 1:3].reshape((-1, 1, 2)),
            camera_matrix,
            distortion_coeffs,
            P=camera_matrix,
        ).reshape(
            (-1, 2),
        )
        events[:, 1:3] = np.rint(events[:, 1:3])

        inbounds_mask = np.logical_and(events[:, 1] >= 0, events[:, 1] < W_img)
        inbounds_mask = np.logical_and(inbounds_mask, events[:, 2] >= 0)
        inbounds_mask = np.logical_and(inbounds_mask, events[:, 2] < H_img)
        events = events[inbounds_mask, :]

        print("Events undistorted")
        np.savetxt(events_corrected_path, events, ["%.9f", "%i", "%i", "%i"])
    else:
        events = read_csv(
            str(events_corrected_path), header=None, delimiter=" "
        ).to_numpy()
    t_events = events[:, 0]

    subseq_images_dir = subseq_dir / "images_corrected"
    if not subseq_images_dir.exists():
        subseq_images_dir.mkdir()

    for i in range(start_idx, end_idx + 1):
        shutil.copy(
            str(images_dir / f"frame_{str(i).zfill(8)}.png"),
            str(subseq_images_dir / f"frame_{str(i-start_idx).zfill(8)}.png"),
        )

    # Get image dimensions
    IMG_H, IMG_W = cv2.imread(
        str(images_dir / "frame_00000001.png"), cv2.IMREAD_GRAYSCALE
    ).shape

    # Read image timestamps
    image_timestamps = np.genfromtxt(sequence_dir / "images.txt", usecols=[0])
    image_timestamps = image_timestamps[start_idx : end_idx + 1]
    np.savetxt(str(subseq_dir / "images.txt"), image_timestamps)
    print(
        f"Image timestamps are in range [{image_timestamps[0]}, {image_timestamps[-1]}]"
    )
    print(f"Event timestamps are in range [{t_events.min()}, {t_events.max()}]")

    # Copy calib and poses
    shutil.copy(str(sequence_dir / "calib.txt"), str(subseq_dir / "calib.txt"))
    shutil.copy(
        str(sequence_dir / "groundtruth.txt"), str(subseq_dir / "groundtruth.txt")
    )

    # Generate debug frames
    debug_dir = sequence_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)
    n_frames_debug = 0
    dt = 0.005
    for i in range(n_frames_debug):
        # Events
        t1 = image_timestamps[i]
        t0 = t1 - dt
        time_mask = np.logical_and(events[:, 0] >= t0, events[:, 0] < t1)
        events_slice = events[time_mask, :]

        on_mask = events_slice[:, 3] == 1
        off_mask = events_slice[:, 3] == 0
        events_slice_on = events_slice[on_mask, :]
        events_slice_off = events_slice[off_mask, :]

        # Image
        img = cv2.imread(
            str(images_dir / f"frame_{str(i).zfill(8)}.png"), cv2.IMREAD_GRAYSCALE
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img, cmap="gray")
        ax.scatter(events_slice_on[:, 1], events_slice_on[:, 2], s=5, c="green")
        ax.scatter(events_slice_off[:, 1], events_slice_off[:, 2], s=5, c="red")
        plt.show()
        fig.savefig(str(debug_dir / f"frame_{str(i).zfill(8)}.png"))
        fig.close()

    # Generate time surfaces
    for dt in [0.01, 0.02]:
        for n_bins in [1, 5]:
            dt_bin = dt / n_bins
            output_ts_dir = (
                subseq_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}"
            )
            if not output_ts_dir.exists():
                output_ts_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = subseq_dir / f"debug_events_{n_bins}"
            debug_dir.mkdir(exist_ok=True)
            for i, t1 in tqdm(
                enumerate(
                    np.arange(image_timestamps[0], image_timestamps[-1] + dt, dt)
                ),
                total=int((image_timestamps[-1] - image_timestamps[0]) / dt),
                desc="Generating time surfaces...",
            ):
                output_ts_path = (
                    output_ts_dir / f"{str(int(i * (dt * 1e6))).zfill(7)}.h5"
                )
                if output_ts_path.exists():
                    continue

                time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.float64)
                t0 = t1 - dt

                # iterate over bins
                for i_bin in range(n_bins):
                    t0_bin = t0 + i_bin * dt_bin
                    t1_bin = t0_bin + dt_bin

                    time_mask = np.logical_and(
                        events[:, 0] >= t0_bin, events[:, 0] < t1_bin
                    )
                    events_slice = events[time_mask, :]

                    for i in range(events_slice.shape[0]):
                        if (
                            0 <= events_slice[i, 2] < IMG_H
                            and 0 <= events_slice[i, 1] < IMG_W
                        ):
                            time_surface[
                                int(events_slice[i, 2]),
                                int(events_slice[i, 1]),
                                2 * i_bin + int(events_slice[i, 3]),
                            ] = (
                                events_slice[i, 0] - t0
                            )
                time_surface = np.divide(time_surface, dt)

                with h5py.File(output_ts_path, "w") as h5f_out:
                    h5f_out.create_dataset(
                        "time_surface",
                        data=time_surface,
                        shape=time_surface.shape,
                        dtype=np.float32,
                        **blosc_opts(complevel=1, shuffle="byte"),
                    )

                # Visualize
                debug_event_frame = ((time_surface[:, :, 0] > 0) * 255).astype(np.uint8)
                cv2.imwrite(
                    str(debug_dir / f"{str(int(i * dt * 1e6)).zfill(7)}.png"),
                    debug_event_frame,
                )


if __name__ == "__main__":
    Fire(prepare_data)
