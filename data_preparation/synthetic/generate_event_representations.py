import multiprocessing
import os
from pathlib import Path

import cv2
import fire
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from utils.representations import VoxelGrid, events_to_voxel_grid
from utils.utils import blosc_opts

IMG_H = 384
IMG_W = 512
VOXEL_GRID_CONSTRUCTOR = VoxelGrid((5, 384, 512), True)


def generate_event_count_images_single(
    input_seq_dir, output_dir, visualize=False, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.01:0.9], generate the event count image for a Multiflow sequence
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / "count_images"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])

        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.npy"
            if output_path.exists():
                continue

            t0 = t1 - dt_us
            first_idx = np.searchsorted(time, t0, side="left")
            last_idx_p1 = np.searchsorted(time, t1, side="right")
            out = {
                "x": np.asarray(h5f["x"][first_idx:last_idx_p1]),
                "y": np.asarray(h5f["y"][first_idx:last_idx_p1]),
                "p": np.asarray(h5f["p"][first_idx:last_idx_p1]),
                "t": time[first_idx:last_idx_p1],
            }
            n_events = out["x"].shape[0]
            img_counts = np.zeros((IMG_H, IMG_W, 2), dtype=np.uint8)
            for i in range(n_events):
                img_counts[out["y"][i], out["x"][i], out["p"][i]] += 1

            # Write to disk
            np.save(str(output_path), img_counts)

            # Visualize
            if visualize:
                img_vis = np.interp(img_counts, (0, img_counts.max()), (0, 255)).astype(
                    np.uint8
                )
                img_vis = np.concatenate([img_vis, np.zeros((IMG_H, IMG_W, 1))], axis=2)
                cv2.imshow("Count Image", img_vis)
                cv2.waitKey(1)


def generate_sbt_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"{dt:.4f}" / f"event_stacks_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        x, y, p, time = (
            np.asarray(h5f["x"]),
            np.asarray(h5f["y"]),
            np.asarray(h5f["p"]),
            np.asarray(h5f["t"]),
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.h5"
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int64)
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_us_bin
                t1_bin = t0_bin + dt_us_bin
                idx0 = np.searchsorted(time, t0_bin, side="left")
                idx1 = np.searchsorted(time, t1_bin, side="right")
                x_bin = x[idx0:idx1]
                y_bin = y[idx0:idx1]
                p_bin = p[idx0:idx1] * 2 - 1

                n_events = len(x_bin)
                for i in range(n_events):
                    time_surface[y_bin[i], x_bin[i], i_bin] += p_bin[i]

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "event_stack",
                    data=time_surface,
                    shape=time_surface.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"Time Surface Bin {i}",
                        (time_surface[:, :, i] * 255).astype(np.uint8),
                    )
                    cv2.waitKey(0)


def generate_time_surface_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / "0.0200" / f"time_surfaces_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f["x"])[idxs_sorted],
            np.asarray(h5f["y"])[idxs_sorted],
            np.asarray(h5f["p"])[idxs_sorted],
            np.asarray(h5f["t"])[idxs_sorted],
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.h5"
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_us_bin
                t1_bin = t0_bin + dt_us_bin
                mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                x_bin, y_bin, p_bin, t_bin = (
                    x[mask_t],
                    y[mask_t],
                    p[mask_t],
                    time[mask_t],
                )
                n_events = len(x_bin)
                for i in range(n_events):
                    time_surface[y_bin[i], x_bin[i], 2 * i_bin + int(p_bin[i])] = (
                        t_bin[i] - t0
                    )
            time_surface = np.divide(time_surface, dt_us)

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_surface",
                    data=time_surface,
                    shape=time_surface.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"Time Surface Bin {i}",
                        (time_surface[:, :, i] * 255).astype(np.uint8),
                    )
                    cv2.waitKey(0)


def generate_voxel_grid_single(input_seq_dir, output_dir, n_bins=5, dt=0.01, **kwargs):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"{dt:.4f}" / f"voxel_grids_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)

    dt_us = dt * 1e6

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f["x"])[idxs_sorted],
            np.asarray(h5f["y"])[idxs_sorted],
            np.asarray(h5f["p"])[idxs_sorted],
            np.asarray(h5f["t"])[idxs_sorted],
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{int(t1)}.h5"
            if output_path.exists():
                continue

            t0 = t1 - dt_us
            mask_t = np.logical_and(time > t0, time <= t1)
            x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
            curr_voxel_grid = events_to_voxel_grid(
                VOXEL_GRID_CONSTRUCTOR, p_bin, t_bin, x_bin, y_bin
            )
            curr_voxel_grid = curr_voxel_grid.numpy()
            curr_voxel_grid = np.transpose(curr_voxel_grid, (1, 2, 0))

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "voxel_grid",
                    data=curr_voxel_grid,
                    shape=curr_voxel_grid.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )


def generate(
    input_dir,
    output_dir,
    representation_type,
    dts=(0.01, 0.02),
    n_bins=5,
    visualize=False,
    **kwargs,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if representation_type == "time_surface":
        generation_function = generate_time_surface_single
    elif representation_type == "voxel_grid":
        generation_function = generate_voxel_grid_single
    elif representation_type == "event_stack":
        generation_function = generate_sbt_single
    elif representation_type == "event_count":
        generation_function = generate_event_count_images_single
    else:
        raise NotImplementedError(f"No generation function for {representation_type}")

    for split in ["train", "test"]:
        split_dir = input_dir / split
        n_seqs = len(os.listdir(str(split_dir)))
        print(f"Generate representations for {split}")

        for input_seq_dir in tqdm(split_dir.iterdir(), total=n_seqs):
            for dt in dts:
                generation_function(
                    input_seq_dir, output_dir, visualize=visualize, n_bins=n_bins, dt=dt
                )


if __name__ == "__main__":
    fire.Fire(generate)
