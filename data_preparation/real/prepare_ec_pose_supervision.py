"""
Generate input event representations for pose refinement.
Between each image, generate N representations.
"""

import multiprocessing
import os
from pathlib import Path

import cv2
import fire
import h5py
import hdf5plugin
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from tqdm import tqdm

from utils.utils import blosc_opts

IMG_H = 180
IMG_W = 240
OUTPUT_DIR = None


def generate_time_surfaces(sequence_dir, r=5, n_bins=5):
    sequence_dir = Path(sequence_dir)
    output_dir = sequence_dir / "events" / f"pose_{r}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Read image timestamps
    frame_ts_arr = np.genfromtxt(sequence_dir / "images.txt", usecols=[0])

    # Read events
    events = read_csv(
        str(sequence_dir / "events_corrected.txt"), delimiter=" "
    ).to_numpy()
    events_times = events[:, 0] * 1e6

    # Debug images
    debug_dir = sequence_dir / "events" / f"pose_{r}_debug"
    if not debug_dir.exists():
        debug_dir.mkdir()

    # Generate time surfaces
    idx_ts = 1
    for i in tqdm(range(len(frame_ts_arr) - 1)):
        dt_us = (frame_ts_arr[i + 1] - frame_ts_arr[i]) * 1e6 // r
        dt_bin_us = dt_us / n_bins

        t0 = frame_ts_arr[i] * 1e6
        for j in range(r):
            if j == r - 1:
                t1 = frame_ts_arr[i + 1] * 1e6
            else:
                t1 = t0 + dt_us

            output_path = output_dir / f"{int(t1)}.h5"
            idx_ts += 1
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)

            # iterate over bins
            for i_bin in range(5):
                t0_bin = t0 + i_bin * dt_bin_us
                if i_bin == 4:
                    t1_bin = t1
                else:
                    t1_bin = t0_bin + dt_bin_us

                first_idx = np.searchsorted(events_times, t0_bin, side="left")
                last_idx_p1 = np.searchsorted(events_times, t1_bin, side="right")

                x_bin = np.rint(np.array(events[first_idx:last_idx_p1, 1])).astype(int)
                y_bin = np.rint(np.array(events[first_idx:last_idx_p1, 2])).astype(int)
                p_bin = np.array(events[first_idx:last_idx_p1, 3])
                t_bin = np.array(events[first_idx:last_idx_p1, 0]) * 1e6

                n_events = len(x_bin)
                for i_e in range(n_events):
                    time_surface[
                        y_bin[i_e], x_bin[i_e], 2 * i_bin + int(p_bin[i_e])
                    ] = (t_bin[i_e] - t0)
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

            t0 = t1


if __name__ == "__main__":
    fire.Fire(generate_time_surfaces)
