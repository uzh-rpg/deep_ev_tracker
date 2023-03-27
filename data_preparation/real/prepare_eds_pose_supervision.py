"""
Generate input event representations for pose refinement.
Between each image, generate N representations.
"""
import multiprocessing
import os
import timeit
from pathlib import Path

import cv2
import fire
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from utils.utils import blosc_opts

IMG_H = 480
IMG_W = 640
OUTPUT_DIR = None


def generate_time_surfaces(sequence_dir, r=3, n_bins=5):
    count = 0
    sequence_dir = Path(sequence_dir)
    output_dir = sequence_dir / "events" / f"pose_{r}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Read image timestamps
    frame_ts_arr = np.genfromtxt(str(sequence_dir / "images_timestamps.txt"))

    # Read events
    events_file = h5py.File(str(sequence_dir / "events_corrected.h5"), "r")
    events_times = np.array(events_file["t"])
    print(f"Last event at: {events_times[-1]}")

    for i in tqdm(range(len(frame_ts_arr) - 1)):
        print(f"{i}/{len(frame_ts_arr)} at {timeit.default_timer()}")
        dt_us = (frame_ts_arr[i + 1] - frame_ts_arr[i]) // r
        dt_bin_us = dt_us / n_bins

        t0 = frame_ts_arr[i]
        for j in range(r):
            count += 1
            if j == r - 1:
                t1 = frame_ts_arr[i + 1]
            else:
                t1 = t0 + dt_us

            output_path = output_dir / f"{int(t1)}.h5"
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

                x_bin = np.rint(
                    np.array(events_file["x"][first_idx:last_idx_p1])
                ).astype(int)
                y_bin = np.rint(
                    np.array(events_file["y"][first_idx:last_idx_p1])
                ).astype(int)
                p_bin = np.array(events_file["p"][first_idx:last_idx_p1])
                t_bin = np.array(events_file["t"][first_idx:last_idx_p1])

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
                    **blosc_opts(complevel=1, shuffle="byte", complib="blosc:zstd"),
                )
            t0 = t1


if __name__ == "__main__":
    fire.Fire(generate_time_surfaces)
