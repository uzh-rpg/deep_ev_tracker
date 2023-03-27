import os
from glob import glob
from pathlib import Path
from shutil import copy

import fire
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.utils import blosc_opts

IMG_H = 480
IMG_W = 640


def generate_subseq(seq_name, start_idx, end_idx, dt):
    """
    Generate a subsequence of an EDS sequence with folder structure:
        events
            <dt>
                <event_representation>
                    0000000.h5
                    ...
        images
            frame_<start_idx>.png
            frame_<start_idx+1>.png
            ...
        image_timestamps.txt
        stamped_groundtruth.txt
    :param seq_name:
    :param start_idx: Image index to start
    :param end_idx: Terminal image index (non-inclusive)
    :param dt: time delete used for time surface generation
    """

    # Pathing
    input_dir = Path(f"<path>/{seq_name}")
    output_dir = Path(
        f"<path>/{seq_name}_{start_idx}_{end_idx}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Copy pose data
    copy(
        str(input_dir / "stamped_groundtruth.txt"),
        str(output_dir / "stamped_groundtruth.txt"),
    )

    # Filter timestamps
    image_timestamps = np.genfromtxt(str(input_dir / "images_timestamps.txt"))[
        start_idx:end_idx
    ]
    np.savetxt(str(output_dir / "images_timestamps.txt"), image_timestamps, fmt="%i")

    # Copy images
    output_image_dir = output_dir / "images_corrected"
    if not output_image_dir.exists():
        output_image_dir.mkdir()

    for idx in tqdm(range(start_idx, end_idx), desc="Copying images..."):
        copy(
            str(input_dir / "images_corrected" / f"frame_{str(idx).zfill(10)}.png"),
            str(
                output_dir
                / "images_corrected"
                / f"frame_{str(idx-start_idx).zfill(10)}.png"
            ),
        )

    # Generate time surfaces
    dt_us = dt * 1e6
    for n_bins in [5]:
        dt_bin_us = dt_us / n_bins
        output_ts_dir = (
            output_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}"
        )
        if not output_ts_dir.exists():
            output_ts_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(input_dir / "events_corrected.h5"), "r") as h5f:
            time = np.asarray(h5f["t"])

            for i, t1 in tqdm(
                enumerate(
                    np.arange(image_timestamps[0], image_timestamps[-1] + dt_us, dt_us)
                ),
                total=int((image_timestamps[-1] - image_timestamps[0]) / dt_us),
                desc="Generating time surfaces...",
            ):
                output_ts_path = output_ts_dir / f"{str(int(i*dt_us)).zfill(7)}.h5"
                if output_ts_path.exists():
                    continue

                time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.uint64)
                t0 = t1 - dt_us

                # iterate over bins
                for i_bin in range(n_bins):
                    t0_bin = t0 + i_bin * dt_bin_us
                    t1_bin = t0_bin + dt_bin_us

                    first_idx = np.searchsorted(time, t0_bin, side="left")
                    last_idx_p1 = np.searchsorted(time, t1_bin, side="right")
                    out = {
                        "x": np.rint(
                            np.asarray(h5f["x"][first_idx:last_idx_p1])
                        ).astype(int),
                        "y": np.rint(
                            np.asarray(h5f["y"][first_idx:last_idx_p1])
                        ).astype(int),
                        "p": np.asarray(h5f["p"][first_idx:last_idx_p1]),
                        "t": time[first_idx:last_idx_p1],
                    }
                    n_events = out["x"].shape[0]

                    for i in range(n_events):
                        time_surface[
                            out["y"][i], out["x"][i], 2 * i_bin + int(out["p"][i])
                        ] = (out["t"][i] - t0)
                time_surface = np.divide(time_surface, dt_us)
                with h5py.File(output_ts_path, "w") as h5f_out:
                    h5f_out.create_dataset(
                        "time_surface",
                        data=time_surface,
                        shape=time_surface.shape,
                        dtype=np.float32,
                        **blosc_opts(complevel=1, shuffle="byte"),
                    )

                # Visualize
                for i in range(n_bins):
                    plt.imshow((time_surface[:, :, i] * 255).astype(np.uint8))
                    plt.show()

        # Storing events in one cropped file
        first_t, last_t = image_timestamps[0], image_timestamps[-1]
        event_idx = np.searchsorted(time, np.asarray([first_t, last_t]), side="left")
        output_path = output_ts_dir / "events.h5"
        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "x", data=h5f["x"][event_idx[0] : event_idx[1]].astype(np.uint16)
            )
            h5f_out.create_dataset(
                "y", data=h5f["y"][event_idx[0] : event_idx[1]].astype(np.uint16)
            )
            h5f_out.create_dataset("p", data=h5f["p"][event_idx[0] : event_idx[1]])
            h5f_out.create_dataset("t", data=time[event_idx[0] : event_idx[1]])


if __name__ == "__main__":
    fire.Fire(generate_subseq)
