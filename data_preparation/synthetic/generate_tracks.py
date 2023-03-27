""" Script for generating feature tracks from the Multiflow dataset. """
import multiprocessing
import os
import shutil
from pathlib import Path

import cv2
import fire
import h5py
import hdf5plugin
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# MultiFlow Image Dims
IMG_H = 384
IMG_W = 512

# Corner Parameters (Global for Multiprocessing)
MAX_CORNERS = 30
MIN_DISTANCE = 30
QUALITY_LEVEL = 0.3
BLOCK_SIZE = 25
K = 0.15
USE_HARRIS_DETECTOR = False
OUTPUT_DIR = None
TRACK_NAME = "shitomasi_custom_v5"

# Filtering
MIN_TRACK_DISPLACEMENT = 5
displacements_all = []


def generate_single_track(seq_dir, dt=0.01):
    tracks = []
    dt_us = dt * 1e6

    # Get split
    split = seq_dir.parents[0].stem

    # Load reference image
    img_t0_p = seq_dir / "images" / "0400000.png"
    if img_t0_p.exists():
        img_t0 = cv2.imread(
            str(seq_dir / "images" / "0400000.png"), cv2.IMREAD_GRAYSCALE
        )
    else:
        print(f"Sequence {seq_dir} has no reference image.")
        return

    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        img_t0,
        MAX_CORNERS,
        QUALITY_LEVEL,
        MIN_DISTANCE,
        k=K,
        useHarrisDetector=USE_HARRIS_DETECTOR,
        blockSize=BLOCK_SIZE,
    )

    # Initialize tracks
    for i_track in range(corners.shape[0]):
        track = np.array([0.4, corners[i_track, 0, 0], corners[i_track, 0, 1]])
        tracks.append(track.reshape((1, 3)))

    # Read flow
    for ts_us in np.arange(400000 + dt_us, 900000 + dt_us, dt_us):
        flow_path = seq_dir / "flow" / f"0{ts_us:.0f}.h5"
        with h5py.File(str(flow_path), "r") as h5f:
            flow = np.asarray(h5f["flow"])

            for i_corner, corner in enumerate(corners):
                x_init, y_init = corners[i_corner, 0, 0], corners[i_corner, 0, 1]
                new_track_entry = np.array([ts_us * 1e-6, x_init, y_init])
                new_track_entry[1] += flow[int(y_init), int(x_init), 0]
                new_track_entry[2] += flow[int(y_init), int(x_init), 1]
                new_track_entry = new_track_entry.reshape((1, 3))
                tracks[i_corner] = np.append(tracks[i_corner], new_track_entry, axis=0)

    # Filter tracks by minimum motion and OOB
    filtered_tracks = []
    for i_corner in range(len(tracks)):
        track = tracks[i_corner][:, 1:]
        # Displacement
        start_pt = track[0, :]
        end_pt = track[-1, :]
        displacement = np.linalg.norm(end_pt - start_pt)
        displacements_all.append(displacement)
        if displacement < MIN_TRACK_DISPLACEMENT:
            continue

        # OOB
        x_inbounds = np.logical_and(track[:, 0] > 0, track[:, 0] < IMG_W - 1).all()
        y_inbounds = np.logical_and(track[:, 1] > 0, track[:, 1] < IMG_H - 1).all()
        if not (x_inbounds and y_inbounds):
            continue

        filtered_tracks.append(tracks[i_corner])

    if len(filtered_tracks) == 0:
        return
    print(f"Remaining tracks after filtering: {len(filtered_tracks)}")

    for track_idx in range(len(filtered_tracks)):
        track = filtered_tracks[track_idx]
        track_idx_column = track_idx * np.ones((track.shape[0], 1), dtype=track.dtype)
        filtered_tracks[track_idx] = np.concatenate([track_idx_column, track], axis=1)

    # Sort row entries
    filtered_tracks = np.concatenate(filtered_tracks, axis=0)
    sorted_idxs = np.lexsort((filtered_tracks[:, 0], filtered_tracks[:, 1]))
    filtered_tracks = filtered_tracks[sorted_idxs]

    # Write tracks to disk
    tracks_dir = OUTPUT_DIR / split / seq_dir.stem / "tracks"
    if not tracks_dir.exists():
        tracks_dir.mkdir()
    output_path = tracks_dir / f"{TRACK_NAME}.gt.txt"
    np.savetxt(output_path, filtered_tracks)


def generate_tracks(dataset_dir, output_dir):
    """
    - For both the train and test splits:\n
        - For each sequence:
            - Detect harris corners at t=0.4
            - For each corner, i_corner:
                - Read displacement from flow images (0.41 <= t <= 0.9) to obtain the track
                - Write track to output_dir/<train/test>/tracks/i_corner.txt
    :param dataset_dir: Directory path to multiflow dataset
    :param output_dir: Output directory to obtained tracks
    """
    global OUTPUT_DIR

    # Input and output pathing
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.exists(), "Path to Multiflow dataset does not exist."

    OUTPUT_DIR = Path(output_dir)
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    # Generate tracks
    for split in ["test", "train"]:
        split_dir = dataset_dir / split
        print(f"Generate tracks for {split}")

        global displacements_all
        displacements_all = []

        n_seqs = len(os.listdir(str(split_dir)))
        with multiprocessing.Pool(10) as p:
            list(tqdm(p.imap(generate_single_track, split_dir.iterdir()), total=n_seqs))


if __name__ == "__main__":
    fire.Fire(generate_tracks)
