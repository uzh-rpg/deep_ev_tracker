"""
Prepare data for a subset of an Event Camera Dataset sequence
- Undistort images and events
- Create time surfaces
- Create an output directory with undistorted images, undistorted event txt, and time surfaces
"""
from pathlib import Path
import os
from glob import glob
from fire import Fire
from tqdm import tqdm

from pandas import read_csv
import numpy as np
import cv2


def prepare_data(root_dir, sequence_name):
    sequence_dir = Path(root_dir) / sequence_name
    if not sequence_dir.exists():
        print(f"Sequence directory does not exist for {sequence_name}")
        exit()

    # Read calib
    calib_data = np.genfromtxt(str(sequence_dir / 'calib.txt'))
    camera_matrix = calib_data[:4]
    distortion_coeffs = calib_data[4:]
    camera_matrix = np.array([[camera_matrix[0], 0, camera_matrix[2]],
                              [0, camera_matrix[1], camera_matrix[3]],
                              [0, 0, 1]])
    print("Calibration loaded")

    # Undistort images
    images_dir = sequence_dir / 'images_corrected'
    images_dir.mkdir()
    for img_idx, img_path in enumerate(tqdm(sorted(glob(str(sequence_dir / 'images' / '*.png'))),
                                            desc="Undistorting images...")):
        img = cv2.imread(img_path)
        img = cv2.undistort(img, cameraMatrix=camera_matrix, distCoeffs=distortion_coeffs)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filename = f'frame_{str(img_idx).zfill(8)}.png'
        cv2.imwrite(os.path.join(str(images_dir / filename)), img)
    img_tmp = cv2.imread(str(images_dir / 'frame_00000000.png'))
    H_img, W_img = img_tmp.shape[:2]

    # Remove first entry in image timestamps if not already
    image_timestamps = np.genfromtxt(str(sequence_dir / 'images.txt'), usecols=[0])
    if len(image_timestamps) == len(glob(str(sequence_dir / 'images' / '*.png'))):
        np.savetxt(str(sequence_dir / 'images.txt'), image_timestamps)

    # Undistort events
    events_corrected_path = sequence_dir / 'events_corrected.txt'
    if not events_corrected_path.exists():
        events = read_csv(str(sequence_dir / 'events.txt'), header=None, delimiter=' ').to_numpy()
        print("Raw events loaded")

        events[:, 1:3] = cv2.undistortPoints(events[:, 1:3].reshape((-1, 1, 2)),
                                             camera_matrix, distortion_coeffs, P=camera_matrix).reshape((-1, 2),)
        events[:, 1:3] = np.rint(events[:, 1:3])

        inbounds_mask = np.logical_and(events[:, 1] >= 0, events[:, 1] < W_img)
        inbounds_mask = np.logical_and(inbounds_mask, events[:, 2] >= 0)
        inbounds_mask = np.logical_and(inbounds_mask, events[:, 2] < H_img)
        events = events[inbounds_mask, :]

        print("Events undistorted")
        np.savetxt(events_corrected_path, events, ["%.9f", "%i", "%i", "%i"])


if __name__ == '__main__':
    Fire(prepare_data)
