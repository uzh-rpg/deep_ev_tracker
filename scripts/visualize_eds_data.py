""" Visualize corrected EDS events and images """
import os
from glob import glob

import cv2
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np

sequence_name = "all_characters"
corrected_image_dir = f"<path>/{sequence_name}/images_corrected"
image_ts_path = f"<path>/{sequence_name}/images_timestamps.txt"
corrected_events_path = (
    f"<path>/{sequence_name}/events_corrected.h5"
)
dt_event_slice = 500

# Read times
image_ts = np.genfromtxt(image_ts_path, skip_header=False)

# Load events
with h5py.File(corrected_events_path, "r") as h5f:
    event_times = np.array(h5f["t"])
    print(f"Event time range: {event_times.min()} - {event_times.max()}")

    # plt.ion()
    # Iterate over images
    for image_idx, image_p in enumerate(
        sorted(glob(os.path.join(corrected_image_dir, "*.png")))
    ):
        # Load grayscale frame
        image = cv2.imread(image_p, cv2.IMREAD_COLOR)
        plt.imshow(image)

        # Get relevant events
        t1 = image_ts[image_idx]
        t0 = t1 - dt_event_slice
        first_idx = np.searchsorted(event_times, t0, side="left")
        last_idx_p1 = np.searchsorted(event_times, t1, side="right")
        x = np.asarray(h5f["x"][first_idx:last_idx_p1])
        y = np.asarray(h5f["y"][first_idx:last_idx_p1])
        p = np.asarray(h5f["p"][first_idx:last_idx_p1])
        on_mask = p == 1
        off_mask = p == 0

        # Draw events
        plt.scatter(x[on_mask], y[on_mask], s=1, c="green")
        plt.scatter(x[off_mask], y[off_mask], s=1, c="red")
        plt.title(f"Image Time: {t1*1e-6}")
        plt.axis("off")
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        # plt.show()
