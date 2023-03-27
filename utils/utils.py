import os
from multiprocessing import Pool

import h5py
import hdf5plugin
import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm


def blosc_opts(complevel=1, complib="blosc:zstd", shuffle="byte"):
    # Inspired by: https://github.com/h5py/h5py/issues/611#issuecomment-353694301
    # More info on options: https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L55-L79
    shuffle = 2 if shuffle == "bit" else 1 if shuffle == "byte" else 0
    compressors = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    complib = ["blosc:" + c for c in compressors].index(complib)
    args = {
        "compression": 32001,
        "compression_opts": (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        # Do not use h5py shuffle if blosc shuffle is enabled.
        args["shuffle"] = False
    return args


def query_events(events_h5, events_t, t0, t1):
    """
    Return a numpy array of events in temporal range [t0, t1)
    :param events_h5: h5 object with events. {x, y, p, t} as keys.
    :param events_t: np array of the uncompressed event times
    :param t0: start time of slice in us
    :param t1: terminal time of slice in us
    :return: (-1, 4) np array
    """
    first_idx = np.searchsorted(events_t, t0, side="left")
    last_idx_p1 = np.searchsorted(events_t, t1, side="right")
    x = np.asarray(events_h5["x"][first_idx:last_idx_p1])
    y = np.asarray(events_h5["y"][first_idx:last_idx_p1])
    p = np.asarray(events_h5["p"][first_idx:last_idx_p1])
    t = np.asarray(events_h5["t"][first_idx:last_idx_p1])
    return {"x": x, "y": y, "p": p, "t": t, "n_events": len(x)}


def events2time_surface(events_h5, events_t, t0, t1, resolution):
    """
    Build a timesurface from events in temporal range [t0, t1)
    :param events_h5: h5 object with events. {x, y, p, t} as keys.
    :param events_t: np array of the uncompressed event times
    :param t0: start time of slice in us
    :param t1: terminal time of slice in us
    :param resolution: 2-element tuple (W, H)
    :return: (H, W) np array
    """
    time_surface = np.zeros((resolution[1], resolution[0]), dtype=np.float64)
    events_dict = query_events(events_h5, events_t, t0, t1)

    for i in range(events_dict["n_events"]):
        x = int(np.rint(events_dict["x"][i]))
        y = int(np.rint(events_dict["y"][i]))

        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
            time_surface[y, x] = (events_dict["t"][i] - t0) / (t1 - t0)

    return time_surface


def read_input(input_path, representation):
    input_path = str(input_path)

    assert os.path.exists(input_path), f"Path to input file {input_path} doesn't exist."

    if "time_surface" in representation:
        return h5py.File(input_path, "r")["time_surface"]

    elif "voxel" in representation:
        return h5py.File(input_path, "r")["voxel_grid"]

    elif "event_stack" in representation:
        return h5py.File(input_path, "r")["event_stack"]

    elif "grayscale" in representation:
        return imread(input_path, IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    else:
        print("Unsupported representation")
        exit()


def propagate_keys(cfg, testing=False):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.data.representation = cfg.representation
        cfg.data.track_name = cfg.track_name
        cfg.data.patch_size = cfg.patch_size

        cfg.model.representation = cfg.representation
        cfg.data.patch_size = cfg.patch_size

        if not testing:
            cfg.model.n_vis = cfg.n_vis
            cfg.model.init_unrolls = cfg.init_unrolls
            cfg.model.max_unrolls = cfg.max_unrolls
            cfg.model.debug = cfg.debug

        cfg.model.pose_mode = cfg.data.name == "pose"


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
