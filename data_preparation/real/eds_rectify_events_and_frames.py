import argparse
import glob
import os
import sys
from os.path import join

import cv2
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from utils.utils import blosc_opts


class Camera:
    def __init__(self, data):
        self.intrinsics = np.eye(3)
        self.intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]] = data["intrinsics"]

        # distortion
        self.distortion_coeffs = np.array(data["distortion_coeffs"])
        self.distortion_model = data["distortion_model"]
        self.resolution = data["resolution"]

        if "T_cn_cnm1" not in data:
            self.R = np.eye(3)
        else:
            self.R = np.array(data["T_cn_cnm1"])[:3, :3]

        self.K = self.intrinsics

    @property
    def num_pixels(self):
        return np.prod(self.resolution)


class CameraSystem:
    def __init__(self, data, fix_rotation=False):
        # load calibration

        self.cam0 = Camera(data["cam0"])
        self.cam1 = Camera(
            data["cam1"]
        )  # if cam0.num_pixels > cam1.num_pixels else (cam1, cam0)

        self.newK = self.cam0.K
        self.newR = self.cam1.R
        self.newRes = self.cam0.resolution

        if not fix_rotation:
            # camera chain parameters
            self.newK = self.event_cam.K

            # tmp = cv2.stereoRectify(self.cam.K, self.cam.distortion_coeffs,
            #                   self.event_cam.K, self.event_cam.distortion_coeffs,
            #                   self.event_cam.resolution, T[:3, :3], T[:3, 3])
            # find new extrinsics
            self.t = T[:3, 3]
            r3_cam0 = self.cam.R[:, 2]

            r1 = self.t / np.linalg.norm(self.t)
            r2 = np.cross(r3_cam0, r1)
            r3 = np.cross(r1, r2)
            self.newR = np.stack([r1, r2, r3], -1)
            print("distance: %s" % (np.linalg.norm(self.t) * self.newK[0, 0]))
        else:
            self.newR = self.cam.R
            self.newK = self.event_cam.K


def vizloop(kwargs, callbacks, image_fun):
    kwargs = {**kwargs, "index": 0}

    while True:
        image = image_fun(kwargs)
        cv2.imshow("Viz", image)

        c = cv2.waitKey(3)
        key = chr(c & 255)

        for k, callback in callbacks.items():
            if key == k:
                ret = callback(kwargs)
                if ret is not None:
                    kwargs.update(ret)

        if c == 27:  # 'q' or 'Esc': Quit
            break

    cv2.destroyAllWindows()

    kwargs.pop("index")
    return kwargs


def _remap_events(events, map, rotate, shape):
    mx, my = map
    x, y = mx[events["y"], events["x"]], my[events["y"], events["x"]]
    p = np.array(events["p"])
    t = np.array(events["t"])

    target_width, target_height = shape

    if rotate:
        x = target_width - 1 - x
        y = target_height - 1 - y

    mask = (x >= 0) & (x <= target_width - 1) & (y >= 0) & (y <= target_height - 1)

    x = x[mask]
    y = y[mask]
    t = t[mask]
    p = p[mask]

    return {"x": x, "y": y, "t": t, "p": p}


def process_events(file, output, maps, shape, rotate=False):
    events = h5py.File(file)
    events = _remap_events(events, maps, rotate, shape)
    with h5py.File(output, "w") as h5f_out:
        h5f_out.create_dataset(
            "x", data=events["x"], **blosc_opts(complevel=1, shuffle="byte")
        )
        h5f_out.create_dataset(
            "y", data=events["y"], **blosc_opts(complevel=1, shuffle="byte")
        )
        h5f_out.create_dataset(
            "p", data=events["p"], **blosc_opts(complevel=1, shuffle="byte")
        )
        h5f_out.create_dataset(
            "t", data=events["t"], **blosc_opts(complevel=1, shuffle="byte")
        )


def _remap_img(img, map, flip, rotate):
    if flip:
        img = img[:, ::-1]
    mx, my = map
    img_remapped = cv2.remap(img, mx, my, cv2.INTER_CUBIC)
    if rotate:
        img_remapped = cv2.rotate(img_remapped, cv2.ROTATE_180)
    return img_remapped


def process_img(img_file, output_folder, distortion_maps, flip, rotate):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    img_remapped = _remap_img(img, distortion_maps, flip, rotate)
    output_path = os.path.join(output_folder, os.path.basename(img_file))
    cv2.imwrite(output_path, img_remapped)


def getRemapping(camsys: CameraSystem):
    # undistort image
    img_mapx, img_mapy = cv2.initUndistortRectifyMap(
        camsys.cam0.K,
        camsys.cam0.distortion_coeffs,
        camsys.newR @ camsys.cam0.R.T,
        camsys.newK,
        camsys.newRes,
        cv2.CV_32FC1,
    )

    ev_mapx, ev_mapy = cv2.initUndistortRectifyMap(
        camsys.cam1.K,
        camsys.cam1.distortion_coeffs,
        camsys.newR @ camsys.cam1.R.T,
        camsys.newK,
        camsys.newRes,
        cv2.CV_32FC1,
    )

    W, H = camsys.cam1.resolution
    coords = (
        np.stack(np.meshgrid(np.arange(W), np.arange(H)))
        .reshape((2, -1))
        .T.reshape((-1, 1, 2))
        .astype("float32")
    )
    points = cv2.undistortPoints(
        coords,
        camsys.cam1.K,
        camsys.cam1.distortion_coeffs,
        None,
        camsys.newR @ camsys.cam1.R.T,
        camsys.newK,
    )
    inv_maps = points.reshape((H, W, 2))

    return {
        "img_mapx": img_mapx,
        "img_mapy": img_mapy,
        "ev_mapx": ev_mapx,
        "ev_mapy": ev_mapy,
        "inv_mapx": inv_maps[..., 0],
        "inv_mapy": inv_maps[..., 1],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Remap images to be aligned with frames""")
    parser.add_argument("sequence_name")
    parser.add_argument("--data_dir")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--map_key", default="")
    parser.add_argument("--debug", action="store_true", default="false")
    parser.add_argument(
        "-n", "--num_processes", help="Number of workers", type=int, default=8
    )

    args = parser.parse_args()

    map_key = args.map_key

    # search for image folder
    image_dir = os.path.join(args.data_dir, args.sequence_name, "images")
    image_paths = sorted(glob.glob(join(image_dir, "*.png")))
    output_image_dir = image_dir + "_corrected"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # search for calibration file
    calibration_path = os.path.join(args.data_dir, "calib.yaml")
    with open(calibration_path, "r") as fh:
        cam_data = yaml.load(fh, Loader=yaml.SafeLoader)

    fix_rotation = False
    if args.rotate:
        fix_rotation = True

    camsys = CameraSystem(cam_data, fix_rotation)
    maps = getRemapping(camsys)

    for f in tqdm(image_paths, desc="Processing images..."):
        process_img(
            f,
            output_image_dir,
            (maps["img_mapx"], maps["img_mapy"]),
            args.flip,
            args.rotate,
        )

    events_path = os.path.join(args.data_dir, args.sequence_name, "events.h5")
    output_events_path = events_path.replace(".h5", "_corrected.h5")
    print("Processing events...")
    process_events(
        events_path,
        output_events_path,
        (maps["inv_mapx"], maps["inv_mapy"]),
        tuple(camsys.cam1.resolution),
        args.rotate,
    )
