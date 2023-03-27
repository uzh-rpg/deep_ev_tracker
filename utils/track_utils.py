"""
Utility functions for managing track data
"""
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


class TrackInterpolator:
    def __init__(self, track_data, terminate_out_of_frame=False, img_size=None):
        self.n_corners = len(np.unique(track_data[:, 0]))
        self.track_interpolators = {}
        self.track_data = {}

        if terminate_out_of_frame:
            track_data = self.terminate_track(track_data, img_size)

        for track_idx in range(self.n_corners):
            track_data_curr = track_data[track_data[:, 0] == track_idx, 1:]
            if track_data_curr.shape[0] > 1:
                t, t_idx = np.unique(track_data_curr[:, 0], return_index=True)
                x = track_data_curr[t_idx, 1]
                y = track_data_curr[t_idx, 2]
                self.track_interpolators[track_idx] = {
                    "x": interp1d(t, x, kind="linear"),
                    "y": interp1d(t, y, kind="linear"),
                    "t_range": [
                        np.min(track_data_curr[:, 0]),
                        np.max(track_data_curr[:, 0]),
                    ],
                }
            else:
                self.track_interpolators[track_idx] = None
            self.track_data[track_idx] = track_data_curr

    def interpolate(self, track_idx, t_query):
        track_interpolator = self.track_interpolators[track_idx]

        if isinstance(track_interpolator, type(None)):
            return None
        elif (
            track_interpolator["t_range"][0]
            <= t_query
            <= track_interpolator["t_range"][1]
        ):
            return np.array(
                [track_interpolator["x"](t_query), track_interpolator["y"](t_query)]
            )
        else:
            return None

    def interpolate_list(self, track_idx, t_query_list):
        track_interpolator = self.track_interpolators[track_idx]

        if (
            track_interpolator["t_range"][0]
            <= np.min(t_query_list)
            <= track_interpolator["t_range"][1]
            and track_interpolator["t_range"][0]
            <= np.max(t_query_list)
            <= track_interpolator["t_range"][1]
        ):
            x_interp = track_interpolator["x"](t_query_list).reshape((-1, 1))
            y_interp = track_interpolator["y"](t_query_list).reshape((-1, 1))
            return np.concatenate([x_interp, y_interp], axis=1)
        else:
            print(
                f"Time range for interpolator is [{track_interpolator['t_range'][0]}, {track_interpolator['t_range'][1]}]"
                f"but queried time range is [{np.min(t_query_list)}, {np.max(t_query_list)}]"
            )
            return None

    def history(self, track_idx, t_query, dt_history):
        track_interpolator = self.track_interpolators[track_idx]
        track_data = self.track_data[track_idx]

        if (
            track_interpolator["t_range"][0]
            <= t_query
            <= track_interpolator["t_range"][1]
        ):
            time_mask = np.logical_and(
                track_data[:, 0] >= t_query - dt_history, track_data[:, 0] <= t_query
            )
            # time_mask = track_data[:, 0] <= t_query
            return track_data[time_mask, 1:]
        else:
            return None

    def terminate_track(self, track_data, img_size):
        new_track_data = []

        for track_idx in np.unique(track_data[:, 0]):
            track_data_curr = track_data[np.isclose(track_data[:, 0], track_idx), :]

            mask_oobx = np.logical_or(
                track_data_curr[:, 2] < 0, track_data_curr[:, 2] > img_size[0] - 1
            )
            mask_ooby = np.logical_or(
                track_data_curr[:, 3] < 0, track_data_curr[:, 3] > img_size[1] - 1
            )
            mask_oob = np.logical_or(mask_oobx, mask_ooby)

            if mask_oob.any():
                idx_oob = int(np.min(np.argwhere(mask_oob)))
                # pdb.set_trace()
                track_data_curr = track_data_curr[:idx_oob, :]

            new_track_data.append(track_data_curr)

        return np.concatenate(new_track_data, axis=0)


class TrackObserver:
    def __init__(self, t_init, u_centers_init):
        self.n_corners = u_centers_init.shape[0]
        idx_col = np.array(range(self.n_corners)).reshape((-1, 1))

        if isinstance(t_init, np.ndarray):
            time_col = t_init
        else:
            time_col = np.ones(idx_col.shape) * t_init
        self.track_data = np.concatenate([idx_col, time_col, u_centers_init], axis=1)

    def add_observation(self, t, u_centers, mask=None):
        idx_col = np.array(range(u_centers.shape[0])).reshape((-1, 1))
        time_col = np.ones(idx_col.shape) * t
        new_track_data = np.concatenate([idx_col, time_col, u_centers], axis=1)

        if not isinstance(mask, type(None)):
            new_track_data = new_track_data[mask, :]

        self.track_data = np.concatenate([self.track_data, new_track_data], axis=0)

    def get_interpolators(self):
        return TrackInterpolator(self.track_data)

    def terminate_oob(self, img_size, padding):
        """
        :param img_size: (H, W)
        :param padding: Padding that must be exceeded for termination to occur
        :return: None, modified internal track data
        """
        new_track_data = []

        for track_idx in np.unique(self.track_data[:, 0]):
            track_data_curr = self.track_data[
                np.isclose(self.track_data[:, 0], track_idx), :
            ]

            mask_oobx = np.logical_or(
                track_data_curr[:, 2] < -padding,
                track_data_curr[:, 2] > img_size[1] - 1 + padding,
            )
            mask_ooby = np.logical_or(
                track_data_curr[:, 3] < -padding,
                track_data_curr[:, 3] > img_size[0] - 1 + padding,
            )
            mask_oob = np.logical_or(mask_oobx, mask_ooby)

            if mask_oob.any():
                idx_oob = int(np.min(np.argwhere(mask_oob)))
                # pdb.set_trace()
                track_data_curr = track_data_curr[:idx_oob, :]

            new_track_data.append(track_data_curr)

        self.track_data = np.concatenate(new_track_data, axis=0)


class TrackTriangulator:
    def __init__(
        self, track_data, pose_interpolator, t_init, camera_matrix, depths=None
    ):
        self.camera_matrix = camera_matrix
        self.camera_matrix_inv = np.linalg.inv(camera_matrix)
        self.T_init_W = pose_interpolator.interpolate(t_init)
        self.T_W_init = np.linalg.inv(self.T_init_W)

        if isinstance(depths, type(None)):
            # Triangulate the points
            self.eigenvalues = []
            self.corners_3D_homo = []
            for idx_track in np.unique(track_data[:, 0]):
                track_data_curr = track_data[track_data[:, 0] == idx_track, 1:]
                n_obs = track_data_curr.shape[0]
                if n_obs < 10:
                    print(f"Warning: not very many observations for triangulation")

                # Construct A
                A = []
                for idx_obs in range(n_obs):
                    corner = track_data_curr[idx_obs, 1:]

                    t = track_data_curr[idx_obs, 0]
                    T_j_W = pose_interpolator.interpolate(t)
                    T_j_init = T_j_W @ np.linalg.inv(self.T_init_W)

                    P = self.camera_matrix @ T_j_init[:3, :]
                    A.append(corner[0] * P[2, :] - P[0, :])
                    A.append(corner[1] * P[2, :] - P[1, :])

                A = np.array(A)
                _, s, vh = np.linalg.svd(A)
                X = vh[-1, :].reshape((-1))
                X /= X[-1]
                self.corners_3D_homo.append(X.reshape((1, 4)))
                self.eigenvalues.append(s[-1])
            self.corners_3D_homo = np.concatenate(self.corners_3D_homo, axis=0)
        else:
            # Back-project using the depths
            self.corners_3D_homo = []
            for idx_track in np.unique(track_data[:, 0]):
                corner_coords = track_data[track_data[:, 0] == idx_track, 1:].reshape(
                    (-1,)
                )
                corner_depth = float(depths[depths[:, 0] == idx_track, 1])
                assert (
                    len(corner_coords) == 2
                ), "Backprojection using depths only supports corner set as input"
                xy_homo = np.array([corner_coords[0], corner_coords[1], 1]).reshape(
                    (3, 1)
                )
                ray_backproj = self.camera_matrix_inv @ xy_homo
                xyz = ray_backproj * corner_depth
                X = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1]).reshape(
                    (1, 4)
                )
                # X = self.T_W_init @ X.T
                self.corners_3D_homo.append(X.reshape((1, 4)))
            self.corners_3D_homo = np.concatenate(self.corners_3D_homo, axis=0)
        self.n_corners = self.corners_3D_homo.shape[0]

    def get_corners(self, T_j_init):
        """
        Determine the 2D position of the features from the initial extraction step
        :param T_j_init
        :return:
        """
        corners_3D = (T_j_init @ self.corners_3D_homo.T).T
        corners_3D = corners_3D[:, :3]
        corners_2D_proj = (self.camera_matrix @ corners_3D.T).T
        corners_2D_proj = corners_2D_proj / corners_2D_proj[:, 2].reshape((-1, 1))
        corners_2D_proj = corners_2D_proj[:, :2]
        return corners_2D_proj

    def get_depths(self, T_j_init):
        """
        Determine the 2D position of the features from the initial extraction step
        :param T_j_init
        :return:
        """
        corners_3D = (T_j_init @ self.corners_3D_homo.T).T
        corners_3D = corners_3D[:, :3]
        return corners_3D[:, 2]


class PoseInterpolator:
    def __init__(self, pose_data, mode="linear"):
        """
        :param pose_data: Nx7 numpy array with [t, x, y, z, qx, qy, qz, qw] as the row format
        """
        self.pose_data = pose_data
        self.x_interp = interp1d(
            pose_data[:, 0], pose_data[:, 1], kind=mode, bounds_error=True
        )
        self.y_interp = interp1d(
            pose_data[:, 0], pose_data[:, 2], kind=mode, bounds_error=True
        )
        self.z_interp = interp1d(
            pose_data[:, 0], pose_data[:, 3], kind=mode, bounds_error=True
        )
        self.rot_interp = Slerp(pose_data[:, 0], Rotation.from_quat(pose_data[:, 4:]))
        #
        # self.qx_interp = interp1d(pose_data[:, 0], pose_data[:, 4], kind='linear')
        # self.qy_interp = interp1d(pose_data[:, 0], pose_data[:, 5], kind='linear')
        # self.qz_interp = interp1d(pose_data[:, 0], pose_data[:, 6], kind='linear')
        # self.qw_interp = interp1d(pose_data[:, 0], pose_data[:, 7], kind='linear')

    def interpolate(self, t):
        """
        Interpolate 6-dof pose from the initial pose data
        :param t: Query time at which to interpolate the pose
        :return: 4x4 Transformation matrix T_j_W
        """
        if t < np.min(self.pose_data[:, 0]) or t > np.max(self.pose_data[:, 0]):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.pose_data[:, 0])}, {np.max(self.pose_data[:, 0])}]"
            )
        T_W_j = np.eye(4)
        T_W_j[0, 3] = self.x_interp(t)
        T_W_j[1, 3] = self.y_interp(t)
        T_W_j[2, 3] = self.z_interp(t)
        T_W_j[:3, :3] = self.rot_interp(t).as_matrix()
        return np.linalg.inv(T_W_j)

    def interpolate_colmap(self, t):
        if t < np.min(self.pose_data[:, 0]) or t > np.max(self.pose_data[:, 0]):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.pose_data[:, 0])}, {np.max(self.pose_data[:, 0])}]"
            )
        T_W_j = np.eye(4)
        T_W_j[0, 3] = self.x_interp(t)
        T_W_j[1, 3] = self.y_interp(t)
        T_W_j[2, 3] = self.z_interp(t)
        T_W_j[:3, :3] = self.rot_interp(t).as_matrix()
        T_j_W = np.linalg.inv(T_W_j)
        quat = Rotation.from_matrix(T_j_W[:3, :3]).as_quat()
        return np.asarray(
            [T_j_W[0, 3], T_j_W[1, 3], T_j_W[2, 3], quat[0], quat[1], quat[2], quat[3]],
            dtype=np.float32,
        )


def retrieve_track_tuples(extra_dir, track_name):
    track_tuples = []
    for extra_seq_dir in tqdm(extra_dir.iterdir(), desc="Fetching track paths..."):
        # Ignore hidden dirs
        if str(extra_seq_dir.stem).startswith("."):
            continue

        # Check if has valid tracks
        track_path = os.path.join(str(extra_seq_dir), "tracks", f"{track_name}.gt.txt")
        if not os.path.exists(track_path):
            continue

        # Store paths
        track_data = np.genfromtxt(track_path)
        n_tracks = len(np.unique(track_data[:, 0]))
        for track_idx in range(n_tracks):
            track_tuples.append((track_path, track_idx))

    return track_tuples


def compute_tracking_errors2(
    pred_track_data,
    gt_track_data,
    klt_track_data,
    pose_interpolator=None,
    camera_matrix=None,
    error_threshold=5,
    n_tracks_dead=0,
):
    """
    Used for computing errors for synchronous methods
    :param track_data: array of predicted tracks
    :param klt_track_data: array of gt tracks
    :param reproj_track_data: array of reproj track
    :param error_threshold: threshold for a live track (5 px used in HASTE paper)
    :return: None, prints the mean relative feature age and mean track-normed error
    """

    fa_rel_arr, te_arr = [0 for _ in range(n_tracks_dead)], []
    klt_consistency, pred_consistency = [], []

    for track_idx in np.unique(pred_track_data[:, 0]):
        klt_track_data_curr = klt_track_data[klt_track_data[:, 0] == track_idx, 1:]
        gt_track_data_curr = gt_track_data[gt_track_data[:, 0] == track_idx, 1:]
        pred_track_data_curr = pred_track_data[pred_track_data[:, 0] == track_idx, 1:]

        # Crop tracks to the same length
        if gt_track_data_curr[-1, 0] > pred_track_data_curr[-1, 0]:
            gt_time_mask = np.logical_and(
                gt_track_data_curr[:, 0] >= np.min(pred_track_data_curr[:, 0]),
                gt_track_data_curr[:, 0] <= np.max(pred_track_data_curr[:, 0]),
            )
            gt_track_data_curr_cropped = gt_track_data_curr[gt_time_mask, :]
            # klt_track_data_curr = klt_track_data_curr[gt_time_mask, :]
        else:
            gt_track_data_curr_cropped = gt_track_data_curr

        # If KLT could not track the feature, skip it
        if gt_track_data_curr_cropped.shape[0] == 1:
            continue

        # Else, compare against the KLT track
        else:
            # Create predicted track interpolators
            x_interp = interp1d(pred_track_data_curr[:, 0], pred_track_data_curr[:, 1])
            y_interp = interp1d(pred_track_data_curr[:, 0], pred_track_data_curr[:, 2])

            # Interpolate predicted track at GT timestamps
            pred_x = x_interp(gt_track_data_curr_cropped[:, 0]).reshape((-1, 1))
            pred_y = y_interp(gt_track_data_curr_cropped[:, 0]).reshape((-1, 1))
            pred_track_data_curr_interp = np.concatenate(
                [gt_track_data_curr_cropped[:, 0:1], pred_x, pred_y], axis=1
            )

            # Compute errors
            tracking_error = (
                gt_track_data_curr_cropped[:, 1:] - pred_track_data_curr_interp[:, 1:]
            )
            tracking_error = tracking_error[1:, :]  # discard initial location
            tracking_error = np.linalg.norm(tracking_error, axis=1).reshape((-1,))

            # Compute relative feature age (idx_end is inclusive)
            if (tracking_error > error_threshold).any():
                idx_end = int(np.min(np.argwhere(tracking_error > error_threshold)))
                if idx_end > 1:
                    idx_end -= 1
                else:
                    idx_end = 0
            else:
                idx_end = -1

            if idx_end == 0:
                fa_rel_arr.append(0)
            else:
                # Tracking GT Error
                t_end_pred = gt_track_data_curr_cropped[idx_end, 0]
                fa = t_end_pred - gt_track_data_curr_cropped[0, 0]
                dt_track = gt_track_data_curr[-1, 0] - gt_track_data_curr[0, 0]
                # Ignore tracks that are short-lived as in HASTE and E-KLT?
                fa_rel = fa / dt_track

                if idx_end != -1:
                    te = np.mean(tracking_error[1 : idx_end + 1])
                else:
                    te = np.mean(tracking_error[1:])

                fa_rel_arr.append(fa_rel)
                te_arr.append(te)

                # Crop tracks and compute consistency
                if idx_end != -1:
                    pred_track_data_curr_interp = pred_track_data_curr_interp[
                        :idx_end, :
                    ]
                    # klt_track_data_curr = klt_track_data_curr[:idx_end, :]

                # Tracking Self-Consistency Error
                # ... for predicted
                if pred_track_data_curr_interp.shape[0] > 1:
                    pred_track_data_reproj = reproject_track(
                        pred_track_data_curr_interp, pose_interpolator, camera_matrix
                    )
                    pred_err_consistency = np.mean(
                        np.linalg.norm(
                            pred_track_data_reproj[:, 1:]
                            - pred_track_data_curr_interp[:, 1:],
                            axis=1,
                        )
                    )
                    pred_consistency.append(pred_err_consistency)

                # ... for klt
                if klt_track_data_curr.shape[0] > 1:
                    klt_track_data_reproj = reproject_track(
                        klt_track_data_curr, pose_interpolator, camera_matrix
                    )
                    klt_err_consistency = np.mean(
                        np.linalg.norm(
                            klt_track_data_reproj[:, 1:] - klt_track_data_curr[:, 1:],
                            axis=1,
                        )
                    )
                    klt_consistency.append(klt_err_consistency)

    return (
        np.array(fa_rel_arr).reshape((-1,)),
        np.array(te_arr).reshape((-1,)),
        np.array(klt_consistency).reshape((-1,)),
        np.array(pred_consistency).reshape((-1,)),
    )


def reproject_track(track_data, pose_interpolator, camera_matrix):
    """
    Reproject a feature track given camera pose and camera matrix (assumes undistorted coords)
    :param track_data: (N_t, 3) array of a feature track (time, x, y)
    :param pose_interpolator: PoseInterpolator object
    :param camera_matrix: (3, 3)
    :return: track_data (N_t, 3) of the re-projected track
    """
    track_data_with_id = np.concatenate(
        [np.zeros((track_data.shape[0], 1)), track_data], axis=1
    )
    track_triangulator = TrackTriangulator(
        track_data_with_id,
        pose_interpolator,
        np.min(track_data_with_id[:, 1]),
        camera_matrix,
    )

    track_data_reproj = []
    for t in track_data[:, 0]:
        # Re-projection
        T_j_W = pose_interpolator.interpolate(t)
        T_j_init = T_j_W @ np.linalg.inv(track_triangulator.T_init_W)
        feature_reproj = track_triangulator.get_corners(T_j_init).reshape((-1,))
        track_data_reproj.append([t, feature_reproj[0], feature_reproj[1]])
    return np.array(track_data_reproj)


def compute_tracking_errors(
    pred_track_data, gt_track_data, asynchronous=True, error_threshold=5
):
    """
    Compute errors for async methods
    :param track_data: array of predicted tracks
    :param klt_track_data: array of gt tracks
    :param error_threshold: threshold for a live track (5 px used in HASTE paper)
    :return: None, prints the mean relative feature age and mean track-normed error
    """

    fa_rel_arr, te_arr = [], []

    for track_idx in np.unique(pred_track_data[:, 0]):
        gt_track_data_curr = gt_track_data[gt_track_data[:, 0] == track_idx, 1:]
        pred_track_data_curr = pred_track_data[pred_track_data[:, 0] == track_idx, 1:]

        if asynchronous:
            # Extend predicted tracks for asynchronous methods (no prediction -> no motion)
            if gt_track_data_curr[-1, 0] > pred_track_data_curr[-1, 0]:
                pred_track_data_curr = np.concatenate(
                    [
                        pred_track_data_curr,
                        np.array(
                            [
                                gt_track_data_curr[-1, 0],
                                pred_track_data_curr[-1, 1],
                                pred_track_data_curr[-1, 2],
                            ]
                        ).reshape((1, 3)),
                    ],
                    axis=0,
                )
        else:
            # Crop gt track for synchronous method (assumes synchronous method is approx. the same length)
            if gt_track_data_curr[-1, 0] > pred_track_data_curr[-1, 0]:
                gt_time_mask = np.logical_and(
                    gt_track_data_curr[:, 0] >= np.min(pred_track_data_curr[:, 0]),
                    gt_track_data_curr[:, 0] <= np.max(pred_track_data_curr[:, 0]),
                )
                gt_track_data_curr = gt_track_data_curr[gt_time_mask, :]

        gt_track_data_curr_cropped = gt_track_data_curr
        # If KLT could not track the feature, skip it
        if gt_track_data_curr_cropped.shape[0] < 2:
            continue

        # Else, compare against the KLT track
        else:
            # Create predicted track interpolators
            x_interp = interp1d(
                pred_track_data_curr[:, 0],
                pred_track_data_curr[:, 1],
                fill_value="extrapolate",
            )
            y_interp = interp1d(
                pred_track_data_curr[:, 0],
                pred_track_data_curr[:, 2],
                fill_value="extrapolate",
            )

            # Interpolate predicted track at GT timestamps
            pred_x = x_interp(gt_track_data_curr_cropped[:, 0]).reshape((-1, 1))
            pred_y = y_interp(gt_track_data_curr_cropped[:, 0]).reshape((-1, 1))
            pred_track_data_curr_interp = np.concatenate([pred_x, pred_y], axis=1)

            # Compute errors
            tracking_error = (
                gt_track_data_curr_cropped[:, 1:] - pred_track_data_curr_interp
            )
            tracking_error = tracking_error[
                1:, :
            ]  # discard initial location which has no error
            tracking_error = np.linalg.norm(tracking_error, axis=1).reshape((-1,))

            # Compute relative feature age (idx_end is inclusive)
            if (tracking_error > error_threshold).any():
                idx_end = int(np.min(np.argwhere(tracking_error > error_threshold)))
                if idx_end > 1:
                    idx_end = idx_end - 1
                else:
                    idx_end = 0
            else:
                idx_end = -1

            if idx_end == 0:
                fa_rel_arr.append(0)
            else:
                t_end_pred = gt_track_data_curr_cropped[idx_end, 0]
                fa = t_end_pred - gt_track_data_curr_cropped[0, 0]
                dt_track = gt_track_data_curr[-1, 0] - gt_track_data_curr[0, 0]
                # Ignore tracks that are short-lived as in HASTE and E-KLT?
                fa_rel = fa / dt_track

                if idx_end != -1:
                    te = np.mean(tracking_error[1 : idx_end + 1])
                else:
                    te = np.mean(tracking_error[1:])

                fa_rel_arr.append(fa_rel)
                te_arr.append(te)

    return np.array(fa_rel_arr).reshape((-1,)), np.array(te_arr).reshape((-1,))


def read_txt_results(results_txt_path):
    """
    Parse an output txt file from E-KLT or Ours of data rows formatted [id, t, x, y]
    :param results_txt_path:
    :return: TrackInterpolator
    """
    return np.genfromtxt(results_txt_path)


def get_gt_corners(results_txt_path):
    """
    Get initial corners from EKLT results
    :param results_txt_path:
    :return:
    """
    track_data = np.genfromtxt(results_txt_path)
    t_start = np.min(track_data[:, 1])
    return track_data[track_data[:, 1] == t_start, 2:]
