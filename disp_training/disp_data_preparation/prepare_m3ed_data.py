import os
import cv2
cv2.setNumThreads(8)
import h5py
import numpy as np
import tqdm
import yaml

from disp_training.disp_data_preparation.utils import project_points, reproject_points, compute_rectify_map, remap_events
from disp_training.disp_data_preparation.feature_filter_depth import FeatureFilter

DATA_DIR = <path>
VIZ_DIR = <path>
SAVE_DIR = <path>
FILTER_TRACK_LENGTH = 20  # Frames to keep tracks
REPROJECTION_ERROR_THRESHOLD = 3
PATCH_MARGIN = 32
MAX_DISPARITY = 60
MIN_STD_MOVEMENT = 10


def read_depth_pose(sequence_dir, time_ts):
    # Depth GT file
    left_event_depth = [file_name for file_name in os.listdir(sequence_dir) if 'depth_gt.h5' in file_name][0]
    f = h5py.File(os.path.join(sequence_dir, left_event_depth), 'r')

    time_differences = (np.abs(f['ts'][:] - time_ts))
    closest_idx = time_differences.argmin()
    if time_differences[closest_idx] > 10000:
        return None, None, None

    T_evleft_W = f['Cn_T_C0'][closest_idx]
    T_W_evleft = np.linalg.inv(T_evleft_W)

    depth_image_evleft = f['depth/prophesee/left'][closest_idx]
    left_event_idx = f['ts_map_prophesee_left'][closest_idx]
    f.close()

    # Data File
    data_path = [file_name for file_name in os.listdir(sequence_dir) if 'data.h5' in file_name][0]
    f = h5py.File(os.path.join(sequence_dir, data_path), 'r')

    intr_evleft = f['prophesee/left/calib/intrinsics'][:]
    K_evleft = np.identity(3)
    K_evleft[0, 0], K_evleft[1, 1], K_evleft[0, 2], K_evleft[1, 2] = intr_evleft

    # Reproject depth to pointcloud
    depth_mask = depth_image_evleft != np.inf
    p_depth_evleft = depth_image_evleft[depth_mask]
    vu_depth_evleft = np.stack(depth_mask.nonzero(), axis=1)
    P_evleft, valid_mask = reproject_points(vu_depth_evleft[:, 1], vu_depth_evleft[:, 0], depth_image=None,
                                            T_C_W=np.identity(4),
                                            K=K_evleft,
                                            z=p_depth_evleft)
    p_depth_evleft = p_depth_evleft[valid_mask]

    # Project pointcloud to left ovc image | Left image will be loaded and directly undistorted
    T_evleft_ovcleft = f['ovc/left/calib/T_to_prophesee_left'][:, :]
    T_W_ovcleft = np.matmul(T_W_evleft, T_evleft_ovcleft)
    intr_ovcleft = f['ovc/left/calib/intrinsics'][:]
    K_ovcleft = np.identity(3)
    K_ovcleft[0, 0], K_ovcleft[1, 1], K_ovcleft[0, 2], K_ovcleft[1, 2] = intr_ovcleft
    uv_depth_ovcleft, valid_mask = project_points(P_evleft.squeeze(2),
                                                  np.linalg.inv(T_evleft_ovcleft),
                                                  K_ovcleft)
    p_depth_ovc_left = p_depth_evleft[valid_mask]

    img_w, img_h = f['ovc/left/calib/resolution'][:]
    depth_image_ovcleft = np.zeros([img_h, img_w])

    u = uv_depth_ovcleft[:, 0].round().astype('int')
    v = uv_depth_ovcleft[:, 1].round().astype('int')

    # Inside frame check
    mask = (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1)
    u, v = u[mask], v[mask]
    depth_image_ovcleft[v, u] = p_depth_ovc_left[mask]

    f.close()

    return T_W_ovcleft, depth_image_ovcleft, left_event_idx


def read_image(sequence_dir, time_ts, return_img_idx=False):
    # Data File
    data_path = [file_name for file_name in os.listdir(sequence_dir) if 'data.h5' in file_name][0]
    f = h5py.File(os.path.join(sequence_dir, data_path), 'r')

    closest_idx = (np.abs(f['ovc/ts'][:] - time_ts)).argmin()

    distorted_img = f['ovc/left/data'][closest_idx]

    intr_ovcleft = f['ovc/left/calib/intrinsics'][:]
    K_ovcleft = np.identity(3)
    K_ovcleft[0, 0], K_ovcleft[1, 1], K_ovcleft[0, 2], K_ovcleft[1, 2] = intr_ovcleft
    img_ovcleft = cv2.undistort(distorted_img,
                                K_ovcleft,
                                f['ovc/left/calib/distortion_coeffs'][:], None,
                                K_ovcleft)
    img_ovcleft = img_ovcleft[:, :, None]

    f.close()

    if return_img_idx:
        return img_ovcleft, closest_idx
    return img_ovcleft


def read_event_image(sequence_dir, img_idx, nr_events, rectification_data=None):
    events, rect_h, rect_w = read_events_batch(sequence_dir, img_idx, nr_events, rectification_data)

    # Visualization
    viz_img = np.zeros([rect_h, rect_w, 3])
    viz_img[events['y'].astype("int"), events['x'].astype("int"), :] = 255

    return viz_img


def read_events_batch(sequence_dir, img_idx, nr_events, rectification_data=None, limit_between_img=False):
    data_path = [file_name for file_name in os.listdir(sequence_dir) if 'data.h5' in file_name][0]
    f = h5py.File(os.path.join(sequence_dir, data_path), 'r')
    stop = f['/ovc/ts_map_prophesee_left_t'][img_idx]

    if img_idx == 0:
        return None

    if limit_between_img:
        if img_idx - 1 < 0:
            nr_events = min(nr_events, f['/ovc/ts_map_prophesee_left_t'][img_idx])
        else:
            nr_events = min(nr_events, stop - f['/ovc/ts_map_prophesee_left_t'][img_idx -1])
    start = int(stop - nr_events)
    events = {'x': f['/prophesee/left/x'][start:stop],
              'y': f['/prophesee/left/y'][start:stop],
              't': f['/prophesee/left/t'][start:stop],
              'p': f['/prophesee/left/p'][start:stop]}
    f.close()

    if rectification_data is not None:
        inv_map_events = rectification_data['inv_map_evleft']
        rect_h, rect_w = rectification_data['map_evleft'].shape[:2]
        event_h, event_w = inv_map_events.shape[:2]

        events = remap_events(
            events,
            (inv_map_events[:, :, 0], inv_map_events[:, :, 1]),
            rotate=False,
            shape=(event_w, event_h),
        )

        return events, rect_h, rect_w

    return events


def read_timstamps_K(sequence_dir):
    data_path = [file_name for file_name in os.listdir(sequence_dir) if 'data.h5' in file_name][0]
    f = h5py.File(os.path.join(sequence_dir, data_path), 'r')
    timestamps = f['ovc/ts'][:]
    intr_ovcleft = f['ovc/left/calib/intrinsics'][:]
    K_ovcleft = np.identity(3)
    K_ovcleft[0, 0], K_ovcleft[1, 1], K_ovcleft[0, 2], K_ovcleft[1, 2] = intr_ovcleft
    f.close()

    return K_ovcleft, timestamps,


def get_rectification_data(sequence_dir, timestamps):
    data_path = [file_name for file_name in os.listdir(sequence_dir) if 'data.h5' in file_name][0]
    f = h5py.File(os.path.join(sequence_dir, data_path), 'r')

    # Compute rectification map
    out = compute_rectify_map(target_group=f['ovc/left/calib/'], source_group=f['prophesee/left/calib'])
    rectification_data = {'map_ovcleft': out[0],
                          'map_evleft': out[1],
                          'inv_map_ovcleft': out[2],
                          'inv_map_evleft': out[3],
                          'proj_ovcleft': out[4],
                          'proj_evleft': out[5],
                          'rot_ovcleft': out[6],
                          'rot_evleft': out[7],
                          'Q': out[8]}
    f.close()

    return rectification_data


def rectify_filter_features(feature_filter, rectification_data, ovc_poses):
    rectified_track_storage = []
    rect_h, rect_w = rectification_data['map_ovcleft'].shape[:2]
    for track in feature_filter.track_storage.track_storage:
        rectified_track = {'P_W': track['P_W'], 'track': {}}
        track_dict = {}
        for i_img, (image_timestep, point) in enumerate(track['track'].items()):
            # Mapping
            int_point = point.astype('int')
            mx, my = rectification_data['inv_map_ovcleft'][:, :, 0], rectification_data['inv_map_ovcleft'][:, :, 1]
            x, y = mx[int_point[1], int_point[0]], my[int_point[1], int_point[0]]
            rectified_point = np.array([x, y, 1])
            if (rectified_point[0] < PATCH_MARGIN or rectified_point[0] >= rect_w - PATCH_MARGIN or
                    rectified_point[1] < PATCH_MARGIN or rectified_point[1] >= rect_h - PATCH_MARGIN):
                if len(track_dict) > len(rectified_track['track']):
                    rectified_track['track'] = track_dict
                track_dict = {}
                continue

            track_dict[image_timestep] = {'uv': rectified_point[:2]}

            if str(image_timestep) not in ovc_poses:
                continue

            # Disparity
            P_unrect_to_rect = np.identity(4)
            P_unrect_to_rect[:3, :3] = rectification_data['rot_ovcleft']
            point_ovcleft = np.matmul(np.matmul(P_unrect_to_rect, np.linalg.inv(ovc_poses[str(image_timestep)])),
                                      track['P_W'][0])
            Z = point_ovcleft[2]
            f = rectification_data['proj_evleft'][0, 0]
            b = - 1 / rectification_data['Q'][3, 2]
            disparity = b * f / Z
            track_dict[image_timestep]['d'] = disparity

            if disparity > MAX_DISPARITY:
                del track_dict[image_timestep]
                if len(track_dict) > len(rectified_track['track']):
                    rectified_track['track'] = track_dict
                track_dict = {}

        if len(track_dict) > len(rectified_track['track']):
            rectified_track['track'] = track_dict

        if len(rectified_track['track']) == 0:
            continue
        uv = np.stack([point['uv'] for point in rectified_track['track'].values()])
        if len(rectified_track['track']) >= FILTER_TRACK_LENGTH and np.std(uv, axis=0).mean() >= MIN_STD_MOVEMENT:
            rectified_track_storage.append(rectified_track)

    return rectified_track_storage


def interpolate_disparities(rectified_tracks):
    for track in rectified_tracks:
        disparities = []
        mask_with_d = np.zeros([len(track['track'])], dtype='bool')
        for i, (image_timestep, uv_d_dict) in enumerate(track['track'].items()):
            if 'd' in uv_d_dict:
                disparities.append(uv_d_dict['d'])
                mask_with_d[i] = True

        disparities = np.array(disparities).squeeze(1)
        x = np.arange(len(track['track']))
        xp = x[mask_with_d]
        inter_disp = np.interp(x, xp, disparities)

        for i, (image_timestep, uv_d_dict) in enumerate(track['track'].items()):
            uv_d_dict['d'] = inter_disp[i]

    return  rectified_tracks


def save_rectified_data(sequence, rectification_data, forward_timestamps):
    sequence_dir = os.path.join(DATA_DIR, sequence)
    save_dir = os.path.join(SAVE_DIR, sequence)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    sequence_h5_name = os.path.join(save_dir, 'rectified_data.h5')
    if os.path.exists(sequence_h5_name):
        os.remove(sequence_h5_name)
    f = h5py.File(sequence_h5_name, "a")

    event_chunk = None
    ovc_chunk = None
    chunk_size = 60
    n_bins = 5
    n_evs_representation = 500000
    last_i = 0
    for i, timestamp in tqdm.tqdm(enumerate(forward_timestamps),
                                           total=len(forward_timestamps),
                                           desc="Saving Loop"):
        img_ovcleft, img_idx = read_image(sequence_dir, timestamp, return_img_idx=True)
        img_ovcleft_rect = cv2.remap(img_ovcleft,
                                     rectification_data['map_ovcleft'][:, :, 0],
                                     rectification_data['map_ovcleft'][:, :, 1],
                                     cv2.INTER_LINEAR)

        # Event Image
        ev_left_rect, rect_h, rect_w = read_events_batch(sequence_dir, img_idx,
                                                         nr_events=n_evs_representation,
                                                         rectification_data=rectification_data,
                                                         limit_between_img=True)
        event_representation = create_sbt_max_representation(ev_left_rect, n_bins, img_h=rect_h, img_w=rect_w)
        last_i = i

        if event_chunk is None:
            event_chunk = np.zeros([chunk_size, rect_h, rect_w, 2 * n_bins], dtype=np.float32)
            ovc_chunk = np.zeros([chunk_size, rect_h, rect_w], dtype=np.uint8)
            event_chunk[i % chunk_size, :, :, :] = event_representation
            ovc_chunk[i % chunk_size, :, :] = img_ovcleft_rect
        elif i % chunk_size == (chunk_size - 1):
            event_chunk[i % chunk_size, :, :, :] = event_representation
            ovc_chunk[i % chunk_size, :, :] = img_ovcleft_rect

            # Add to file
            if i == (chunk_size - 1):
                f.create_dataset('ovcleft', data=ovc_chunk, chunks=(chunk_size, rect_h, rect_w),
                                 maxshape=(None, rect_h, rect_w), dtype=np.uint8, compression="gzip")
                f.create_dataset('evleft_sbtmax', data=event_chunk, chunks=(chunk_size, rect_h, rect_w, 2 * n_bins),
                                 maxshape=(None, rect_h, rect_w, 2 * n_bins), dtype=np.float32, compression="gzip")
                f.create_dataset('evleft_sbtmax_max_events', data=n_evs_representation)

            else:
                nr_samples = f['ovcleft'].shape[0]
                f['ovcleft'].resize(nr_samples + chunk_size, axis=0)
                f['evleft_sbtmax'].resize(nr_samples + chunk_size, axis=0)
                f['ovcleft'][-chunk_size:, :, :] = ovc_chunk
                f['evleft_sbtmax'][-chunk_size:, :, :, :] = event_chunk

            event_chunk = np.zeros([chunk_size, rect_h, rect_w, 2 * n_bins], dtype=np.float32)
            ovc_chunk = np.zeros([chunk_size, rect_h, rect_w], dtype=np.uint8)

        else:
            event_chunk[i % chunk_size, :, :, :] = event_representation
            ovc_chunk[i % chunk_size, :, :] = img_ovcleft_rect
            

    if last_i % chunk_size != (chunk_size - 1):
        n_remaining = last_i % chunk_size + 1
        nr_samples = f['ovcleft'].shape[0]
        f['ovcleft'].resize(nr_samples + n_remaining, axis=0)
        f['evleft_sbtmax'].resize(nr_samples + n_remaining, axis=0)
        f['ovcleft'][-n_remaining:, :, :] = ovc_chunk[:n_remaining, :, :]
        f['evleft_sbtmax'][-n_remaining:, :, :, :] = event_chunk[:n_remaining, :, :, :]


    assert f['ovcleft'].shape[0] == len(forward_timestamps)
    f.create_dataset('timestamps', data=np.array(forward_timestamps))

    f.close()


def save_tracks(sequence, tracks, forward_timestamps):
    save_dir = os.path.join(SAVE_DIR, sequence)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    sequence_h5_name = os.path.join(save_dir, 'rectified_tracks.h5')
    if os.path.exists(sequence_h5_name):
        os.remove(sequence_h5_name)
    f = h5py.File(sequence_h5_name, "a")

    timestamp_to_tracks = {}
    for timestamp in forward_timestamps:
        timestamp_to_tracks[timestamp] = {'track_id': [],
                                          'track_length': [],
                                          'uv': [],
                                          'd': []}

    for i_track, track in enumerate(tracks):
        old_ts = -1
        uvd_array = np.zeros([len(track['track']), 3])
        for i_step, (image_timestep, uv_d_dict) in enumerate(track['track'].items()):
            uvd_array[i_step, :2] = uv_d_dict['uv']
            uvd_array[i_step, 2] = uv_d_dict['d']

            assert old_ts < image_timestep
            old_ts = image_timestep

            if image_timestep not in timestamp_to_tracks:
                continue

            timestamp_to_tracks[image_timestep]['track_id'].append(i_track)
            timestamp_to_tracks[image_timestep]['track_length'].append(len(track['track']) - i_step)
            timestamp_to_tracks[image_timestep]['uv'].append(uv_d_dict['uv'])
            timestamp_to_tracks[image_timestep]['d'].append(uv_d_dict['d'])

    for timestamp, img_tracks in timestamp_to_tracks.items():
        f.create_dataset('{}/track_id'.format(str(timestamp)), data=np.array(timestamp_to_tracks[timestamp]['track_id']))
        f.create_dataset('{}/track_length'.format(str(timestamp)), data=np.array(timestamp_to_tracks[timestamp]['track_length']))
        f.create_dataset('{}/uv'.format(str(timestamp)), data=np.array(timestamp_to_tracks[timestamp]['uv']))
        f.create_dataset('{}/d'.format(str(timestamp)), data=np.array(timestamp_to_tracks[timestamp]['d']))

    f.close()


def save_rectification_calibration(sequence, rectification_data):
    save_dir = os.path.join(SAVE_DIR, sequence)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(SAVE_DIR, sequence, "rectification_calibration.yaml")
    if os.path.exists(save_path):
        os.remove(save_path)
    save_dict = {'proj_ovcleft': rectification_data['proj_ovcleft'].flatten().tolist(),
                 'proj_evleft': rectification_data['proj_evleft'].flatten().tolist(),
                 'Q': rectification_data['Q'].flatten().tolist()}

    with open(save_path, 'w') as file:
        yaml.dump(save_dict, file, default_flow_style=None)

    rectification_maps_path = os.path.join(SAVE_DIR, sequence, "rectification_maps.h5")
    if os.path.exists(rectification_maps_path):
        os.remove(rectification_maps_path)
    f = h5py.File(rectification_maps_path, "a")
    f.create_dataset('map_ovcleft', data=rectification_data['map_ovcleft'])
    f.create_dataset('map_evleft', data=rectification_data['map_evleft'])
    f.create_dataset('inv_map_ovcleft', data=rectification_data['inv_map_ovcleft'])
    f.create_dataset('inv_map_evleft', data=rectification_data['inv_map_evleft'])
    f.close()


def create_sbt_max_representation(events, n_bins, img_h, img_w):
    time_surface = np.zeros((img_h, img_w, 2 * n_bins), dtype=np.uint64)

    repr_t = events['t']
    repr_x = events['x'].astype(np.uint64)
    repr_y = events['y'].astype(np.uint64)
    t0, t1 = repr_t.min(), repr_t.max()
    dt_us = t1 - t0
    dt_bin_us = dt_us / n_bins

    for i_bin in range(n_bins):
        t0_bin = t0 + i_bin * dt_bin_us
        t1_bin = t0_bin + dt_bin_us

        first_idx = np.searchsorted(repr_t, t0_bin, side='left')
        last_idx_p1 = np.searchsorted(repr_t, t1_bin, side='right')
        events_bin = {
            'x': repr_x[first_idx:last_idx_p1],
            'y': repr_y[first_idx:last_idx_p1],
            'p': events['p'][first_idx:last_idx_p1],
            't': repr_t[first_idx:last_idx_p1],
        }

        n_events = events_bin['x'].shape[0]
        for i in range(n_events):
            time_surface[events_bin['y'][i], events_bin['x'][i], 2 * i_bin + int(events_bin['p'][i])] = events_bin['t'][i] - t0

    time_surface = np.divide(time_surface, dt_us)

    return time_surface


def filter_tracks(feature_filter, ovc_poses, K_ovcleft, save_stats=False):
    if save_stats:
        remove_tracks_stats = {'no_depth': 0,
                               'too_short': 0,}
    else:
        remove_tracks_stats = None
    tracks_to_keep = []
    for track_idx, track_dict in tqdm.tqdm(enumerate(feature_filter.track_storage.track_storage),
                                           total=len(feature_filter.track_storage.track_storage),
                                           desc="Filter Loop"):
        if len(track_dict['P_W']) == 0:
            if save_stats:
                remove_tracks_stats['no_depth'] += 1
            continue
        if len(track_dict['track'].keys()) < FILTER_TRACK_LENGTH:
            if save_stats:
                remove_tracks_stats['too_short'] += 1
            continue

        image_ids = list(track_dict['track'].keys())
        image_ids.sort()
        valid_image_point = np.ones([len(image_ids)])
        P_W_track = np.stack(track_dict['P_W'], axis=0)
        P_W_mean = feature_filter.track_storage.get_aggregated_P_W(P_W_track)
        for image_id in image_ids:
            if str(image_id) not in ovc_poses:
                continue
            T_W_ovcleft = ovc_poses[str(image_id)]
            proj_point, in_front_mask = project_points(P_W_mean[None, :, 0], np.linalg.inv(T_W_ovcleft), K_ovcleft)
            if not in_front_mask[0]:
                valid_image_point[image_ids.index(image_id)] = 0
                continue
            reproj_err = np.linalg.norm(proj_point[0, :2] - track_dict['track'][image_id])
            if reproj_err >= REPROJECTION_ERROR_THRESHOLD:
                valid_image_point[image_ids.index(image_id)] = 0

        if (valid_image_point == 0).sum() == 0:
            tracks_to_keep.append(track_idx)
            continue

        # Find subtrack with the highest number of valid images. This assumes that feature tracks are ordered in time
        failed_reproj = valid_image_point == 0
        failed_reproj[-1] = True
        borders = np.concatenate([[-1], failed_reproj.nonzero()[0]])
        borders[-1] += 1 if valid_image_point[-1] != 0 else 0
        nr_images_successful = np.cumsum(valid_image_point)[failed_reproj]
        nr_images_successful[1:] = nr_images_successful[1:] - nr_images_successful[:-1]
        max_images_idx = np.argmax(nr_images_successful)
        assert max_images_idx < len(borders) - 1

        if nr_images_successful[max_images_idx] < FILTER_TRACK_LENGTH:
            if save_stats:
                remove_tracks_stats['too_short'] += 1
            continue

        subtrack_ids_to_keep = image_ids[borders[max_images_idx] + 1:borders[max_images_idx + 1]]
        new_track_dict = {k: v for k, v in track_dict['track'].items() if k in subtrack_ids_to_keep}
        tracks_to_keep.append(track_idx)
        track_dict['track'] = new_track_dict
        track_dict['P_W'] = [P_W_mean]

        feature_filter.track_storage.track_storage[track_idx] = track_dict

    return tracks_to_keep, remove_tracks_stats


def process_sequence(sequence):
    sequence_dir = os.path.join(DATA_DIR, sequence)

    K_ovcleft, timestamps = read_timstamps_K(sequence_dir)
    rectification_data = get_rectification_data(sequence_dir, timestamps)
    save_rectification_calibration(sequence, rectification_data)
    feature_filter = FeatureFilter(REPROJECTION_ERROR_THRESHOLD)

    ovc_poses = {}
    forward_timestamps = []
    #  ==========  Forward Loop ==========
    for i_time, time_ts in tqdm.tqdm(enumerate(timestamps), total=len(timestamps), desc="Forward Loop"):
        T_W_ovcleft, depth_image_ovcleft, left_event_idx = read_depth_pose(sequence_dir, time_ts)
        img_ovcleft = read_image(sequence_dir, time_ts)

        forward_timestamps.append(time_ts)
        if T_W_ovcleft is not None:
            ovc_poses[str(time_ts)] = T_W_ovcleft

    # ========== Backward Loop ==========
    for i_time in tqdm.tqdm(range(len(forward_timestamps)), total=len(forward_timestamps), desc="Backward Loop"):
        time_ts = forward_timestamps[-(1 + i_time)]
        T_W_ovcleft, depth_image_ovcleft, left_event_idx = read_depth_pose(sequence_dir, time_ts)
        img_ovcleft = read_image(sequence_dir, time_ts)

        viz_dict = feature_filter.backward_step(depth_image_ovcleft, img_ovcleft, T_W_ovcleft, K_ovcleft,
                                                frame_id=time_ts)

    # ========== Filter Loop ==========
    # Sort image timestamps
    for track in feature_filter.track_storage.track_storage:
        track['track'] = dict(sorted(track['track'].items()))
    nr_tracks_before = len(feature_filter.track_storage.track_storage)
    tracks_to_keep, remove_tracks_stats = filter_tracks(feature_filter, ovc_poses, K_ovcleft, save_stats=True)
    feature_filter.track_storage.track_storage = [feature_filter.track_storage.track_storage[i] for i in tracks_to_keep]

    # ========== Rectification ==========
    rectified_tracks = rectify_filter_features(feature_filter, rectification_data, ovc_poses)
    rectified_tracks = interpolate_disparities(rectified_tracks)

    for track in rectified_tracks:
        assert len(track['track'].keys()) >= FILTER_TRACK_LENGTH

    # =============== Saving =================
    forward_timestamps = forward_timestamps[1:]  # Remove first timestamp since no events are available
    save_rectified_data(sequence, rectification_data, forward_timestamps)
    save_tracks(sequence, rectified_tracks, forward_timestamps)


def main():
    sequences = os.listdir(DATA_DIR)

    sequences_wo_gt = [
        'car_urban_day_schuylkill_tunnel',
        'car_urban_day_ucity_big_loop',
        'car_urban_night_schuylkill_tunnel',
        'car_urban_night_ucity_big_loop',
        'car_forest_sand_2',
        'falcon_outdoor_day_fast_flight_3',
        'falcon_outdoor_day_penno_parking_3',
        'falcon_forest_road_3',
        'spot_outdoor_day_penno_building_loop',
        'spot_outdoor_day_skatepark_3',
        'spot_outdoor_night_penno_building_loop',
        'spot_forest_road_2',
        'falcon_into_forest_3',
    ]

    for sequence in sequences:
        if sequence in sequences_wo_gt:
            continue
        process_sequence(sequence)


if __name__ == '__main__':
    main()
