import numpy as np

from disp_training.disp_data_preparation.utils import project_points, reproject_points


class TrackStorage:
    def __init__(self, reprojection_thresh):
        self.track_storage = []
        self.track_counter = 0
        self.tracked_points = np.zeros([0, 2], dtype=np.float32)
        self.tracked_point_to_storage_id = np.zeros([0], dtype='int')
        self.reprojection_thresh = reprojection_thresh

    def add_new_points(self, new_points, depth_image, T_W_C, K, frame_id):
        if T_W_C is not None:
            P_W_array, points_with_depth_mask = reproject_points(new_points[:, 0], new_points[:, 1],
                                                                 depth_image, np.linalg.inv(T_W_C), K)
        else:
            points_with_depth_mask = np.zeros([new_points.shape[0]], dtype='bool')

        new_points_to_track_id = []
        P_W_counter = 0
        for i_point in range(new_points.shape[0]):
            if points_with_depth_mask[i_point]:
                P_W_list = [P_W_array[P_W_counter]]
                P_W_counter += 1
            else:
                P_W_list = []
            track_dict = {
                'P_W': P_W_list,
                'track': {int(frame_id): new_points[i_point, :]}
            }
            self.track_storage.append(track_dict)
            new_points_to_track_id.append(self.track_counter)
            self.track_counter += 1

        self.tracked_points = np.concatenate([self.tracked_points, new_points], axis=0)
        self.tracked_point_to_storage_id = np.concatenate([self.tracked_point_to_storage_id,
                                                           np.array(new_points_to_track_id)], axis=0)

    def update_tracked_points(self, tracked_points, status, depth_image, T_W_C, K, frame_id):
        success_tracking = status.squeeze(1).astype('bool')
        self.tracked_points = tracked_points[success_tracking, :]
        self.tracked_point_to_storage_id = self.tracked_point_to_storage_id[success_tracking]

        # Triangulate 3D points for tracked points with depth information and make reprojection error check
        if T_W_C is not None:
            # Reprojection error check
            valid_tracks = np.ones([self.tracked_points.shape[0]], dtype='bool')
            valid_P_W = np.zeros([self.tracked_points.shape[0]], dtype='bool')
            P_W_tracked_points = np.zeros([self.tracked_points.shape[0], 4, 1])
            for i_point in range(self.tracked_points.shape[0]):
                storage_id = self.tracked_point_to_storage_id[i_point]
                # Get 3D point corresponding to track
                if len(self.track_storage[storage_id]['P_W']) == 0:
                    continue

                P_W_track = np.stack(self.track_storage[storage_id]['P_W'], axis=0)
                P_W_mean = self.get_aggregated_P_W(P_W_track)
                P_W_tracked_points[i_point] = P_W_mean
                valid_P_W[i_point] = True

            if valid_P_W.sum() != 0:
                projected_P_W, in_front_camera_mask = project_points(P_W_tracked_points[valid_P_W, :, 0],
                                                                     np.linalg.inv(T_W_C), K)
                if in_front_camera_mask.sum() != 0:
                    project_point_idx = valid_P_W.nonzero()[0][in_front_camera_mask]
                    reprojection_error = np.sqrt(((projected_P_W[:, :2] - self.tracked_points[project_point_idx, :])**2).sum(1))
                    valid_tracks[project_point_idx] *= reprojection_error < self.reprojection_thresh

            # Add 3D point if depth information is available
            P_W_array, points_with_depth_mask = reproject_points(self.tracked_points[:, 0], self.tracked_points[:, 1],
                                                                 depth_image, np.linalg.inv(T_W_C), K)
            points_with_depth_idx = points_with_depth_mask.nonzero()[0]
            for i_point in range(self.tracked_points.shape[0]):
                if points_with_depth_mask[i_point] and valid_tracks[i_point]:
                    storage_id = self.tracked_point_to_storage_id[i_point]
                    self.track_storage[storage_id]['P_W'].append(P_W_array[points_with_depth_idx == i_point, :, :].squeeze(0))

            self.tracked_points = self.tracked_points[valid_tracks, :]
            self.tracked_point_to_storage_id = self.tracked_point_to_storage_id[valid_tracks]

        # Add 2D point
        for i_point in range(self.tracked_points.shape[0]):
            storage_id = self.tracked_point_to_storage_id[i_point]
            self.track_storage[storage_id]['track'][int(frame_id)] = self.tracked_points[i_point]

    def get_tracked_points_reverse(self, prev_frame_id):
        prev_points = []
        prev_point_to_storage_id = []
        for i_track, track_dict in enumerate(self.track_storage):
            if prev_frame_id in track_dict['track']:
                prev_points.append(track_dict['track'][prev_frame_id])
                prev_point_to_storage_id.append(i_track)

        return np.stack(prev_points), np.array(prev_point_to_storage_id)

    def get_aggregated_P_W(self, P_W_track):
        nr_P_W = P_W_track.shape[0]
        P_W_distances = ((P_W_track[:, None, :3, 0] - P_W_track[None, :, :3, 0]) ** 2).sum(2)
        P_W_distances = P_W_distances.mean(axis=1)
        nr_cluster_samples = max((nr_P_W // 3) * 2, 1)
        cluster_P_W_idx = P_W_distances.argsort()[:nr_cluster_samples]
        P_W_mean = np.mean(P_W_track[cluster_P_W_idx], axis=0)

        return P_W_mean

    def get_points_with_frame_id(self, frame_id):
        points = []
        points_to_storage_id = []
        for i_track, track_dict in enumerate(self.track_storage):
            if len(track_dict['P_W']) == 0:
                continue
            if frame_id in track_dict['track']:
                points.append(track_dict['track'][frame_id])
                points_to_storage_id.append(i_track)

        if len(points) == 0:
            return None, None

        return np.stack(points), np.array(points_to_storage_id)

    def get_projected_points_with_frame_id(self, frame_id, T_W_C, K):
        points_to_storage_id = []
        P_W = []
        for i_track, track_dict in enumerate(self.track_storage):
            if len(track_dict['P_W']) == 0:
                continue
            if frame_id in track_dict['track']:
                P_W_track = np.stack(track_dict['P_W'], axis=0)
                P_W.append(self.get_aggregated_P_W(P_W_track))
                points_to_storage_id.append(i_track)

        if len(P_W) == 0:
            return None, None

        P_W = np.stack(P_W, axis=0)
        projected_points, in_front_camera_mask = project_points(P_W[:, :, 0], np.linalg.inv(T_W_C), K)

        return projected_points[:, :2], np.array(points_to_storage_id)[in_front_camera_mask]

    def get_projected_points_with_storage_idx(self, storage_idxs, T_W_C, K):
        P_W = []
        projection_mask = np.zeros([storage_idxs.shape[0]], dtype='bool')
        for i_idx, storage_idx in enumerate(storage_idxs):
            track_dict = self.track_storage[storage_idx]
            if len(track_dict['P_W']) == 0:
                continue
            P_W_track = np.stack(track_dict['P_W'], axis=0)
            P_W.append(self.get_aggregated_P_W(P_W_track))
            projection_mask[i_idx] = True

        if len(P_W) == 0:
            return projection_mask, None

        P_W = np.stack(P_W, axis=0)
        projected_points, in_front_camera_mask = project_points(P_W[:, :, 0], np.linalg.inv(T_W_C), K)
        projection_mask[projection_mask] = in_front_camera_mask

        return projection_mask, projected_points[:, :2]
