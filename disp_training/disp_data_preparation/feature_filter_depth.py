import cv2
import numpy as np
import scipy.ndimage
import matplotlib

from disp_training.disp_data_preparation.track_storage import TrackStorage


class FeatureFilter:
    def __init__(self, reprojection_thresh):
        self.forward_counter = 0
        self.backward_counter = 0
        self.max_corners = 1000
        self.quality_level = 0.001  # Lower means more points are detected
        self.minimum_distance = 31
        self.block_size = None
        self.k = 0.01
        self.minEigThreshold = 1e-7  # Lower threshold means less points are tracked

        self.track_storage = TrackStorage(reprojection_thresh)
        self.prev_img = None

    def forward_step(self, depth_image, image, T_W_C, K, frame_id):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        # Visual Tracking
        if self.forward_counter == 0:
            # High Blurring since event camera has lower resolution and more noise
            detect_image = cv2.medianBlur(gray_image.copy(), 15)
            new_points = cv2.goodFeaturesToTrack(detect_image,
                                                 self.max_corners, self.quality_level, self.minimum_distance,
                                                 k=self.k,
                                                 useHarrisDetector=True,
                                                 blockSize=self.block_size).squeeze(1)
        else:
            # Tracking
            nextPoints = self.track_storage.tracked_points.astype(np.float32)
            if T_W_C is not None:
                projection_mask, projected_points = self.track_storage.get_projected_points_with_storage_idx(
                    self.track_storage.tracked_point_to_storage_id, T_W_C, K)
                nextPoints[projection_mask, :] = projected_points

            tracked_points, status, err = cv2.calcOpticalFlowPyrLK(prevImg=self.prev_img,
                                                                   nextImg=gray_image,
                                                                   prevPts=self.track_storage.tracked_points.astype(np.float32),
                                                                   nextPts=nextPoints if nextPoints.shape[0] > 0 else None,
                                                                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                                                   minEigThreshold=self.minEigThreshold,
                                                                   winSize=(25, 25), maxLevel=5)

            # Remove points tracked outside image
            if tracked_points is not None:
                _, in_frame_mask = self.get_image_points(tracked_points, gray_image.shape[0], gray_image.shape[1])
                status *= in_frame_mask[:, None].astype(np.uint8)
                self.track_storage.update_tracked_points(tracked_points, status, depth_image, T_W_C, K, frame_id)

            # Detection
            # High Blurring since event camera has lower resolution and more noise
            detect_image = cv2.medianBlur(gray_image.copy(), 15)
            new_points = cv2.goodFeaturesToTrack(detect_image,
                                                 self.max_corners, self.quality_level, self.minimum_distance,
                                                 k=self.k,
                                                 useHarrisDetector=True,
                                                 blockSize=self.block_size)

            if new_points is not None:
                new_points = new_points.squeeze(1)

                if tracked_points is not None:
                    tracked_points, _ = self.get_image_points(tracked_points[status.astype(bool).squeeze(1), :],
                                                              image.shape[0], image.shape[1])

                    track_mask = self.create_nms_mask(tracked_points, image.shape[0], image.shape[1])

                    new_points_int = self.get_image_points(new_points, image.shape[0], image.shape[1])[0].astype(int)
                    untracked_mask = track_mask[new_points_int[:, 1], new_points_int[:, 0]]
                    new_points = new_points[untracked_mask, :]

        if new_points is not None and new_points.shape[0] > 0:
            self.track_storage.add_new_points(new_points, depth_image, T_W_C, K, frame_id)

        self.prev_img = gray_image
        self.forward_counter += 1

        viz_dict = None
        return viz_dict

    def backward_step(self, depth_image, image, T_W_C, K, frame_id):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        if self.backward_counter == 0:
            self.prev_img = gray_image
            self.backward_counter += 1
            self.prev_frame_id = frame_id
            viz_dict = None
            return viz_dict

        # Image points tracked in the forward loop will be overwritten
        prev_points, prev_point_to_storage_id = self.track_storage.get_tracked_points_reverse(self.prev_frame_id)
        nextPoints = prev_points.copy()
        if T_W_C is not None:
            projection_mask, projected_points = self.track_storage.get_projected_points_with_storage_idx(
                prev_point_to_storage_id, T_W_C, K)
            nextPoints[projection_mask, :] = projected_points
        tracked_points, status, err = cv2.calcOpticalFlowPyrLK(prevImg=self.prev_img,
                                                               nextImg=gray_image,
                                                               prevPts=prev_points,
                                                               nextPts=nextPoints,
                                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                                               minEigThreshold=self.minEigThreshold,
                                                               winSize=(25, 25), maxLevel=5)
        _, in_frame_mask = self.get_image_points(tracked_points, gray_image.shape[0], gray_image.shape[1])
        status *= in_frame_mask[:, None].astype(np.uint8)

        self.track_storage.tracked_point_to_storage_id = prev_point_to_storage_id
        self.track_storage.update_tracked_points(tracked_points, status, depth_image, T_W_C, K, frame_id)

        self.prev_img = gray_image
        self.backward_counter += 1
        self.prev_frame_id = frame_id

        # Visualization
        viz_dict = None

        return viz_dict

    def visualization_step(self, depth_image, image, T_W_C, K, frame_id):
        viz_image = image.copy()
        viz_dict = {}

        if T_W_C is not None:
            # Projected Points
            viz_projected_points, proj_points_to_storage_id = self.track_storage.get_projected_points_with_frame_id(frame_id, T_W_C, K)
            if viz_projected_points is not None:
                viz_projected_points, in_frame_mask = self.get_image_points(viz_projected_points, image.shape[0], image.shape[1])
                proj_points_to_storage_id = proj_points_to_storage_id[in_frame_mask]
                viz_dict['projected_features'] = self.draw_corners(viz_projected_points, viz_image,
                                                                   proj_points_to_storage_id)

            # Tracked Points
            viz_tracked_points, tracked_points_to_storage_id = self.track_storage.get_points_with_frame_id(frame_id)
            if viz_projected_points is not None:
                viz_tracked_points, in_frame_mask = self.get_image_points(viz_tracked_points, image.shape[0], image.shape[1])
                tracked_points_to_storage_id = tracked_points_to_storage_id[in_frame_mask]
                viz_dict['tracked_features'] = self.draw_corners(viz_tracked_points, viz_image, tracked_points_to_storage_id)

            if viz_projected_points is not None and viz_tracked_points is not None:
                viz_dict['combined_features'] = self.draw_corners(np.concatenate([viz_projected_points, viz_tracked_points], axis=0),
                                                                  viz_image,
                                                                  np.concatenate([proj_points_to_storage_id,
                                                                                  tracked_points_to_storage_id], axis=0))

        return viz_dict

    def get_image_points(self, points, height, width):
        in_frame_mask = np.logical_and(points[:, 0] < width,
                                       points[:, 0] > 0)
        in_frame_mask = np.logical_and(in_frame_mask, points[:, 1] < height)
        in_frame_mask = np.logical_and(in_frame_mask, points[:, 1] > 0)

        return points[in_frame_mask, :], in_frame_mask

    def create_nms_mask(self, tracked_points, img_h, img_w):
        track_mask = np.zeros([img_h, img_w])
        track_mask[tracked_points[:, 1].astype(int),
        tracked_points[:, 0].astype(int)] = 1
        pooled_track_mask = scipy.ndimage.maximum_filter(track_mask,
                                                         (self.minimum_distance * 2, self.minimum_distance * 2))
        pooled_track_mask[:self.minimum_distance, :] = 1
        pooled_track_mask[-self.minimum_distance:, :] = 1
        pooled_track_mask[:, :self.minimum_distance] = 1
        pooled_track_mask[:, -self.minimum_distance:] = 1
        track_mask = (1 - pooled_track_mask).astype(bool)

        return track_mask

    def draw_corners(self, corners, image, corner_track_id=None):
        if corner_track_id is not None:
            cmap = matplotlib.cm.get_cmap('hsv')
            colors = []
            for track_id in corner_track_id:
                track_id = track_id % 256
                color = cmap(track_id)
                colors.append((color[0] * 255, color[1] * 255, color[2] * 255))
        else:
            colors = [(0, 255, 0) for _ in range(len(corners))]
        if image.shape[2] == 1:
            image = np.tile(image, [1, 1, 3])

        viz_img = image.copy()
        corners = corners.astype('int')
        for i_point, point in enumerate(corners):
            cv2.circle(viz_img, tuple(point), radius=3, color=colors[i_point], thickness=-1)

        return viz_img
