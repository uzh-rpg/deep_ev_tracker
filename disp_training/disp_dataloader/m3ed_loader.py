from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from utils.utils import *
from utils.augmentations import *
from disp_training.disp_utils.disp_utils_torch import get_patches, get_event_patches

torch.multiprocessing.set_sharing_strategy('file_system')

# Data Classes for Training
@dataclass
class TrackDataConfig:
    patch_size: int
    augment: bool
    disp_patch_range: int


def concat_collate(batch_sample):
    if len(batch_sample[0]) == 4:
        frame_patches, event_patches, y_disps, frame_idx = [], [], [], []
        for batch_data in batch_sample:
            frame_patches.append(batch_data[0])
            event_patches.append(batch_data[1])
            y_disps.append(batch_data[2])
            frame_idx.append(batch_data[3])

        return (np.concatenate(frame_patches, axis=0), np.concatenate(event_patches, axis=0),
                np.concatenate(y_disps, axis=0), np.concatenate(frame_idx, axis=0))

    elif len(batch_sample[0]) == 6:
        frame_patches, event_patches, y_disps, frame_idx, seq_names, img_points = [], [], [], [], [], []
        for batch_data in batch_sample:
            frame_patches.append(batch_data[0])
            event_patches.append(batch_data[1])
            y_disps.append(batch_data[2])
            frame_idx.append(batch_data[3])
            seq_names.append(batch_data[4])
            img_points.append(batch_data[5])

        return (np.concatenate(frame_patches, axis=0), np.concatenate(event_patches, axis=0),
                np.concatenate(y_disps, axis=0), np.concatenate(frame_idx, axis=0), seq_names,
                np.concatenate(img_points, axis=0))

    else:
        raise ValueError("Invalid batch sample length.")

train_sequences = [
    'falcon_indoor_flight_2',
    'spot_outdoor_day_srt_under_bridge_2',
    'falcon_forest_road_1',
    'falcon_indoor_flight_1',
    'falcon_outdoor_day_penno_parking_2',
    'falcon_forest_into_forest_1',
    'car_urban_day_ucity_small_loop',
    'car_urban_night_city_hall',
    'falcon_outdoor_day_penno_parking_1',
    'car_urban_night_ucity_small_loop',
    'falcon_forest_road_2',
    'falcon_outdoor_night_penno_parking_1',
    'car_forest_into_ponds_short',
    'car_urban_night_penno_small_loop_darker',
    'falcon_outdoor_night_penno_parking_2',
    'car_urban_day_city_hall',
    'car_urban_day_penno_small_loop',
    'falcon_forest_into_forest_4',
    'spot_indoor_stairwell',
    'spot_outdoor_day_srt_under_bridge_1',
    'spot_outdoor_day_skatepark_2',
    'falcon_forest_up_down',
    'spot_outdoor_day_art_plaza_loop',
    'car_urban_day_penno_big_loop',
    'spot_forest_easy_1',
    'spot_outdoor_day_skatepark_1',
    'car_urban_night_rittenhouse',
    'spot_forest_road_1',
    'car_urban_day_rittenhouse',
    'car_urban_night_penno_small_loop',
    'spot_forest_hard',
    'car_urban_night_penno_big_loop',
    'spot_outdoor_day_rocky_steps',
]

val_sequences = [
    'falcon_outdoor_day_fast_flight_1',
    'falcon_indoor_flight_3',
    'spot_indoor_stairs',
    'falcon_forest_into_forest_2',
    'spot_indoor_building_loop',
    'spot_outdoor_night_penno_short_loop',
    'spot_outdoor_day_penno_short_loop',
    'car_forest_into_ponds_long',
    'spot_forest_road_3',
    'spot_forest_easy_2',
]

test_sequences = [
    'car_forest_sand_1',
    'falcon_forest_road_forest',
    'car_forest_tree_tunnel',
    'falcon_outdoor_day_penno_cars',
    'falcon_outdoor_day_fast_flight_2',
    'spot_outdoor_day_srt_green_loop',
    'falcon_outdoor_night_high_beams',
    'falcon_outdoor_day_penno_trees',
    'spot_indoor_obstacles',
    'spot_outdoor_night_penno_plaza_lights',
    'car_urban_day_horse',
    'falcon_outdoor_day_penno_plaza',
]


class TrackData:
    """
    Dataloader for a single feature track. Returns input patches and displacement labels relative to
    the current feature location. Current feature location is either updated manually via accumulate_y_hat()
    or automatically via the ground-truth displacement.
    """
    def __init__(self, track_tuple, config, track_length, additional_outputs=False, augment=False):
        """
        Dataset for a single feature track
        :param track_tuple: (Path to track.gt.txt, track_id)
        :param config:
        """
        self.config = config

        # Get input paths
        self.data_path = os.path.join(track_tuple[0], 'rectified_data.h5')
        self.track_path = os.path.join(track_tuple[0], 'rectified_tracks.h5')
        self.start_idx = track_tuple[1]
        self.track_ids = track_tuple[2]
        self.track_length = track_length
        self.additional_outputs = additional_outputs
        self.augment = augment

        # Representation-specific Settings
        self.channels_in_per_patch = 10

    def reset(self):
        pass

    def add_xy_grid(self, images):
        x_grid, y_grid = np.meshgrid(np.arange(images.shape[2]), np.arange(images.shape[1]))
        x_grid = x_grid.astype(np.float32) / images.shape[2] - 0.5
        y_grid = y_grid.astype(np.float32) / images.shape[1] - 0.5
        xy_grid = np.tile(np.stack([x_grid, y_grid], axis=2), [self.track_length, 1, 1, 1])
        images = np.concatenate([images[:, :, :, None], xy_grid], axis=3)
        return images

    def get_data(self, idx_track):
        with h5py.File(self.data_path, 'r') as h5_sensor_data:
            event_reprs = h5_sensor_data['evleft_sbtmax'][self.start_idx:self.start_idx + self.track_length, :, :, :]
            images = h5_sensor_data['ovcleft'][self.start_idx:self.start_idx + self.track_length, :, :]

        images = self.add_xy_grid(images)

        h5_tracks = h5py.File(self.track_path, 'r')
        timestamp_list = [int(timstamp) for timstamp in list(h5_tracks.keys())]
        timestamp_list.sort()
        timestamps = timestamp_list[self.start_idx:self.start_idx + self.track_length]

        n_tracks = self.track_ids.shape[0]
        y_disp = np.zeros([n_tracks, self.track_length], dtype=np.float32)

        frame_patches = np.zeros([n_tracks, self.track_length, self.config.patch_size, self.config.patch_size, 3], dtype=np.float32)
        event_patches = np.zeros([n_tracks, self.track_length, self.config.disp_patch_range, self.config.patch_size, 10], dtype=np.float32)
        if self.additional_outputs:
            frame_centers = np.zeros([n_tracks, self.track_length, 2], dtype=np.float32)

        for i_t, timestamp in enumerate(timestamps):
            timestamp = str(timestamp)
            track_idx = np.nonzero(h5_tracks[timestamp]['track_id'][:][None, :] == self.track_ids[:, None])[1]
            assert track_idx.shape[0] == self.track_ids.shape[0]

            d = h5_tracks[timestamp]['d'][:]
            y_disp[:, i_t] = d[track_idx]

            uv = h5_tracks[timestamp]['uv'][:]
            frame_center = np.rint(uv[track_idx])

            frame_patches[:, i_t, :, :, :] = get_patches(images[i_t, :, :], frame_center, self.config.patch_size)
            event_patches[:, i_t, :, :, :] = get_event_patches(event_reprs[i_t, :, :, :], frame_center,
                                                               self.config.patch_size, self.config.disp_patch_range)
            if self.additional_outputs:
                frame_centers[:, i_t, :] = frame_center

        h5_tracks.close()

        frame_patches = frame_patches.astype(np.float32)
        frame_patches[:, :, :, :, 0] = frame_patches[:, :, :, :, 0] / 255.

        if self.augment:
            h, w = images.shape[1:3]
            transl_x = np.random.uniform(-20 / w, 20 / w, [n_tracks, 1])
            transl_y = np.random.uniform(-20 / h, 20 / h, [n_tracks, 1])
            transl = np.concatenate([transl_x, transl_y], axis=1)
            frame_patches[:, :, :, :, 1:] += transl[:, None, None, None, :]

        frame_idx_array = np.ones([n_tracks], dtype=np.int32) * idx_track

        if self.additional_outputs:
            seq_name = self.data_path.split('/')[-2] + '/' + str(timestamps[0])
            return frame_patches, event_patches, y_disp, frame_idx_array, [seq_name] * n_tracks, frame_centers
        else:
            return frame_patches, event_patches, y_disp, frame_idx_array


class TrackDataset(Dataset):
    """
    Dataloader for a collection of feature tracks. __getitem__ returns an instance of TrackData.
    """
    def __init__(self, track_tuples, min_track_length, augment=False, patch_size=31, disp_patch_range=62,
                 additional_outputs=False, randomize_tracks=False, max_tracks_per_sample=None):
        super(TrackDataset, self).__init__()
        self.track_tuples = track_tuples
        self.min_track_length = min_track_length
        self.patch_size = patch_size
        self.disp_patch_range = disp_patch_range
        self.augment = augment
        self.additional_outputs = additional_outputs
        self.randomize_tracks = randomize_tracks
        self.max_tracks_per_sample = max_tracks_per_sample
        print(f"Created dataset with {len(self.track_tuples)} frames.")

    def __len__(self):
        return len(self.track_tuples)

    def __getitem__(self, idx_track):
        track_tuple = self.track_tuples[idx_track]

        if self.randomize_tracks:
            nr_tracks = track_tuple[2].shape[0]
            permutation_idx = torch.randperm(nr_tracks).numpy()
            track_tuple[2] = track_tuple[2][permutation_idx][:min(nr_tracks, self.max_tracks_per_sample)]

        data_config = TrackDataConfig(self.patch_size,
                                      self.augment,
                                      self.disp_patch_range)
        track_loader = TrackData(track_tuple, data_config, self.min_track_length, self.additional_outputs, self.augment)
        return track_loader.get_data(idx_track)


class M3EDDataModule(LightningDataModule):
    def __init__(self, data_dir, disp_patch_range, min_tracks_per_sample, max_tracks_per_sample, min_track_length,
                 batch_size=16, num_workers=8, patch_size=31, augment=False, n_train=20000, n_val=2000, **kwargs):
        super(M3EDDataModule, self).__init__()

        random.seed(1234)

        self.num_workers = num_workers
        self.n_train = n_train
        self.n_val = n_val

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.patch_size = patch_size
        self.min_track_length = min_track_length
        self.min_tracks_per_sample = min_tracks_per_sample
        self.max_tracks_per_sample = max_tracks_per_sample
        self.disp_patch_range = disp_patch_range

        self.dataset_train, self.dataset_val = None, None

        self.split_track_tuples = {'train': [], 'val': []}
        self.split_max_samples = {'train': n_train, 'val': n_val}
        sequences = os.listdir(self.data_dir)

        self.extract_sequences(sequences)

    def extract_sequences(self, sequences):
        for sequence in sequences:
            if sequence in val_sequences:
                split_name = 'val'
                complete_track_list = False
            elif sequence in train_sequences:
                split_name = 'train'
                complete_track_list = True
            elif sequence in test_sequences:
                continue
            else:
                raise ValueError(f"Sequence {sequence} not found in train, val or test sequences.")

            if self.split_max_samples[split_name] < len(self.split_track_tuples[split_name]):
                self.split_track_tuples[split_name] = self.split_track_tuples[split_name][:self.split_max_samples[split_name]]
                continue

            seq_track_list = self.get_tracks(sequence, margin=10, complete_track_list=complete_track_list)
            self.split_track_tuples[split_name] += seq_track_list

    def get_tracks(self, sequence, margin, complete_track_list=False):
        sequence_path = os.path.join(self.data_dir, sequence)
        valid_timestamps = []
        f = h5py.File(os.path.join(sequence_path, 'rectified_tracks.h5'), 'r')

        margin_counter = 0  # timestamps
        timestamp_list = [int(timstamp) for timstamp in list(f.keys())]
        timestamp_list.sort()
        for i, timestamp in enumerate(timestamp_list):
            if margin_counter < margin:
                margin_counter += 1
                continue

            timestamp = str(timestamp)
            track_ids = f[timestamp]['track_id'][:]
            mask_valid_tracks = f[timestamp]['track_length'][:] >= self.min_track_length

            # Batch tracks per frame
            nr_valid_tracks = mask_valid_tracks.sum()
            if nr_valid_tracks < self.min_tracks_per_sample:
                continue

            valid_indices = np.nonzero(mask_valid_tracks)[0].astype(np.uint16)

            if complete_track_list:
                n_max_tracks = nr_valid_tracks // self.max_tracks_per_sample
                valid_timestamps += [[sequence_path, i, track_ids[valid_indices]] for _ in range(n_max_tracks + 1)]

            else:
                valid_indices = np.random.permutation(valid_indices)
                n_max_tracks = nr_valid_tracks // self.max_tracks_per_sample
                n_remaining_tracks = nr_valid_tracks % self.max_tracks_per_sample
                valid_indices_list = []
                if n_max_tracks > 0:
                    max_valid_indices = valid_indices[:n_max_tracks * self.max_tracks_per_sample]
                    max_valid_indices = max_valid_indices.reshape([n_max_tracks, -1])
                    valid_indices_list += list(max_valid_indices)
                if n_remaining_tracks >= self.min_tracks_per_sample:
                    remain_valid_indices = valid_indices[n_max_tracks * self.max_tracks_per_sample:]
                    valid_indices_list += [remain_valid_indices]
                valid_timestamps += [[sequence_path, i, track_ids[valid_idx]] for valid_idx in valid_indices_list]

            margin_counter = 0

        f.close()

        return valid_timestamps

    def setup(self, stage=None):
        # Create train and val splits
        self.dataset_train = TrackDataset(self.split_track_tuples['train'],
                                          min_track_length=self.min_track_length,
                                          patch_size=self.patch_size, augment=self.augment,
                                          disp_patch_range=self.disp_patch_range,
                                          randomize_tracks=True,
                                          max_tracks_per_sample=self.max_tracks_per_sample,
                                          )

        self.dataset_val = TrackDataset(self.split_track_tuples['val'],
                                        min_track_length=self.min_track_length,
                                        patch_size=self.patch_size, augment=False,
                                        disp_patch_range=self.disp_patch_range)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, collate_fn=concat_collate, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, collate_fn=concat_collate, pin_memory=True)


class M3EDTestDataModule(M3EDDataModule):
    def __init__(self, data_dir, disp_patch_range, tracks_per_sample, min_track_length, batch_size=16, num_workers=8,
                 patch_size=31, augment=False, n_train=20000, n_val=2000, **kwargs):
        self.test_track_tuples = []
        self.tracks_per_sample = tracks_per_sample
        super(M3EDTestDataModule, self).__init__(data_dir, disp_patch_range, min_tracks_per_sample=None,
                                                 max_tracks_per_sample=None, min_track_length=min_track_length,
                                                 batch_size=batch_size, num_workers=num_workers, patch_size=patch_size,
                                                 **kwargs)

    def extract_sequences(self, sequences):
        for sequence in sequences:
            if sequence in val_sequences or sequence in train_sequences:
                continue
            elif sequence not in test_sequences:
                raise ValueError(f"Sequence {sequence} not found in train, val or test sequences.")

            seq_track_list = self.get_tracks(sequence, margin=10)
            self.test_track_tuples += seq_track_list

    def get_tracks(self, sequence, margin):
        sequence_path = os.path.join(self.data_dir, sequence)
        valid_timestamps = []
        f = h5py.File(os.path.join(sequence_path, 'rectified_tracks.h5'), 'r')

        margin_counter = 0  # timestamps
        timestamp_list = [int(timstamp) for timstamp in list(f.keys())]
        timestamp_list.sort()
        for i, timestamp in enumerate(timestamp_list):
            if margin_counter < margin:
                margin_counter += 1
                continue

            timestamp = str(timestamp)
            track_ids = f[timestamp]['track_id'][:]
            mask_valid_tracks = f[timestamp]['track_length'][:] >= self.min_track_length

            # Batch tracks per frame
            nr_valid_tracks = mask_valid_tracks.sum()
            if nr_valid_tracks < self.tracks_per_sample:
                continue

            valid_indices = np.nonzero(mask_valid_tracks)[0].astype(np.uint16)
            valid_indices = np.random.permutation(valid_indices)
            valid_indices = valid_indices[:nr_valid_tracks // self.tracks_per_sample * self.tracks_per_sample]
            valid_indices = valid_indices.reshape([nr_valid_tracks // self.tracks_per_sample, -1])
            valid_indices_list = list(valid_indices)
            valid_timestamps += [[sequence_path, i, track_ids[valid_idx]] for valid_idx in valid_indices_list]

            margin_counter = 0
        f.close()

        return valid_timestamps

    def setup(self, phase=None):
        # Create train and val splits
        self.dataset_test = TrackDataset(self.test_track_tuples,
                                          min_track_length=self.min_track_length,
                                          patch_size=self.patch_size, augment=self.augment,
                                          disp_patch_range=self.disp_patch_range, additional_outputs=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=False, collate_fn=concat_collate, pin_memory=True, shuffle=False)
