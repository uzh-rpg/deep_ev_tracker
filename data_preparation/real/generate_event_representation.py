from pathlib import Path

import fire
import h5py
import numpy as np
from tqdm import tqdm

from data_preparation.synthetic.generate_voxel_grids import (
    blosc_opts,
    events_to_voxel_grid,
)
from utils.dataset import ECPoseSegmentDataset, EDSPoseSegmentDataset
from utils.representations import (
    EventStack,
    VoxelGrid,
    events_to_event_stack,
    events_to_time_surface,
)


def generate(sequence_dir, sequence_type, representation_type, dt):
    """
    :param sequence_dir:
    :param sequence_type:
    :param representation_type:
    :param dt:
    :return:
    """

    sequence_dir = Path(sequence_dir)
    output_dir = sequence_dir / "events" / f"{dt:.4f}" / f"{representation_type}s_5"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if sequence_type == "EDS":
        dataset_class = EDSPoseSegmentDataset
    elif sequence_type == "EC":
        dataset_class = ECPoseSegmentDataset
    else:
        raise NotImplementedError(f"No dataset class for {sequence_type}")
        # dataset_class = FPVPoseSegmentDataset

    if representation_type == "time_surface":
        generation_function = events_to_time_surface
    if representation_type == "voxel_grid":
        generation_function = events_to_voxel_grid
        representation = VoxelGrid(
            (5, dataset_class.resolution[1], dataset_class.resolution[0]), False
        )
    elif representation_type == "event_stack":
        generation_function = events_to_event_stack
        representation = EventStack(
            (5, dataset_class.resolution[1], dataset_class.resolution[0])
        )
    else:
        raise NotImplementedError(f"No generation function for {representation_type}")

    ev_iterator = dataset_class.get_events_iterator(sequence_dir, dt=dt)

    for dt_evs, evs in tqdm(ev_iterator, desc="Generating ev reps..."):
        # Generate
        rep_tensor = generation_function(
            representation, evs["p"], evs["t"], evs["x"], evs["y"]
        ).numpy()
        rep_np = np.transpose(rep_tensor, (1, 2, 0))

        # Write to disk
        output_path = output_dir / f"{str(int(round(dt_evs*1e6))).zfill(7)}.h5"
        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                f"{representation_type}",
                data=rep_np,
                shape=rep_np.shape,
                dtype=np.float32,
                **blosc_opts(complevel=1, shuffle="byte"),
            )


if __name__ == "__main__":
    fire.Fire(generate)
