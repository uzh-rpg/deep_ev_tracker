""" Predict tracks for a sequence with a network """
import logging
import os
from pathlib import Path

import hydra
import imageio
import IPython
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict
from prettytable import PrettyTable
from tqdm import tqdm

from utils.dataset import CornerConfig, ECSubseq, EDSSubseq, EvalDatasetType
from utils.timers import CudaTimer, cuda_timers
from utils.track_utils import (
    TrackObserver,
    get_gt_corners,
)
from utils.visualization import generate_track_colors, render_pred_tracks, render_tracks

# Configure GPU order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Logging
logger = logging.getLogger(__name__)
results_table = PrettyTable()
results_table.field_names = ["Inference Time"]

# Configure datasets
corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

EvalDatasetConfigDict = {
    EvalDatasetType.EC: {"dt": 0.010, "root_dir": "<path>"},
    EvalDatasetType.EDS: {"dt": 0.005, "root_dir": "<path>"},
}

EVAL_DATASETS = [
    ("peanuts_light_160_386", EvalDatasetType.EDS),
    ("rocket_earth_light_338_438", EvalDatasetType.EDS),
    ("ziggy_in_the_arena_1350_1650", EvalDatasetType.EDS),
    ("peanuts_running_2360_2460", EvalDatasetType.EDS),
    ("shapes_translation_8_88", EvalDatasetType.EC),
    ("shapes_rotation_165_245", EvalDatasetType.EC),
    ("shapes_6dof_485_565", EvalDatasetType.EC),
    ("boxes_translation_330_410", EvalDatasetType.EC),
    ("boxes_rotation_198_278", EvalDatasetType.EC),
]


def evaluate(model, sequence_dataset, dt_track_vis, sequence_name, visualize):
    tracks_pred = TrackObserver(
        t_init=sequence_dataset.t_init, u_centers_init=sequence_dataset.u_centers
    )

    model.reset(sequence_dataset.n_tracks)
    event_generator = sequence_dataset.events()

    cuda_timer = CudaTimer(model.device, sequence_dataset.sequence_name)

    with torch.no_grad():
        # Predict network tracks
        for t, x in tqdm(
            event_generator,
            total=sequence_dataset.n_events - 1,
            desc="Predicting tracks with network...",
        ):
            with cuda_timer:
                x = x.to(model.device)
                y_hat = model.forward(x)

                sequence_dataset.accumulate_y_hat(y_hat)
            tracks_pred.add_observation(t, sequence_dataset.u_centers.cpu().numpy())

        if visualize:
            # Visualize network tracks
            gif_img_arr = []
            tracks_pred_interp = tracks_pred.get_interpolators()
            track_colors = generate_track_colors(sequence_dataset.n_tracks)
            for i, (t, img_now) in enumerate(
                tqdm(
                    sequence_dataset.frames(),
                    total=sequence_dataset.n_frames - 1,
                    desc="Rendering predicted tracks... ",
                )
            ):
                fig_arr = render_pred_tracks(
                    tracks_pred_interp, t, img_now, track_colors, dt_track=dt_track_vis
                )
                gif_img_arr.append(fig_arr)
            imageio.mimsave(f"{sequence_name}_tracks_pred.gif", gif_img_arr)

    # Save predicted tracks
    np.savetxt(
        f"{sequence_name}.txt",
        tracks_pred.track_data,
        fmt=["%i", "%.9f", "%i", "%i"],
        delimiter=" ",
    )

    metrics = {}
    metrics["latency"] = sum(cuda_timers[sequence_dataset.sequence_name])

    return metrics


@hydra.main(config_path="configs", config_name="eval_real_defaults")
def track(cfg):
    pl.seed_everything(1234)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model.representation = cfg.representation
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Configure model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    state_dict = torch.load(cfg.weights_path, map_location="cuda:0")["state_dict"]
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Run evaluation on each dataset
    for seq_name, seq_type in EVAL_DATASETS:
        if seq_type == EvalDatasetType.EC:
            dataset_class = ECSubseq
        elif seq_type == EvalDatasetType.EDS:
            dataset_class = EDSSubseq
        else:
            raise ValueError

        dataset = dataset_class(
            EvalDatasetConfigDict[seq_type]["root_dir"],
            seq_name,
            -1,
            cfg.patch_size,
            cfg.representation,
            EvalDatasetConfigDict[seq_type]["dt"],
            corner_config,
        )

        # Load ground truth corners for this seq and override initialization
        gt_features_path = str(Path(cfg.gt_path) / f"{seq_name}.gt.txt")
        gt_start_corners = get_gt_corners(gt_features_path)

        dataset.override_keypoints(gt_start_corners)

        metrics = evaluate(model, dataset, cfg.dt_track_vis, seq_name, cfg.visualize)

        logger.info(f"=== DATASET: {seq_name} ===")
        logger.info(f"Latency: {metrics['latency']} s")

        results_table.add_row([metrics["latency"]])

    logger.info(f"\n{results_table.get_string()}")


if __name__ == "__main__":
    track()
