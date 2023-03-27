"""
Compare our results against KLT with reduced frame-rate
python -m scripts.benchmark
"""
from pathlib import Path

import matplotlib.pyplot as plt

# from sklearn.metrics import auc as compute_auc
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from utils.dataset import EvalDatasetType
from utils.track_utils import compute_tracking_errors, read_txt_results

plt.rcParams["font.family"] = "serif"

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

error_threshold_range = np.arange(1, 32, 1)
results_dir = Path(
    "<path>/benchmark_data"
)
out_dir = Path(
    "<path>/benchmark_results"
)
methods = ["network_pred"]

table_keys = [
    "age_5_mu",
    "age_5_std",
    "te_5_mu",
    "te_5_std",
    "age_mu",
    "age_std",
    "inliers_mu",
    "inliers_std",
    "expected_age",
]
tables = {}
for k in table_keys:
    tables[k] = PrettyTable()
    tables[k].title = k
    tables[k].field_names = ["Sequence Name"] + methods

for eval_sequence in EVAL_DATASETS:
    sequence_name = eval_sequence[0]
    track_data_gt = read_txt_results(
        str(results_dir / "gt" / f"{sequence_name}.gt.txt")
    )

    rows = {}
    for k in tables.keys():
        rows[k] = [sequence_name]

    for method in methods:
        inlier_ratio_arr, fa_rel_nz_arr = [], []

        track_data_pred = read_txt_results(
            str(results_dir / f"{method}" / f"{sequence_name}.txt")
        )

        if track_data_pred[0, 1] != track_data_gt[0, 1]:
            track_data_pred[:, 1] += -track_data_pred[0, 1] + track_data_gt[0, 1]

        for thresh in error_threshold_range:
            fa_rel, _ = compute_tracking_errors(
                track_data_pred,
                track_data_gt,
                error_threshold=thresh,
                asynchronous=False,
            )

            inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
            if inlier_ratio > 0:
                fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
            else:
                fa_rel_nz = [0]
            inlier_ratio_arr.append(inlier_ratio)
            fa_rel_nz_arr.append(np.mean(fa_rel_nz))

        mean_inlier_ratio, std_inlier_ratio = np.mean(inlier_ratio_arr), np.std(
            inlier_ratio_arr
        )
        mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz_arr), np.std(fa_rel_nz_arr)
        expected_age = np.mean(np.array(inlier_ratio_arr) * np.array(fa_rel_nz_arr))

        rows["age_mu"].append(mean_fa_rel_nz)
        rows["age_std"].append(std_fa_rel_nz)
        rows["inliers_mu"].append(mean_inlier_ratio)
        rows["inliers_std"].append(std_inlier_ratio)
        rows["expected_age"].append(expected_age)

        fa_rel, te = compute_tracking_errors(
            track_data_pred, track_data_gt, error_threshold=5, asynchronous=False
        )
        inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
        if inlier_ratio > 0:
            fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
        else:
            fa_rel_nz = [0]
            te = [0]

        mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz), np.std(fa_rel_nz)
        mean_te, std_te = np.mean(te), np.std(te)
        rows["age_5_mu"].append(mean_fa_rel_nz)
        rows["age_5_std"].append(std_fa_rel_nz)
        rows["te_5_mu"].append(mean_te)
        rows["te_5_std"].append(std_te)

    # Load results
    for k in tables.keys():
        tables[k].add_row(rows[k])

with open((out_dir / f"benchmarking_results.csv"), "w") as f:
    for k in tables.keys():
        f.write(f"{k}\n")
        f.write(tables[k].get_csv_string())

        print(tables[k].get_string())
