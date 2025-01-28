import os
import csv
import numpy as np

RESULTS_DICT = {
    "Ours": <path>/results.npz,
}
GT_PATH = <path>/ground_truth.npz
DATASET_PATH = <path>
OUT_DIR = <path>


def save_metrics(methods_avg, methods_end_point):
    with open(os.path.join(OUT_DIR, "metrics.csv"), 'w') as file:
        writer = csv.writer(file)

        writer.writerow(["Overall Average"] + [method_key for method_key in methods_avg.keys()])

        first_method = list(methods_avg.keys())[0]
        for metric_key in methods_avg[first_method].keys():
            row = [metric_key]
            for method in methods_avg:
                row.append(methods_avg[method][metric_key])
            writer.writerow(row)

        writer.writerow([""])

        writer.writerow(["End-Point Average"] + [method_key for method_key in methods_avg.keys()])
        for metric_key in methods_end_point[first_method].keys():
            row = [metric_key]
            for method in methods_end_point:
                row.append(methods_end_point[method][metric_key])
            writer.writerow(row)

def compute_metrics(disp_gt, disp_pred, depth_gt, depth_pred):
    eps = 1e-5
    metrics = {}

    # Disparity Metrics
    disp_abs_diff = np.abs(disp_gt-disp_pred)
    metrics["disp/MAE"] = disp_abs_diff.mean(axis=0)
    metrics["disp/RMSE"] = np.sqrt((disp_abs_diff ** 2).mean(axis=0))
    metrics["disp/ratio_pe_1"] = np.mean(disp_abs_diff > 1, axis=0)
    metrics["disp/ratio_pe_2"] = np.mean(disp_abs_diff > 2, axis=0)
    metrics["disp/ratio_pe_3"] = np.mean(disp_abs_diff > 3, axis=0)
    metrics["disp/ratio_pe_4"] = np.mean(disp_abs_diff > 4, axis=0)
    metrics["disp/ratio_pe_5"] = np.mean(disp_abs_diff > 5, axis=0)

    # Depth Metrics
    depth_abs_diff = np.abs(depth_gt-depth_pred)
    metrics["depth/MAE"] = depth_abs_diff.mean(axis=0)
    metrics["depth/MAE_rel"] = (depth_abs_diff/(depth_gt+eps)).mean(axis=0)
    metrics["depth/RMS"] = np.sqrt((depth_abs_diff ** 2).mean(axis=0))
    ratio = np.max(np.stack([depth_gt / (depth_pred + eps), depth_pred / (depth_gt + eps)]), axis=0)
    metrics["depth/ratio_delta_1.25"] = np.mean(ratio <= 1.25, axis=0)
    metrics["depth/ratio_delta_1.25^2"] = np.mean(ratio <= 1.25 ** 2, axis=0)
    metrics["depth/ratio_delta_1.25^3"] = np.mean(ratio <= 1.25 ** 3, axis=0)

    return metrics

def process_method(method_name, method_path):
    ground_truth = np.load(GT_PATH)
    disp_gt = ground_truth["disparity_gt"]
    seq_names_gt = ground_truth["seq_names"]

    if method_name == 'Ours':
        predictions = np.load(method_path)
        disp_pred = predictions["disparity_pred"]
        seq_names_pred = predictions["seq_names"]
        assert np.all(seq_names_pred == seq_names_gt), "Sequence names do not match"

    else:
        raise ValueError("Method not supported")


    depth_gt = np.zeros_like(disp_gt)
    depth_pred = np.zeros_like(disp_pred)

    # Metric Calculations
    avg_metrics = compute_metrics(disp_gt.flatten(), disp_pred.flatten(), depth_gt.flatten(), depth_pred.flatten())
    end_point_metrics = compute_metrics(disp_gt[:, -1], disp_pred[:, -1], depth_gt[:, -1], depth_pred[:, -1])

    return avg_metrics, end_point_metrics


def main():
    methods_avg, methods_end_point = {}, {}
    for method_name, method_path in RESULTS_DICT.items():
        avg_metrics, end_point_metrics = process_method(method_name, method_path)
        methods_avg[method_name] = avg_metrics
        methods_end_point[method_name] = end_point_metrics

    # Save metrics
    save_metrics(methods_avg, methods_end_point)

    # Print
    for method_name in methods_avg:
        print(f"{method_name}:")
        for metric_key in methods_avg[method_name]:
            if "disp" in metric_key:
                print(f"\t{metric_key}: {methods_avg[method_name][metric_key]}")

if __name__ == "__main__":
    main()
