import random

random.seed(1234)

import matplotlib.pyplot as plt
import numpy as np


def generate_track_colors(n_tracks):
    track_colors = []
    for i_track in range(n_tracks):
        track_colors.append(
            (
                random.randint(0, 255) / 255.0,
                random.randint(0, 255) / 255.0,
                random.randint(0, 255) / 255.0,
            )
        )
    return track_colors


def fig_to_img(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    w, h = fig.canvas.get_width_height()
    return data.reshape((h, w, 3))


def render_tracks(
    pred_track_interpolator,
    gt_track_interpolator,
    t,
    img,
    dt_track=0.2,
    error_threshold=15,
    track_counter=False,
):
    """
    Plot pred and gt tracks on an image with past tracks in the time window [t-dt_track, t].
    Predicted tracks that exceed the error threshold are not drawn.
    :param pred_track_interpolator:
    :param gt_track_interpolator:
    :param t:
    :param img:
    :param dt_track:
    :param error_threshold:
    :return:
    """
    h, w = img.shape[:2]
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    if img.ndim == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap="gray")
    ax.autoscale(False)

    active_pred_tracks = 0
    active_gt_tracks = 0

    # Draw each track
    for track_id in range(gt_track_interpolator.n_corners):
        gt_track_data_curr = gt_track_interpolator.interpolate(track_id, t)
        pred_track_data_curr = pred_track_interpolator.interpolate(track_id, t)

        # Check time
        if gt_track_data_curr is not None:
            out_of_frame = (
                (gt_track_data_curr[1] >= h)
                or (gt_track_data_curr[0] >= w)
                or (gt_track_data_curr[1] < 0)
                or (gt_track_data_curr[0] < 0)
            )
        if isinstance(gt_track_data_curr, type(None)) or out_of_frame:
            # print(f"No gt tracks at queried time for track idx {track_id}.")
            continue

        else:
            active_gt_tracks += 1

            # Draw tracks at query time
            ax.scatter(
                gt_track_data_curr[0],
                gt_track_data_curr[1],
                # color=[0, 1, 0], alpha=1., linewidth=1, s=30, marker='o')
                color=[255 / 255.0, 255 / 255.0, 0],
                alpha=1.0,
                linewidth=1,
                s=30,
                marker="o",
            )

            gt_track_data_hist = gt_track_interpolator.history(track_id, t, dt_track)
            ax.plot(
                gt_track_data_hist[:, 0],
                gt_track_data_hist[:, 1],
                # color=[0, 1, 0], alpha=0.5, linewidth=4, linestyle='solid')
                color=[255 / 255.0, 255 / 255.0, 0],
                alpha=0.5,
                linewidth=4,
                linestyle="solid",
            )

            if (
                not isinstance(pred_track_data_curr, type(None))
                and np.linalg.norm(pred_track_data_curr - gt_track_data_curr)
                < error_threshold
            ):
                ax.scatter(
                    pred_track_data_curr[0],
                    pred_track_data_curr[1],
                    # color=[0, 0, 1], alpha=1., linewidth=1, s=30, marker='o')
                    color=[0 / 255.0, 255 / 255.0, 255 / 255.0],
                    alpha=1.0,
                    linewidth=1,
                    s=30,
                    marker="o",
                )

                pred_track_data_hist = pred_track_interpolator.history(
                    track_id, t, dt_track
                )
                ax.plot(
                    pred_track_data_hist[:, 0],
                    pred_track_data_hist[:, 1],
                    # color=[0, 0, 1], alpha=0.5, linewidth=4, linestyle='solid')
                    color=[0 / 255.0, 255 / 255.0, 255 / 255.0],
                    alpha=0.5,
                    linewidth=4,
                    linestyle="solid",
                )

                active_pred_tracks += 1

    if track_counter:
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.imshow(img, cmap='gray')
        # ax.autoscale(False)
        # ax.text(2.5, 10, 'Active Tracks: {} / {}'.format(active_pred_tracks, active_gt_tracks),
        ax.text(
            8,
            28.5,
            "Active Tracks: {} / {}".format(active_pred_tracks, active_gt_tracks),
            fontsize=15,
            c="yellow",
            bbox=dict(facecolor="black", alpha=0.75),
        )
        # ax.axis('off')
        # plt.savefig("tmp.png")

    ax.axis("off")
    fig_array = fig_to_img(fig)
    plt.close(fig)
    return fig_array


def render_pred_tracks(pred_track_interpolator, t, img, track_colors, dt_track=0.0025):
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img, cmap="gray")
    ax.autoscale(False)

    for track_id in range(pred_track_interpolator.n_corners):
        pred_track_data_curr = pred_track_interpolator.interpolate(track_id, t)

        if not isinstance(pred_track_data_curr, type(None)):
            ax.scatter(
                pred_track_data_curr[0],
                pred_track_data_curr[1],
                color=track_colors[track_id],
                alpha=0.5,
                linewidth=1.0,
                s=30,
                marker="o",
            )

            pred_track_data_hist = pred_track_interpolator.history(
                track_id, t, dt_track
            )

            # ToDo: Change back
            pred_track_data_hist = np.concatenate(
                [pred_track_data_hist, pred_track_data_curr[None, :]], axis=0
            )

            ax.plot(
                pred_track_data_hist[:, 0],
                pred_track_data_hist[:, 1],
                color=track_colors[track_id],
                alpha=0.5,
                linewidth=4.0,
                linestyle="solid",
            )

    ax.axis("off")
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig_array = fig_to_img(fig)
    plt.close(fig)
    return fig_array
