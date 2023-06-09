# Data-driven Feature Tracking for Event Cameras

<p align="center">
 <a href="https://youtu.be/dtkXvNXcWRY">
  <img src="doc/thumbnail.PNG" alt="youtube_video" width="800"/>
 </a>
</p>

This is the code for the CVPR23 paper **Data-driven Feature Tracking for Event Cameras**
([PDF](https://rpg.ifi.uzh.ch/docs/Arxiv22_Messikommer.pdf)) by [Nico Messikommer\*](https://messikommernico.github.io/), [Carter Fang\*](https://ctyfang.github.io/), [Mathis Gehrig](https://magehrig.github.io/), and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html).
For an overview of our method, check out our [video](https://youtu.be/dtkXvNXcWRY).

If you use any of this code, please cite the following publication:

```bibtex
@Article{Messikommer23cvpr,
  author  = {Nico Messikommer* and Carter Fang* and Mathias Gehrig and Davide Scaramuzza},
  title   = {Data-driven Feature Tracking for Event Cameras},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition},
  year    = {2023},
}
```

## Abstract

Because of their high temporal resolution, increased resilience to motion blur, and very sparse output, event cameras have been shown to be ideal for low-latency and low-bandwidth feature tracking, even in challenging scenarios.
Existing feature tracking methods for event cameras are either handcrafted or derived from first principles but require extensive parameter tuning, are sensitive to noise, and do not generalize to different scenarios due to unmodeled effects.
To tackle these deficiencies, we introduce the first data-driven feature tracker for event cameras, which leverages low-latency events to track features detected in a grayscale frame.
We achieve robust performance via a novel frame attention module, which shares information across feature tracks.
By directly transferring zero-shot from synthetic to real data, our data-driven tracker outperforms existing approaches in relative feature age by up to 120% while also achieving the lowest latency.
This performance gap is further increased to 130% by adapting our tracker to real data with a novel self-supervision strategy.

<p align="center">
  <img alt="ziggy" src="./doc/ziggy_in_the_arena_1350_1650-opt.gif" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="shapes_6dof" src="./doc/shapes_6dof_485_565_tracks.gif" width="45%">
</p>

---

## Content

This document describes the usage and installation for this repository.<br>

1. [Installation](#Installation)<br>
2. [Test Sequences and Pretrained Weights](#Test-Sequences-and-Pretrained-Weights)<br>
3. [Preparing Synthetic Data](#Preparing-Synthetic-Data)<br>
4. [Training on Synthetic Data](#Training-on-Synthetic-Data)<br>
5. [Preparing Pose Data](#Preparing-Pose-Data)<br>
6. [Training on Pose Data](#Training-on-Pose-Data)<br>
7. [Preparing Evaluation Data](#Preparing-Evaluation-Data)<br>
8. [Running Ours](#Running-Ours)<br>
9. [Evaluation](#Evaluation)<br>
10. [Visualization](#Visualization)<br>

---

## Installation

This guide assumes use of Python 3.9.7<br>

1. If desired, a conda environment can be created using the following command:

```bash
conda create -n <env_name>
```

2.  Install the dependencies via the requirements.txt file<br>
    `pip install -r requirements.txt`<br><br>

    Dependencies for training:
    <ul>
        <li>PyTorch</li>
        <li>Torch Lightning</li>
        <li>Hydra</li>
    </ul><br>
    
    Dependencies for pre-processing:
    <ul>
        <li>numpy</li>
        <li>OpenCV</li>
        <li>H5Py and HDF5Plugin</li>
    </ul><br>
    
    Dependencies for visualization:
    <ul>
        <li>matplotlib</li>
        <li>seaborn</li>
        <li>imageio</li>
    </ul><br>
---

## Test Sequences and Pretrained Weights

To facilitate the evaluation of the tracking performance, we provide the raw events, multiple event representation, etc., for 
the used test sequences of the [Event Camera Dataset](https://download.ifi.uzh.ch/rpg/CVPR23_deep_ev_tracker/ec_subseq.zip)
and the [EDS dataset](https://download.ifi.uzh.ch/rpg/CVPR23_deep_ev_tracker/eds_subseq.zip).
The ground truth tracks for both EC and EDS datasets generated based on the camera poses and KLT tracks can be downloaded [here](https://download.ifi.uzh.ch/rpg/CVPR23_deep_ev_tracker/gt_tracks.zip).

Furthermore, we also provide the [network weights](https://download.ifi.uzh.ch/rpg/CVPR23_deep_ev_tracker/pretrained_weights.zip) trained on the Multiflow dataset, the weights fine-tuned on the EC, 
and fine-tuned on the EDS dataset using our proposed pose supervision strategy.


---

## Preparing Synthetic Data

### Download MultiFlow Dataset

Download links:

- [train (1.3 TB)](https://download.ifi.uzh.ch/rpg/multiflow/train.tar)
- [test (258 GB)](https://download.ifi.uzh.ch/rpg/multiflow/test.tar)

If you use this dataset in an academic context, please cite:

```bibtex
@misc{Gehrig2022arxiv,
 author = {Gehrig, Mathias and Muglikar, Manasi and Scaramuzza, Davide},
 title = {Dense Continuous-Time Optical Flow from Events and Frames},
 url = {https://arxiv.org/abs/2203.13674},
 publisher = {arXiv},
 year = {2022}
}
```

The models were pre-trained using an older version of this dataset, available at the time of the submission.
The download links above link to the up-to-date version of the dataset.

### Pre-Processing Instructions

Preparation of the synthetic data involves generating input representations for the
Multiflow sequences and extracting the ground-truth tracks.<br>

To generate ground-truth tracks, run:<br>
`python data_preparation/synthetic/generate_tracks.py <path_to_multiflow_dataset> 
<path_to_multiflow_extras_dir>`<br>

Where the Multiflow Extras directory contains data needed to train our network such as the
ground-truth tracks and input event representations.<br>

To generate input event representations, run:<br>
`python data_preparation/synthetic/generate_event_representations <path_to_multiflow_dataset> 
<path_to_multiflow_extras_dir> <representation_type>`<br>

The resulting directory structure is:<br>

```
multiflow_reloaded_extra/
├─ sequence_xyz/
│  ├─ events/
│  │  ├─ 0.0100/
│  │  │  ├─ representation_abc/
│  │  │  │  ├─ 0400000.h5
│  │  │  │  ├─ 0410000.h5
│  │  ├─ 0.0200/
│  │  │  ├─ representation_abc/
│  │
│  ├─ tracks/
│  │  ├─ shitomasi.gt.txt
```

---

## Training on Synthetic Data

Training on synthetic data involves configuring the dataset, model, and training. The high-level config is at
`configs/train_defaults.yaml`.<br>

To configure the dataset:

<ul>
    <li>Set data field to mf</li>
    <li>Configure the synthetic dataset in configs/data/mf.yaml</li>
    <li>Set the track_name field (default is shitomasi_custom)</li>
    <li>Set the event representation (default is SBT Max, referred to as time_surfaces_v2_5 here)</li>
</ul>

Important parameters in mf.yaml are:<br>

<ul>
    <li>augment - Whether to augment the tracks or not. The actual limits for augmentations are defined as 
        global variables in utils/datasets.py</li>
    <li>mixed_dt - Whether to use both timesteps of 0.01 and 0.02 during training.</li>
    <li>n_tracks/val - Number of tracks to use for validation and training. All tracks are loaded, shuffled, 
        then trimmed.</li>
</ul>

To configure the model, set the model field to one of the available options in `configs/model`. Our default
model is `correlation3_unscaled`.<br>

To configure the training process:

<ul>
    <li>Set the learning rate in configs/optim/adam.yaml (Default is 1e-4)</li>
    <li>In configs/training/supervised_train.yaml, set the sequence length schedule via init_unrolls, max_unrolls, unroll_factor, and the schedule. 
        At each of the specified training steps, the number of unrolls will be multiplied by the unroll factor.</li>
    <li>Configure the synthetic dataset in configs/data/mf.yaml</li>
</ul>

The last parameter to set is `experiment` for organizational purposes.<br>

With everything configured, we can begin training by running

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py
```

Hydra will then instantiate the dataloader and model.
PyTorch Lightning will handle the training and validation loops.
All outputs (checkpoints, gifs, etc) will be written to the log directory.<br>

The correlation_unscaled model inherits from `models/template.py` since it contains the core logic for training and validation.
At each training step, event patches are fetched for each feature track (via `TrackData` instances) and concatenated
prior to inference. Following inference, the `TrackData` instances accumulate the predicted feature displacements.
The template file also contains the validation logic for visualization and metric computation.

To inspect models during training, we can launch an instance of tensorboard for the log directory:
`tensorboard --logdir <log_dir>`.

---

## Preparing Pose Data

To prepare pose data for fine-tuning, we need to rectify the data, run colmap, and generate event
representations. <br>

To rectify the data, run `python data_preparation/real/rectify_ec.py` or
`python data_preparation/real/eds_rectify_events_and_frames.py`.<br>

To refine the pose data with colmap, see `data_preparation/colmap.py`. We first run `colmap.py generate`.
This will convert the pose data to a readable format for COLMAP to serve as an initial guess,
generated in the `colmap` directory of the sequence. We then follow the instructions
[here](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses) from the COLMAP
FAQ regarding refining poses.<br>

Essentially:

<ol>
    <li>Navigate to the colmap directory of a sequence</li>
    <li>colmap feature_extractor --database_path database.db</li>
    <li>colmap exhaustive_matcher --database_path database.db --image_path ../images_corrected</li>
    <li>colmap point_triangulator --database_path database.db --image_path ../images_corrected/ 
        --input_path . --output_path .</li>
    <li>Launch the colmap gui, import the model files, and re-run Bundle Adjustment ensuring that
        only extrinsics are refined.</li>
    <li>Run colmap.py extract to convert the pose data from COLMAP format back to our standard format.</li>
</ol>

To generate event representations, run `python data_preparation/real/prepare_eds_pose_supervision.py`
or `prepare_ec_pose_supervision.py`. These scripts generate `r` event representations between frames.
The time-window of the last event representation in the interval is trimmed. Currently, these scripts
only support SBT-Max as a representation.

---

## Training on Pose Data

To train on pose data, we again need to configure the dataset, model, and training. The model configuration is the
same as before. The `data` field now needs to be set to `pose_ec`, and `configs/data/pose_ec.yaml` must be
configured.<br>

Important parameters to set in `pose.yaml` include:

<ul>
    <li>root_dir - Directory with prepared pose data sequences.</li>
    <li>n_frames_skip - How many frames to skip when chunking a sequence into several
        sub-sequences for pose training.</li>
    <li>n_event_representations_per_frame - r value used when generating the event representations.</li>
</ul>

In terms of dataset configuration must also set `pose_mode = True` in `utils/dataset.py`. This overrides the loading of
event representations from the time-step directories (eg `0.001`) and instead from the pose data directories
(eg `pose_3`).<br>

In terms of the training process, for pose supervision we use a single sequence length so `init_unrolls` and
`max_unrolls` should be the same value. Also, the schedule should have a single value indicating when to stop training.
The default learning rate for pose supervision is `1e-6`.<br>

Since we are fine-tuning on pose, we must also set the `checkpoint_path` in `configs/training/pose_finetuning_train_ec.yaml` to the path of our pretrained model.

We are then ready to run `train.py` and fine-tune the network.
Again, during training, we can launch tensorboard.
For pose supervision, the re-projected features are visualized.<br>

---

## Running Ours

### Preparing Input Data

The `SequenceDataset` class is responsible for loading data for inference.
It expects a similar data format for the sequence as with synthetic training:<br>

```
sequence_xyz/
├─ events/
│  ├─ 0.0100/
│  │  ├─ representation_abc/
│  │  │  ├─ 0000000.h5
│  │  │  ├─ 0010000.h5
├─ images_corrected/
```

To prepare a single sequence for inference, we rectify the sequence, a sequence segment, and generate
event representations.<br>

#### Rectification

For the EDS dataset, we download the txt-based version of a sequence and run `data_preparation/real/eds_rectify_events_and_frames.py`.<br>
For the Event-Camera dataset, we download the txt-based version of a sequence and run `data_preparation/real/rectify_ec.py`.<br>

#### Sequence Cropping and Event Generation

For the EDS dataset, we run `data_preparation/real/prepare_eds_subseq` with the index range for
the cropped sequence as inputs. This will generate a new sub-sequence directory, copy the
relevant frames for the selected indices, and generate event representations.

### Inference

The inference script is `evaluate_real.py` and the configuration file is `eval_real_defaults.yaml`.
We must set the event representation and checkpoint path before running the script.<br>

The list of sequences is defined in the `EVAL_DATASETS` variable in `evaluate_real.py`.
The script iterates over these sequences, instantiates a SequenceDataset instance for each one,
and performs inference on the event representations generated in the previous section.

For benchmarking, the provided feature points need to be downloaded and used in order to ensure that all methods use
the same features.
The `gt_path` needs to be set in `eval_real_defaults.yaml` to the directory containing the text files.

---

## Evaluation

Once we have predicted tracks for a sequence using all methods, we can
benchmark their performance using `scripts/benchmark.py`. This script loads
the predicted tracks for each method and compares them against the re-projected,
frame-based ground-truth tracks, which can be downloaded here.<br>
Inside the `scripts/benchmark.py`, the evaluation sequences, the results directory, the output directory and
the name of the test methods `<method_name_x>` need to be specified.
The result directory should have the following structure:

```
sequence_xyz/
├─ gt/
│  ├─ <seq_0>.gt.txt
│  ├─ <seq_1>.gt.txt
│  ├─ ...
├─ <method_name_1>/
│  ├─ <seq_0>.txt
│  ├─ <seq_1>.txt
│  ├─ ...
├─ <method_name_2>/
│  ├─ <seq_0>.txt
│  ├─ <seq_1>.txt
│  ├─ ...
```

The results are printed to the console and written to a CSV in the output directory.
