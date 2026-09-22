# SkNeXt

**Skeleton-guided neuronal instance and semantic segmentation for volumetric microscopy.**

[Source repository](https://github.com/albertding1113-spec/SkNeXt)

## Overview

SkNeXt trains segmentation models and uses existing SWC skeletons to guide
instance and semantic segmentation of microscopy volumes. Skeletons identify
the image blocks to process and provide neuronal identities for watershed
reconstruction. Supply prepared, proofread skeletons for large-volume inference.

## Acknowledgments

SkNeXt's **ConvNeXt V2 backbone implementation and main training/inference
workflow are based on [BiaPy](https://github.com/BiaPyX/BiaPy)**, an open-source
Python library and application for bioimage analysis. These components have
been adapted for SkNeXt's skeleton-guided neuronal reconstruction tasks.

We thank the BiaPy developers and contributors for making their code and
bioimage analysis workflows openly available. Please see the
[BiaPy repository](https://github.com/BiaPyX/BiaPy) for the upstream project,
documentation, and citation information.

## Features

- ConvNeXt V2 encoder–decoder with configurable feature widths and anisotropic downsampling.
- Foreground, center/skeleton, contour, affinity, and semantic training targets.
- Weighted binary cross-entropy and Dice loss, with channel-wise IoU monitoring.
- TIFF-to-Zarr patch preparation, optional training-patch filtering, and paired augmentation.
- Blockwise Imaris/OME-Zarr inference restricted to blocks intersecting skeletons.
- Overlap-weighted patch blending and skeleton-seeded watershed instance reconstruction.
- OME-Zarr label output, multiresolution pyramids, and resumable block progress.
- Desktop configuration editor with GPU selection, workflow controls, and live logs.
- Python utilities for SWC manipulation and exporting skeleton-associated components.

## Installation

Use an environment satisfying the package's declared **Python >= 3.14**
requirement in `pyproject.toml`. The older Python >= 3.10 comment in
`requirements.txt` does not override package metadata.

Training and inference require an NVIDIA GPU, a working driver, and
CUDA-enabled PyTorch. The current workflow rejects CPU execution and selects
one visible GPU. The GUI can edit configurations without loading PyTorch, but
requires Tkinter/Tcl-Tk; GPU discovery additionally uses `nvidia-smi`.

1. Clone the repository and enter its root:

   ```sh
   git clone https://github.com/albertding1113-spec/SkNeXt.git
   cd SkNeXt
   ```

2. In your Python environment, install compatible **torch and torchvision**
   builds using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
   Choose the CUDA build appropriate to your platform and driver.

3. Install the remaining runtime dependencies and the package:

   ```sh
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

   Installing the package alone installs only its declared configuration
   dependencies; it does not install the complete scientific runtime.

4. Check CUDA access:

   ```sh
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
   ```

Use a Zarr 3.x Python package: patch preparation uses its v3 API, while inference
reads/writes Zarr v2 OME stores through that API. Optional 3D skeleton viewing
needs the relevant NAVis viewer backend. LZW-compressed component exports may
also require `imagecodecs`.

No pretrained checkpoint is bundled in this checkout. Train a model or provide
a checkpoint compatible with the configured architecture and output channels.

## Data preparation

### Training and validation

Prepare separate directories for raw TIFF volumes and their labels:

```text
dataset/
├── train/
│   ├── raw/
│   │   ├── sample_0000.tif
│   │   └── sample_0001.tif
│   └── label/
│       ├── sample_0000.tif
│       ├── sample_0000.soma.tif
│       ├── sample_0001.tif
│       └── sample_0001.soma.tif
└── val/
    ├── raw/
    │   └── sample_0000.tif
    └── label/
        ├── sample_0000.tif
        └── sample_0000.soma.tif
```

Raw images are **ZYX** for a single channel or **CZYX** for multiple channels.
Ground-truth volumes are ZYX and must match the corresponding raw spatial shape.

- Instance labels use zero for background and positive integer IDs for objects.
- A semantic channel such as `S.soma` reads files ending in `.soma.tif`
  or `.soma.tiff`; positive values become its binary target.
- Raw and each label group are paired by **sorted file order**, not filename
  matching. Use consistent zero-padded names and equal counts.
- Foreground, center, contour, and affinity targets are generated from instance
  labels. Center/skeleton targets are generated independently within each patch.

Training recreates `train.ome.zarr` beside the training raw directory and
`val.ome.zarr` beside the validation raw directory. These contain `raw` and
`label` arrays in **NCZYX** order. Existing generated stores at those paths are
replaced when training starts.

### Inference

Provide:

- An Imaris `.ims` file or an existing **Zarr v2 OME-Zarr group with CZYX axes**.
- A directory containing one prepared neuron per `.swc` file.
- A trained checkpoint and the matching model/channel configuration.
- A destination directory or `.zarr` path for segmentation output.

The current inference reader does not accept a directory of TIFFs or arbitrary
HDF5 datasets. It uses the readers' default resolution level and time point.

SWC coordinates must already align with the input volume's voxel coordinate
system. SWC nodes store X/Y/Z; spatial bounds and configuration sizes generally
use Z/Y/X. Skeletons traced at another resolution need the appropriate
coordinate transformation before use. The CLI does not automatically scale them.

Instance IDs in output are the skeleton's position in the manager's
deterministically sorted list **plus one**, with zero reserved for background.
They are not necessarily numeric SWC filenames. Keep the SWC collection and
ordering unchanged when resuming inference.

## Train a model

Save the following as `train.yaml` in the repository root and adapt the paths.
This example trains foreground, skeleton, and soma channels using the default
model feature widths. Adjust this starting configuration to your dataset and
available hardware.

```yaml
SYSTEM:
  DEVICE: gpu

PATHS:
  RESULT_DIR:
    PATH: ./result

TASK:
  CHANNELS: [F, P, S.soma]
  CHANNEL_WEIGHTS: [1, 1, 1]
  CHANNELS_EXTRA_OPTS:
    - P:
        type: skeleton
        skeleton_mode: full
        dilation: 1
  WATERSHED:
    SEED_CHANNELS: [P]
    SEED_CHANNELS_THRESH: [auto]
    TOPOGRAPHIC_SURFACE_CHANNEL: F
    GROWTH_MASK_CHANNELS: [F]
    GROWTH_MASK_CHANNELS_THRESH: [0.3]

DATA:
  PATCH_SIZE: [32, 128, 128, 1]
  TRAIN:
    PATH: ./dataset/train/raw
    GT_PATH: ./dataset/train/label
  VAL:
    PATH: ./dataset/val/raw
    GT_PATH: ./dataset/val/label

MODEL:
  LOAD_CHECKPOINT: false

AUGMENTOR:
  ENABLE: true
  HFLIP: true
  VFLIP: true
  ZFLIP: true

TRAIN:
  ENABLE: true
  OPTIMIZER: ADAMW
  LR: 0.0001
  BATCH_SIZE: 1
  EPOCHS: 100
  PATIENCE: 20
  LR_SCHEDULER:
    NAME: warmupcosine
    MIN_LR: 0.000001
    WARMUP_COSINE_DECAY_EPOCHS: 5

INFER:
  ENABLE: false
```

Run:

```sh
python main.py --config train.yaml --run_id 0 --gpu 0
```

The equivalent installed command is `sknext --config train.yaml --run_id 0 --gpu 0`;
`python -m sknext` also accepts the same arguments.

Training saves weights when validation loss improves. Set
`MODEL.LOAD_CHECKPOINT: true` to initialize from the run's existing weights.
This **does not restore optimizer, scheduler, or epoch state**.

## Run skeleton-guided inference

Save this as `infer.yaml`. It matches the example training configuration.
If you changed the training architecture, channel order, or preprocessing,
carry those settings into your inference configuration as well.

```yaml
SYSTEM:
  DEVICE: gpu

PATHS:
  RESULT_DIR:
    PATH: ./result

TASK:
  CHANNELS: [F, P, S.soma]
  CHANNEL_WEIGHTS: [1, 1, 1]
  WATERSHED:
    SEED_CHANNELS: [P]
    SEED_CHANNELS_THRESH: [auto]
    TOPOGRAPHIC_SURFACE_CHANNEL: F
    GROWTH_MASK_CHANNELS: [F]
    GROWTH_MASK_CHANNELS_THRESH: [0.3]

DATA:
  PATCH_SIZE: [32, 128, 128, 1]
  INFER:
    PATH: ./dataset/volume.ome.zarr
    GT_PATH: ./prediction.ome.zarr
    SKELETON_PATH: ./dataset/skeletons
    CHANNEL: [0]
    OVERLAP: [0.125, 0.125, 0.125]
    BLOCK_FACTOR: [4, 4, 4]
    BLOCK_CENTRAL_FACTOR: [3, 3, 3]

MODEL:
  LOAD_CHECKPOINT: true

TRAIN:
  ENABLE: false
  BATCH_SIZE: 1

INFER:
  ENABLE: true
  SKELETON:
    ENABLE: true
  POST_PROCESSING:
    ENABLE: true
    OPERATIONS: [fill_holes, remove_small]
    VALUES: [null, 100]
```

Run with the **same result root and run ID used for training**:

```sh
python main.py --config infer.yaml --run_id 0 --gpu 0
```

The worker loads:

```text
<RESULT_DIR.PATH>/results/<run_id>/checkpoints/checkpoint_<run_id:02d>.pth
```

For this example, that is `./result/results/0/checkpoints/checkpoint_00.pth`.
Inference always loads a checkpoint, even if `MODEL.LOAD_CHECKPOINT` is false.

Inference examines spatial blocks, skips blocks without intersecting skeletons,
predicts overlapping patches in the remaining blocks, and blends probabilities.
Watershed combines predicted seeds with skeleton IDs. The writer saves the
central region of each block and builds a multiresolution pyramid on completion.

`DATA.INFER.GT_PATH` is the **output** destination despite its historical name.
A directory destination receives `result<run_id>.ome.zarr`; a `.zarr` path is
used directly. With the example channels, output channel 0 contains neuronal
instance IDs and channel 1 contains the binary soma mask. Semantic outputs use
a fixed probability threshold of 0.5.

### Resuming inference

Progress is saved after each block to
`<parent of DATA.INFER.PATH>/infer_log/infer_log_<run_id:02d>.json`.
Rerunning the same job resumes from that progress record.

Keep the input, output store, skeleton set, model, and block geometry unchanged.
The log checks the run ID and output path; it is not a complete configuration
fingerprint. For a fresh inference over the same input and run ID, archive the
old progress file and use a fresh output store before starting.

## Desktop GUI

Launch from the same Python environment used for training:

```sh
python main.py --gui
python main.py --gui --config train.yaml
```

After installation, `sknext-gui` or `sknext --gui` also opens the editor.

1. Load YAML or edit the defaults.
2. Choose **Training** or **Inference** and enter the run ID.
3. Set the working directory; all relative paths resolve against it.
4. Select an NVIDIA GPU; **Refresh GPUs** updates free/total memory.
5. Edit settings in the configuration tabs and optionally save YAML.
6. Click **Start workflow** to launch a worker and follow its live log.

The GUI uses the selected GPU's UUID, launches the current Python interpreter,
and saves timestamped configuration snapshots and console logs in the run's
results directory. It leaves `config.py` unchanged. Opening a YAML file does
not change the working directory.

**Stop** terminates the worker without saving an extra checkpoint or completing
the current batch. Closing an active window asks whether to stop the workflow.
One GUI instance runs one workflow at a time.

## Configuration reference

Defaults live in `sknext/config/config.py`. YAML overrides known keys;
unknown keys raise an error. Derived output/checkpoint paths are recomputed
from the result root and run ID. Set exactly one workflow flag: if both are
true, the backend gives training priority.

| Setting | Meaning |
| --- | --- |
| `DATA.PATCH_SIZE` | Model patch dimensions in **Z, Y, X, C** order. |
| `TASK.CHANNELS` | Ordered prediction targets. Keep instance targets before semantic targets for inference. |
| `TASK.CHANNEL_WEIGHTS` | One loss weight per actual output channel, including expanded affinity channels. |
| `TASK.CHANNELS_EXTRA_OPTS` | A list containing one dictionary of per-target options. |
| `MODEL.FEATURE_MAPS` | Feature widths at successive model levels; default `[16, 32, 64, 128, 256]`. |
| `MODEL.Z_DOWN`, `MODEL.YX_DOWN` | Spatial downsampling at each level transition. |
| `MODEL.ISOTROPY`, `MODEL.CONVNEXT_LAYERS` | Per-level geometry/block settings; lengths must match feature levels. |
| `TRAIN.BATCH_SIZE` | Patch batch size for **both training and inference**. |
| `TRAIN.LR`, `EPOCHS`, `PATIENCE` | Learning rate, maximum epochs, and early-stopping patience. |
| `TRAIN.LR_SCHEDULER.NAME` | `warmupcosine` or `one_cycle`. |
| `DATA.NORMALIZATION` | Min/max or mean/std normalization, with optional percentile clipping. |
| `DATA.TRAIN.FILTER` | Optional filters on original raw mean or foreground label fraction; validation is unfiltered. |
| `DATA.*.OVERLAP` | Per-axis overlap fractions in `[0, 1)`, in ZYX order. |
| `DATA.*.PADDING` | Per-axis padding in voxels, in ZYX order. |
| `DATA.INFER.CHANNEL` | List of input channel indices; its length must match patch C. |
| `DATA.INFER.BLOCK_FACTOR` | Expanded inference block size as a multiple of spatial patch size. |
| `DATA.INFER.BLOCK_CENTRAL_FACTOR` | Written central block size; cannot exceed the expanded block. |
| `TASK.WATERSHED` | Seed/growth channels and thresholds; provide one threshold per selected channel or an empty list for automatic thresholds. |
| `INFER.POST_PROCESSING` | Supported workflow operations: `fill_holes`, `remove_small`, `remove_large`. |

Supported task channels:

| Channel | Generated target |
| --- | --- |
| `F` | Foreground from nonzero instance labels. |
| `P` | Per-instance skeleton or centroid; skeleton mode can be `full` or `main`. |
| `C` | Instance contours. |
| `A` | Same-instance affinities for configured Z/Y/X offsets. |
| `S.<name>` | Independent binary semantic target from the corresponding label files. |

Patch dimensions must be compatible with the model's stem and cumulative
downsampling. Increasing patch size, feature widths, batch size, or block
factors increases memory demand; inference also allocates large block buffers
in host RAM. The examples are not a guarantee of fitting a particular GPU.

## Outputs

For `RESULT_DIR.PATH: ./result` and `--run_id 0`:

```text
result/results/0/
├── checkpoints/
│   └── checkpoint_00.pth
├── log/                     # TensorBoard events
├── chart/                   # Training curves exported every five epochs
├── gui-<timestamp>.yaml     # Configuration snapshot for GUI launches
└── gui-<timestamp>.log      # Console output for GUI launches
```

Segmentation output is stored separately at `DATA.INFER.GT_PATH`.
Training patch stores and inference progress logs are placed beside the
configured data paths, as described above.

View training metrics with:

```sh
tensorboard --logdir ./result/results/0/log
```

Checkpoints contain the model state dictionary and architecture name. OME-Zarr
segmentation arrays use **CZYX** layout and **uint16** values; zero is background
and positive instance IDs must fit within 65,535.

## Skeleton-associated component export

The Python-only `Skeleton_Workflow` can associate connected components with
neurons and export cropped masks, spatial metadata, and paths to a soma/root:

```python
from sknext.run.skeletonworkflow import Skeleton_Workflow

workflow = Skeleton_Workflow(
    skeleton_path="./dataset/skeletons",
    file_path="./prediction.ome.zarr",
    result_path="./components",
    channel=[1],             # Soma semantic channel in the example output
    min_values=[100],        # Minimum connected-component voxel count
    subregion=[[-1]],        # Accept all skeleton compartments
)
workflow.run()
```

Exports include `data.tif`, `data.json`, and `data.swc` grouped by input channel
and skeleton index. `subregion` can restrict association to SWC compartment
labels such as soma (1), axon (2), basal dendrite (3), or apical dendrite (4).
This workflow is separate from the GUI's training/inference selector.

## Current implementation notes

- The bundled `sknext/config/config.yaml` contains machine-specific paths and
  a seed-threshold list whose length does not match its seed-channel list.
  Adapt it before use, or start with the examples in this README.
- `config_ins_A.yaml` uses legacy `TASK.TYPE`/`TASK.INSTANCE` keys and a scalar
  inference channel. The current schema uses flat `TASK.CHANNELS`,
  `TASK.CHANNELS_EXTRA_OPTS`, and a channel list.
- Inference currently requires SWC files even though the configuration includes
  a skeleton-enable flag. Blocks outside the supplied skeletons are skipped.
- Several configuration fields are not active switches in the current inference
  path, including `INFER.SAVE_MODEL_RAW_OUTPUT`,
  `DATA.INFER.RESULT_INTO_FILE`, and `DATA.INFER.INPUT_AXES_ORDER`.
  The implemented writer produces a separate OME-Zarr label volume.
- The CLI accepts a comma-separated GPU selector, but the workflow uses
  `cuda:0` among visible devices; it does not launch distributed or multi-GPU training.
- Files in `sknext/test/` include manual utilities with local paths and optional
  dependencies. Inspect them before execution.

## Project layout

```text
main.py                  # CLI / GUI launcher
sknext/
├── __init__.py          # Command-line argument handling
├── _sknext.py           # Configuration and CUDA setup
├── gui.py               # Desktop editor and worker management
├── config/              # Defaults and example YAML files
├── data/                # I/O, patches, augmentation, labels, processing
├── model/               # U-NeXt V2, blocks, heads, loss, schedulers
├── run/                 # Segmentation and component-export workflows
├── skeleton/            # SWC geometry, rasterization, queries, visualization
├── utils/               # Reproducibility and model/memory reporting
└── test/                # Manual data-processing utilities
tests/                   # Launcher unit tests and desktop smoke check
```

Run the self-contained launcher tests:

```sh
python -m unittest discover -s tests -v
```

On a machine with Tkinter and a display, the separate desktop smoke check is:

```sh
python tests/gui_smoke.py
```

These checks exercise the launcher and configuration handling. Segmentation
accuracy should be evaluated separately on your own validation data.
