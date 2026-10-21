# SkNeXt

**Instance segmentation, semantic-only segmentation, and image restoration for volumetric microscopy.**

[Source repository](https://github.com/albertding1113-spec/SkNeXt)

## Overview

SkNeXt uses a ConvNeXt V2 encoder–decoder for three workflows, selected from
the YAML configuration through the same `sknext` command:

- **Instance/mixed segmentation:** predict foreground, skeleton, contour,
  affinity, and optional semantic channels. Prepared SWC skeletons can select
  blocks and provide neuronal identities for watershed reconstruction.
- **Semantic-only segmentation:** predict independent binary `S.<name>` masks
  over the image without SWCs or watershed.
- **Restoration/super-resolution:** predict continuous fluorescence intensities
  at 1×, 2×, or 4× XY scale, with unchanged Z resolution.

SWC-free instance inference is also available. Its local overlap-based ID
stitching does not guarantee correct long-range neuron identities; supply
proofread skeletons when those identities are required.

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
- Blockwise Imaris/OME-Zarr inference, with optional SWC-based block selection.
- Overlap-weighted patch blending and skeleton-seeded watershed instance reconstruction.
- OME-Zarr label output, multiresolution pyramids, and resumable block progress.
- Desktop configuration editor with GPU selection, workflow controls, and live logs.
- Python utilities for SWC manipulation and exporting skeleton-associated components.
- Semantic-only probability thresholds and per-channel validation IoU.
- Paired low/high-resolution TIFF training and float32 restoration output.
- Training loss updates at 20 points per epoch for semantic-only and restoration workflows.
- GPU allocated/reserved/peak memory reporting during training and inference.

## Installation

Use an isolated environment satisfying the package's declared **Python >= 3.14**
requirement in `pyproject.toml` on each computer.

Training and inference through `sknext` require an NVIDIA GPU, a working driver,
and CUDA-enabled PyTorch. The unified launcher rejects CPU execution and selects
one visible GPU. The GUI can edit configurations without loading PyTorch, but
requires Tkinter/Tcl-Tk; GPU discovery additionally uses `nvidia-smi`.

1. Clone the repository and enter its root:

   ```sh
   git clone https://github.com/albertding1113-spec/SkNeXt.git
   cd SkNeXt
   ```

2. In your Python environment, install compatible **torch and torchvision**
   builds using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
   Choose a matching pair for that computer's operating system, Python version,
   NVIDIA driver and supported CUDA runtime. Run the selector's installation
   command before installing `requirements.txt`. The shared requirements file
   intentionally contains no PyTorch build pins or CUDA-specific package index,
   so different machines can use different supported CUDA builds.

3. Install the remaining runtime dependencies and the package:

   ```sh
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

   Installing the package alone installs only its declared configuration
   dependencies; it does not install the complete scientific runtime.

4. Check CUDA access:

   ```sh
   python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
   ```

Use a Zarr 3.x Python package: patch preparation uses its v3 API, while inference
reads/writes Zarr v2 OME stores through that API. Optional 3D skeleton viewing
needs the relevant NAVis viewer backend. LZW-compressed component exports may
require `imagecodecs`, included in `requirements.txt`.

An optional Conda environment can be created with:

```sh
conda create -n sknext python=3.14 pip
conda activate sknext
```

Use a PyTorch release that supplies wheels for both this Python version and
your target platform/CUDA build. A CUDA build from an older PyTorch release
may not support Python 3.14; the package's current Python requirement still applies.

The main launcher requires NVIDIA CUDA on a supported Windows or Linux machine.
Installing shared dependencies on macOS does not add Apple MPS or CPU support
to that launcher. The GUI can still be used for configuration editing when
Tkinter/Tcl-Tk is installed.

If an existing Conda environment reports an MKL/OpenMP runtime conflict, apply
the following only to the affected environment, replacing `ENV_NAME`, and then
reactivate it:

```sh
conda env config vars set -n ENV_NAME MKL_THREADING_LAYER=SEQUENTIAL
conda deactivate
conda activate ENV_NAME
```

This makes MKL CPU operations single-threaded without disabling CUDA. It is an
environment setting, not a pip dependency; `requirements.txt` cannot apply it.

No pretrained checkpoint is bundled in this checkout. Train a model or provide
a checkpoint compatible with the configured architecture and output channels.

## Select a workflow with `sknext`

All workflows use the same command. `TASK.CHANNELS` selects the runner:

| Channels in YAML | Workflow |
| --- | --- |
| Only `S.<name>` channels, e.g. `[S.globular]` | Semantic-only segmentation; no SWCs or watershed |
| `[SR]` | Image restoration / XY super-resolution, configured with `SR.UPSAMPLE` |
| `F`, `P`, `C`, `A`, optionally with `S.<name>` | Instance or mixed segmentation |

Enable exactly one of `TRAIN.ENABLE` and `INFER.ENABLE` in the YAML. For example:

```sh
sknext --config sknext/example/globular_train.yaml --run_id 1 --gpu 0
sknext --config sknext/example/globular_infer.yaml --run_id 1 --gpu 0
sknext --config sknext/example/super_resolution_train.yaml --run_id 2 --gpu 0
sknext --config sknext/example/super_resolution_infer.yaml --run_id 2 --gpu 0
```

Adapt the example data/result paths before running. Use the same run ID and
model settings for training and inference, or set `MODEL.CHECKPOINT_INPUT_PATH`
to an existing compatible checkpoint. `python -m sknext` accepts the same arguments.

Semantic-only inference accepts `INFER.THRESHOLD` (default `0.5`), either a
single number or one number per channel in `TASK.CHANNELS` order. Set
`INFER.SAVE_PROBABILITIES: true` to also save probability maps. Semantic options
are now part of the shared schema in `sknext/config/config.py`. The standalone
`python -m sknext.run.semantic_workflow` command remains available.

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
- Raw and labels are paired by **sample identity**, with duplicate, missing,
  and ambiguous pairs rejected. Supported label names include `sample.tif`,
  `sample_gt.tif`, and `prefix_gt_0001.tif` for raw `prefix_0001.tif`;
  semantic suffixes such as `.soma` are handled separately.
- Foreground, center, contour, and affinity targets are generated from instance
  labels. Center/skeleton targets are generated independently within each patch.

Instance/mixed training recreates `train.ome.zarr` beside the training raw directory and
`val.ome.zarr` beside the validation raw directory. These contain `raw` and
`label` arrays in **NCZYX** order. Existing generated stores at those paths are
replaced when training starts.

### Instance/mixed inference

Provide:

- An Imaris `.ims` file or an existing **Zarr v2 OME-Zarr group with CZYX axes**.
- A directory containing one prepared neuron per `.swc` file when
  `INFER.SKELETON.ENABLE: true`.
- A trained checkpoint and the matching model/channel configuration.
- A destination directory or `.zarr` path for segmentation output.

Segmentation readers do not accept a directory of TIFFs or arbitrary HDF5
datasets. They use the readers' default resolution level and time point.
Restoration additionally accepts TIFF inputs, as described below.

SWC coordinates must already align with the input volume's voxel coordinate
system. SWC nodes store X/Y/Z; spatial bounds and configuration sizes generally
use Z/Y/X. Skeletons traced at another resolution need the appropriate
coordinate transformation before use. The CLI does not automatically scale them.

Instance IDs in output are the skeleton's position in the manager's
deterministically sorted list **plus one**, with zero reserved for background.
They are not necessarily numeric SWC filenames. Keep the SWC collection and
ordering unchanged when resuming inference.

## Train an instance/mixed model

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

For continuation from the latest training state, use `TRAIN.RESUME: true` and
`MODEL.LOAD_CHECKPOINT: false`. The latest checkpoint is stored separately as
`checkpoint_<run_id:02d>_latest.pth`. Instance/mixed and semantic runners restore
verified optimizer/scheduler/RNG state when available; legacy or incomplete
checkpoints can fall back to a reported weight-only warm start. Keep the model,
data, and training schedule compatible with the saved run.

## Semantic-only segmentation

Only use `S.<name>` entries in `TASK.CHANNELS`. For `[S.globular, S.reticular]`,
prepare these pairs in each training/validation split:

```text
raw/sample_0001.tif
label/sample_0001.globular.tif
label/sample_0001.reticular.tif
```

No instance-label TIFFs or SWCs are required. Semantic labels are ZYX masks;
positive values become foreground. Each output channel is independent, so
channels may overlap rather than forming an exclusive multiclass partition.

Example `semantic_train.yaml`:

```yaml
PATHS:
  RESULT_DIR:
    PATH: ./result
TASK:
  CHANNELS: [S.globular]
  CHANNEL_WEIGHTS: [1]
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
TRAIN:
  ENABLE: true
  OPTIMIZER: ADAMW
  BATCH_SIZE: 1
  EPOCHS: 100
  AMP: true
  NUM_WORKERS: 0
INFER:
  ENABLE: false
  THRESHOLD: 0.5
  SAVE_PROBABILITIES: false
```

```sh
sknext --config semantic_train.yaml --run_id 1 --gpu 0
```

The semantic-only runner currently supports the `ADAMW` optimizer. Training
uses weighted BCE/Dice loss and reports per-channel validation IoU at a fixed
probability threshold of 0.5. Raw data is normalized per patch consistently
between training and inference. Prepared stores are recreated beside the raw
directories as `train_semantic_patches.zarr` and `val_semantic_patches.zarr`.

For inference, copy the training YAML so model and normalization settings match,
then replace the following sections:

```yaml
TRAIN:
  ENABLE: false
  BATCH_SIZE: 1
DATA:
  PATCH_SIZE: [32, 128, 128, 1]
  INFER:
    PATH: ./dataset/volume.ome.zarr
    GT_PATH: ./semantic_prediction.ome.zarr
    CHANNEL: [0]
    OVERLAP: [0.25, 0.25, 0.25]
    BLOCK_CENTRAL_FACTOR: [1, 2, 2]
    HALO: []
INFER:
  ENABLE: true
  THRESHOLD: 0.5
  SAVE_PROBABILITIES: true
  SKIP_ZERO_PATCHES: true
  AMP: true
```

```sh
sknext --config semantic_infer.yaml --run_id 1 --gpu 0
```

Input must be an Imaris `.ims` or CZYX Zarr v2 OME-Zarr store. `HALO` is in ZYX
voxels; an empty list selects half the patch size, with at least one voxel per
axis. Only the central block is committed after predictions are blended.

`INFER.THRESHOLD` accepts a scalar or a list in `TASK.CHANNELS` order, for example
`[0.4, 0.6]` for two channels. Values must be strictly between 0 and 1. Sigmoid
probabilities are blended across patches before applying `probability >= threshold`.
Thresholds are configured, not automatically optimized. This setting applies
to the semantic-only runner; semantic channels in the instance/mixed runner
still use the fixed `probability > 0.5` rule.

Output masks are uint8 CZYX. With `SAVE_PROBABILITIES: true`, a separate
`<output-name>_prob.zarr` store contains probabilities scaled to uint16
(`stored_value / 65535` recovers the probability). A directory `GT_PATH` receives
`semantic_<run_id>.ome.zarr`; a `.zarr` path is used directly. Progress is stored
beside it as `<output-name>.progress.json`. Rerun the same command to resume.
Changed thresholds, normalization, overlap, zero-patch handling, checkpoint,
and other recorded prediction settings are rejected during resume. Older
progress files without the required metadata cannot be verified.

The semantic-only runner currently does not apply `INFER.POST_PROCESSING`
operations, even when they are present in the shared configuration.

## Image restoration and super-resolution

Set `TASK.CHANNELS: [SR]` exclusively. `SR.UPSAMPLE` accepts `1`, `2`, or `4`:
1× keeps spatial dimensions, while 2×/4× enlarge Y and X and preserve Z.
The model predicts continuous intensities rather than segmentation masks.

Prepare matched raw/GT TIFFs:

```text
raw/sample_0001.tif
gt/sample_0001.SR.tif
```

Both use ZYX or CZYX axes and the same channel count. For scale `s`, GT should
have spatial shape `(Z, Y*s, X*s)`. `SR.RESCALE_GT: true` permits XY resampling
of registered GT; it does not register images or permit a different Z size.

Example `sr_train.yaml`:

```yaml
PATHS:
  RESULT_DIR:
    PATH: ./result
TASK:
  CHANNELS: [SR]
SR:
  UPSAMPLE: 2
  RESCALE_GT: false
  RAW_RANGE: []
  GT_RANGE: []
  OUTPUT_CLIP: true
DATA:
  PATCH_SIZE: [32, 128, 128, 1]
  TRAIN:
    PATH: ./dataset/train/raw
    GT_PATH: ./dataset/train/gt
  VAL:
    PATH: ./dataset/val/raw
    GT_PATH: ./dataset/val/gt
  INFER:
    PATH: ./dataset/infer/raw
    CHANNEL: [0]
    PADDING: [4, 16, 16]
    BLOCK_FACTOR: [4, 4, 4]
    BLOCK_CENTRAL_FACTOR: [3, 3, 3]
MODEL:
  LOAD_CHECKPOINT: false
AUGMENTOR:
  ENABLE: true
  HFLIP: true
  VFLIP: true
TRAIN:
  ENABLE: true
  BATCH_SIZE: 1
  EPOCHS: 100
INFER:
  ENABLE: false
```

```sh
sknext --config sr_train.yaml --run_id 2 --gpu 0
```

Floating-point TIFFs are not necessarily normalized: a public dataset may store
raw intensities or processed values above 1 or below 0. With `RAW_RANGE: []`
and `GT_RANGE: []`, SR normalizes **each complete raw image and each complete
GT image independently**, for both uint16 and float TIFFs:

```text
normalized_image = (image - image.min()) / (image.max() - image.min())
```

Constant images become zero. A uint16 image spanning `[0, 65535]` is divided by
65535, while a different float image spanning `[0, 1.39344]` is divided by
1.39344. Extrema are never shared between files or between training and validation.
Raw and GT images also use independent bounds. Scaling happens before cropping;
patches from the same image keep a consistent intensity mapping.

Per-image ranges are printed and recorded in the patch store metadata.
`sr_intensity_ranges.json` and checkpoints record the normalization policy and
any explicit overrides, rather than fitted dataset-wide bounds. During inference,
each source volume is scanned in bounded blocks to obtain its own range, which
is then reused for every inference patch. This adds an initial read pass but
avoids seams caused by fitting a different range to every block.

Explicit `RAW_RANGE`/`GT_RANGE: [min, max]` settings still override automatic
scaling. Values outside those explicit bounds, NaN, and Inf raise errors.
SR uses these settings rather than segmentation `DATA.NORMALIZATION`. Outputs
remain in the normalized GT domain; absolute GT intensity units are not
automatically reconstructed after independent per-image scaling.

Checkpoints made before the `per_image_minmax_v1` policy are rejected to prevent
silently restoring the previous dataset-wide calibration. Start a fresh run with
`TRAIN.RESUME: false` and `MODEL.LOAD_CHECKPOINT: false` when switching policies.

To infer, keep the same model/SR settings and result root, set
`TRAIN.ENABLE: false` and `INFER.ENABLE: true`, save as `sr_infer.yaml`, then run:

```sh
sknext --config sr_infer.yaml --run_id 2 --gpu 0
```

SR accepts a TIFF file, a directory of TIFFs, an Imaris `.ims`, or a CZYX Zarr v2
OME-Zarr store. Patch padding supplies context; block margins must contain the
entire patch halo. Geometry is validated before execution.

Outputs are float32 CZYX stores under the run's result directory:
`<source-stem>_SR_<scale>x.ome.zarr`, with adjacent `.progress.json` files.
**SR does not use `DATA.INFER.GT_PATH` as its output destination.**
`SR.INFER.RESUME` defaults to true; `SR.INFER.OVERWRITE: true` explicitly replaces
existing SR output. Keep source stems distinct when processing a TIFF directory.
Set `SR.INFER.VOXEL_SIZE: [z, y, x]` for inputs without usable calibration;
otherwise the reader's voxel size is used when available. XY output spacing
is divided by the upscale factor.

SR training reports loss and validation PSNR. Its loss combines configured L1,
MSE, and spatial-gradient terms. IoU does not apply to continuous intensity
predictions. `TRAIN.RESUME: true` restores the SR model/optimizer/scheduler state
from its latest checkpoint, but currently does not restore RNG states or the
early-stopping counter.

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
The log records a manifest covering checkpoint content, source metadata,
skeleton file identities, model/task settings, preprocessing, output paths,
and block/patch geometry. Incompatible manifests are rejected. Source identity
checks do not hash every chunk of large Zarr inputs. For a fresh inference over
the same input and run ID, archive the old progress file and use fresh output stores.

### SWC-free instance inference and probability output

Set `INFER.SKELETON.ENABLE: false` to process all spatial blocks without SWCs.
Provide valid watershed seed, topographic, and growth-mask channels in
`TASK.WATERSHED`. Local watershed labels are assigned global uint32 IDs and
matched against previously committed neighboring blocks using the inference
halo. `INFER.STITCH_MIN_OVERLAP_VOXELS` and `STITCH_MIN_FRACTION` control ID reuse;
this is local matching rather than guaranteed neuron tracing.

In the instance/mixed runner, `INFER.OUTPUT_MODE` selects `uint16`, `float32`,
or `both`. The historical `uint16` option produces uint32 instance labels in
SWC-free mode. Float32 output contains blended sigmoid probabilities, one
channel per model output, and skips watershed when used alone. Set
`DATA.INFER.RAW_OUTPUT_PATH` to customize its destination. With skeleton mode
enabled, `INFER.RAW_ALL_BLOCKS: true` allows probability prediction outside
SWC-selected blocks. Semantic-only and SR runners use their separate output
settings described above.

## Training progress, memory, and logs

Semantic-only and SR training print **20 batch-loss updates per epoch**, or
one per batch when there are fewer than 20 batches. The last batch is always
reported. Each line includes epoch, batch number, current loss, running mean
loss, and learning rate. Semantic-only/SR console messages and GPU memory
reports start with local time in `[YYYY.MM.DD||HH:MM:SS]` format, matching
`networkflow.py`. On a server, this uses that server's local clock. For example:

```text
[YYYY.MM.DD||HH:MM:SS] [SEMANTIC] [EPOCH0001/0100] [TRAIN] [BATCH000005/000100] loss: ..., mean_loss: ..., lr: ...
[YYYY.MM.DD||HH:MM:SS] [SR] [EPOCH0001/0100] [TRAIN] [BATCH000005/000100] loss: ..., mean_loss: ..., lr: ...
[YYYY.MM.DD||HH:MM:SS] [BLOCK0000001/0000100] [INFER] results saved, core=...
```

The TIFF preparation progress bars count source volumes, not training batches
or epochs. CPU augmentation and Zarr reads occur between GPU steps; low GPU
utilization may indicate input preparation rather than a stopped process.

Semantic-only and SR workflows print GPU memory after model loading, after
the first training batch of each epoch, after validation, and at selected
inference block intervals. Allocated/reserved/peak values describe the current
PyTorch process; device used/free/total values include device-wide usage.
Values are in MiB. Peak values are measured since the last CUDA peak reset,
not necessarily just the most recent batch. CPU execution skips CUDA queries.

| Training output | Instance/mixed | Semantic-only | SR |
| --- | --- | --- | --- |
| Console losses | Training and validation | Training and validation | Training and validation |
| Console quality metric | Per-channel training/validation IoU | Per-channel validation IoU | Validation PSNR in dB |
| TensorBoard | `log/`: loss, IoU, LR | Not currently written | `sr_tensorboard/`: loss, PSNR, LR |
| Separate JSONL metric history | No | No | `sr_history.jsonl` |
| PNG training curves | `chart/`, every five epochs | Not currently written | Not currently written |
| Best/latest checkpoints | Yes | Yes | Yes |

Directories in this table are relative to
`<PATHS.RESULT_DIR.PATH>/results/<run_id>/`. GUI launches additionally save
timestamped configuration snapshots and console `.log` files. CLI launches do
not automatically save their console output. On Linux, capture it with:

```sh
mkdir -p logs
sknext --config semantic_train.yaml --run_id 1 --gpu 0 > logs/semantic_train_1.log 2>&1
```

In PowerShell:

```powershell
sknext --config semantic_train.yaml --run_id 1 --gpu 0 2>&1 | Tee-Object -FilePath semantic_train_1.log
```

### Automatic mixed precision

`TRAIN.AMP: true` enables float16 autocasting and GradScaler for semantic-only
training; `INFER.AMP` controls its inference autocasting. AMP can reduce GPU
memory and improve throughput. Set `TRAIN.AMP: false` to diagnose float16
overflow, allowing for increased memory consumption. Instance/mixed and SR
training currently run in float32 and do not use these AMP flags.

The semantic training loop currently advances its scheduler even when GradScaler
skips an optimizer update after detecting nonfinite gradients. This can produce
the `lr_scheduler.step() before optimizer.step()` warning. It does not itself
block execution, but the schedule can advance without a weight update. Disabling
AMP avoids that skipped-update mechanism; the warning should not be treated as
proof of a stalled process.

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
from the result root and run ID. Set exactly one of `TRAIN.ENABLE` and
`INFER.ENABLE`; the launcher rejects configurations where both or neither are enabled.

| Setting | Meaning |
| --- | --- |
| `DATA.PATCH_SIZE` | Model patch dimensions in **Z, Y, X, C** order. |
| `TASK.CHANNELS` | Selects the runner and output channel order; `[SR]` must be exclusive. |
| `TASK.CHANNEL_WEIGHTS` | One loss weight per actual output channel, including expanded affinity channels. |
| `TASK.CHANNELS_EXTRA_OPTS` | A list containing one dictionary of per-target options. |
| `MODEL.FEATURE_MAPS` | Feature widths at successive model levels; default `[16, 32, 64, 128, 256]`. |
| `MODEL.Z_DOWN`, `MODEL.YX_DOWN` | Spatial downsampling at each level transition. |
| `MODEL.ISOTROPY`, `MODEL.CONVNEXT_LAYERS` | Per-level geometry/block settings; isotropy accepts a boolean or a per-level boolean list. |
| `MODEL.LOAD_CHECKPOINT` | Initialize model weights; mutually exclusive with `TRAIN.RESUME`. |
| `MODEL.CHECKPOINT_INPUT_PATH` | Read a checkpoint from another location without redirecting normal checkpoint writes. |
| `TRAIN.RESUME` | Continue from a latest training checkpoint; state guarantees differ by workflow. |
| `TRAIN.BATCH_SIZE` | Training batch size and segmentation inference patch batch size; SR inference currently predicts one patch at a time. |
| `TRAIN.NUM_WORKERS` | Semantic-only training DataLoader workers; default 0. |
| `TRAIN.AMP`, `INFER.AMP` | Semantic-only mixed precision; default true on CUDA. |
| `TRAIN.LR`, `EPOCHS`, `PATIENCE` | Learning rate, maximum epochs, and early-stopping patience. |
| `TRAIN.LR_SCHEDULER.NAME` | `warmupcosine` or `one_cycle`. |
| `DATA.NORMALIZATION` | Min/max or mean/std normalization, with optional percentile clipping. |
| `DATA.TRAIN.FILTER` | Optional filters on original raw mean or foreground label fraction; validation is unfiltered. |
| `DATA.*.OVERLAP` | Per-axis overlap fractions in `[0, 1)`, in ZYX order. |
| `DATA.*.PADDING` | Per-axis padding in voxels, in ZYX order. |
| `DATA.INFER.CHANNEL` | List of input channel indices; its length must match patch C. |
| `DATA.INFER.BLOCK_FACTOR` | Expanded inference block size as a multiple of spatial patch size. |
| `DATA.INFER.BLOCK_CENTRAL_FACTOR` | Written central block size; cannot exceed the expanded block. |
| `DATA.INFER.HALO` | Semantic-only block context in ZYX voxels; `[]` selects half the patch size. |
| `TASK.WATERSHED` | Seed/growth channels and thresholds; provide one threshold per selected channel or an empty list for automatic thresholds. |
| `INFER.THRESHOLD` | Semantic-only scalar or per-channel threshold; default 0.5. |
| `INFER.SAVE_PROBABILITIES` | Semantic-only uint16 probability export in a separate store. |
| `INFER.OUTPUT_MODE` | Instance/mixed labels, float32 probabilities, or both. |
| `INFER.POST_PROCESSING` | Instance/mixed operations: `fill_holes`, `remove_small`, `remove_large`; not applied by semantic-only or SR runners. |
| `SR.UPSAMPLE` | XY restoration scale: 1, 2, or 4. |
| `SR.RAW_RANGE`, `SR.GT_RANGE` | `[]`: independent min/max scaling per complete raw/GT image for every dtype; explicit `[min,max]` overrides it. |
| `SR.INFER.RESUME`, `SR.INFER.OVERWRITE` | Resume SR output or explicitly replace it. |

Supported task channels:

| Channel | Generated target |
| --- | --- |
| `F` | Foreground from nonzero instance labels. |
| `P` | Per-instance skeleton or centroid; skeleton mode can be `full` or `main`. |
| `C` | Instance contours. |
| `A` | Same-instance affinities for configured Z/Y/X offsets. |
| `S.<name>` | Independent binary semantic target from the corresponding label files. |
| `SR` | Continuous restoration target from matching `.SR.tif`/`.SR.tiff` files. |

Patch dimensions must be compatible with the model's stem and cumulative
downsampling. Increasing patch size, feature widths, batch size, or block
factors increases memory demand; inference also allocates large block buffers
in host RAM. The examples are not a guarantee of fitting a particular GPU.

## Outputs

For instance/mixed training with `PATHS.RESULT_DIR.PATH: ./result` and `--run_id 0`:

```text
result/results/0/
├── checkpoints/
│   ├── checkpoint_00.pth
│   └── checkpoint_00_latest.pth
├── log/                     # TensorBoard events
├── chart/                   # Training curves exported every five epochs
├── gui-<timestamp>.yaml     # Configuration snapshot for GUI launches
└── gui-<timestamp>.log      # Console output for GUI launches
```

Semantic-only runs also save best/latest checkpoints, but do not create the
instance/mixed TensorBoard and chart outputs. SR runs additionally contain:

```text
result/results/2/
├── checkpoints/
│   ├── checkpoint_02.pth
│   └── checkpoint_02_latest.pth
├── train_sr_patches.zarr/
├── val_sr_patches.zarr/
├── sr_tensorboard/
├── sr_history.jsonl
├── sr_intensity_ranges.json
├── <source-stem>_SR_2x.ome.zarr/
└── <source-stem>_SR_2x.progress.json
```

Segmentation output is stored separately at `DATA.INFER.GT_PATH`; SR prediction
stores are placed under the run directory. Training patch stores and progress
logs follow the workflow-specific locations described above.

View training metrics with:

```sh
tensorboard --logdir ./result/results/0/log
tensorboard --logdir ./result/results/2/sr_tensorboard
```

All prediction stores use **CZYX** layout. Skeleton-guided instance labels use
uint16 IDs (at most 65,535 SWCs); SWC-free instance labels use uint32 IDs;
semantic-only masks use uint8; SR predictions and instance/mixed raw probability
outputs use float32. Semantic-only optional probability output uses scaled uint16.

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

- Adapt the placeholder paths in `sknext/example/` before use. Globular and
  reticular examples select semantic-only segmentation; super-resolution
  examples select SR; the F/P/soma/spine/bouton examples select mixed segmentation.
- The schema uses flat `TASK.CHANNELS` and `TASK.CHANNELS_EXTRA_OPTS` keys.
  Legacy `TASK.TYPE`/`TASK.INSTANCE` configurations must be converted.
- SWCs are required only when the instance/mixed inference path uses them.
  Semantic-only and SR inference do not require skeleton data.
- Input axes are determined by the implemented readers. Setting
  `DATA.INFER.INPUT_AXES_ORDER` does not transpose arbitrary input layouts.
- The instance/mixed writer produces separate label/probability OME-Zarr
  outputs. `DATA.INFER.RESULT_INTO_FILE` is not an active output-mode switch.
- The CLI accepts a comma-separated GPU selector, but the workflow uses
  `cuda:0` among visible devices; it does not launch distributed or multi-GPU training.
- Semantic-only training currently has no TensorBoard/PNG metric logging;
  redirect console output when a persistent training trace is required.
- SR resume restores optimizer/scheduler state but not RNG or patience history.
- The shared defaults currently allow partial checkpoint loading through
  `MODEL.CHECKPOINT_ALLOW_PARTIAL`. Set it to `false` when a complete tensor
  match is required; partial loads retain initialization for omitted tensors.
- Files in `sknext/test/` include manual utilities with local paths and optional
  dependencies. Inspect them before execution.

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| `sknext` is not found | Activate the intended environment and run `python -m pip install -e .` from the project root, or use `python -m sknext`. |
| `Unknown config key: INFER.THRESHOLD` | Update the code on the machine running the job. The current shared schema accepts semantic options and the CLI selects the semantic runner automatically. |
| `torchvision::nms` is missing during import | Verify matching torch/torchvision releases and compute builds. A CPU torch installation with CUDA torchvision is incompatible; reinstall the pair using the official PyTorch selector. |
| OpenMP reports both `libomp` and `libiomp` | For the affected MKL/Conda environment, use the sequential MKL setting in Installation. Do not bypass runtime checks with `KMP_DUPLICATE_LIB_OK`. |
| CUDA is unavailable | Check the active interpreter, NVIDIA driver, and `torch.version.cuda`; CPU-only PyTorch cannot run the unified CUDA launcher. |
| No progress immediately after TIFF conversion | Preparation bars count TIFFs. Model creation, patch loading, and CPU augmentation follow; loss lines appear at the batch reporting points. Check activity with `nvidia-smi` on the compute node. |
| Scheduler-before-optimizer warning with AMP | See the AMP section; GradScaler may have skipped an update. This is a known scheduler-handling limitation, not a blocking warning. |
| Resume rejects configuration or output metadata | Keep the original settings, weights, inputs, and stores; use a fresh run/output for a changed experiment. |

When moving to another computer, transfer the updated source, YAML, required
data and compatible checkpoints. Recreate the Python environment and install a
matching PyTorch pair for that machine rather than copying CUDA-specific wheels
or absolute dataset paths from the previous installation.

## Project layout

```text
main.py                  # CLI / GUI launcher
sknext/
├── __init__.py          # Command-line argument handling
├── _sknext.py           # Configuration and CUDA setup
├── gui.py               # Desktop editor and worker management
├── config/              # Shared configuration defaults
├── example/             # Instance/mixed, semantic-only, and SR YAML examples
├── data/                # I/O, patches, augmentation, labels, processing
├── model/               # U-NeXt V2, blocks, heads, loss, schedulers
├── run/                 # Instance/mixed, semantic-only, SR, and component export
├── skeleton/            # SWC geometry, rasterization, queries, visualization
├── utils/               # Checkpoints, reproducibility, memory and progress reporting
└── test/                # Manual data-processing utilities
tests/                   # CLI dispatch, GUI, runtime regressions and desktop smoke check
```

Run the automated tests in an environment with the project dependencies:

```sh
python -m unittest discover -s tests -v
```

On a machine with Tkinter and a display, the separate desktop smoke check is:

```sh
python tests/gui_smoke.py
```

These checks exercise CLI routing, configuration handling, augmentation,
resume guards, and memory reporting. Some tests use substitutes for GPU/model
components; they do not constitute an end-to-end GPU training benchmark.
Evaluate segmentation and restoration quality separately on your validation data.
