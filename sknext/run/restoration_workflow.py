"""SkNeXt single-head fluorescence image restoration: 1x/2x/4x XY SR.

Train: paired raw ``sample.tif`` and GT ``sample.SR.tif`` in DATA.* folders.
Infer: 3-D/4-D TIFF, directory of TIFFs, OME-Zarr, or Imaris IMS.
Prediction: float32 CZYX OME-Zarr, intensities in the normalized GT domain.

SR is intentionally incompatible with all segmentation target channels.
Training and inference normalize each source image independently before tiling.
Explicit intensity bounds override automatic per-image min/max scaling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

import zarr
from zarr.codecs import BloscCodec
from sknext.data.imageIO import get_tif_path_in_folder, read_one_3D_tif, patch_coordinates_iter
from sknext.data.augmentor import PairedAugmentor
from sknext.data.datasetIO import IMSReader, ZarrIOManager, detect_path_type
from sknext.model.unext_v2 import U_NeXt_V2
from sknext.utils.parallel import model_state_dict
from sknext.utils.execution import (prepare_once, make_loader, set_epoch, primary,
    reduce_sums, validation_model, NullWriter, training_rng_states)
from sknext.model.scheduler import build_warmup_cosine_scheduler, build_onecycle_scheduler
from sknext.utils.checkpoint import read_checkpoint, load_model_weights
from sknext.utils.model_info import print_cuda_memory, print_model_parameters
from sknext.utils.progress import loss_log_steps, log_message
from sknext.utils.training_charts import TrainingLossChart


def sr_infer_geometry(cfg) -> dict[str, tuple[int, int, int]]:
    """Validate and derive SR patch and central-block geometry (all in input ZYX).

    DATA.INFER.PADDING is a per-side *patch* halo, not a second block halo.
    The block margin is derived from the full/central block sizes, identical
    to SkNeXt's segmentation Base_Workflow calculation (rounded down for odd
    differences). The block margin must contain the entire patch halo so no
    interior patch is fabricated at a block boundary.
    """
    patch = tuple(int(v) for v in cfg.DATA.PATCH_SIZE[:3])

    def integer_zyx(values, name, minimum):
        values = list(values)
        if len(values) != 3 or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
                                   or v < minimum for v in values):
            raise ValueError(f"{name} must contain three integers >= {minimum}, in ZYX order")
        return tuple(int(v) for v in values)

    padding = integer_zyx(cfg.DATA.INFER.PADDING, "DATA.INFER.PADDING", 0)
    block_factor = integer_zyx(cfg.DATA.INFER.BLOCK_FACTOR, "DATA.INFER.BLOCK_FACTOR", 1)
    central_factor = integer_zyx(cfg.DATA.INFER.BLOCK_CENTRAL_FACTOR,
                                 "DATA.INFER.BLOCK_CENTRAL_FACTOR", 1)
    if any(2 * pad >= p for pad, p in zip(padding, patch)):
        raise ValueError("DATA.INFER.PADDING must be smaller than half DATA.PATCH_SIZE along each axis")
    if any(c > b for c, b in zip(central_factor, block_factor)):
        raise ValueError("DATA.INFER.BLOCK_CENTRAL_FACTOR must not exceed BLOCK_FACTOR")
    block_size = tuple(int(v * p) for v, p in zip(block_factor, patch))
    central_block = tuple(int(v * p) for v, p in zip(central_factor, patch))
    block_margin = tuple((b - c) // 2 for b, c in zip(block_size, central_block))
    if any(m < pad for m, pad in zip(block_margin, padding)):
        raise ValueError("Block margin (BLOCK_FACTOR - BLOCK_CENTRAL_FACTOR) * "
                         "PATCH_SIZE / 2 must be >= DATA.INFER.PADDING on every axis; "
                         "increase BLOCK_FACTOR or reduce padding")
    return {"patch": patch, "padding": padding,
            "patch_central": tuple(p - 2 * pad for p, pad in zip(patch, padding)),
            "block_size": block_size, "central_block": central_block,
            "block_margin": block_margin}


def validate_sr(cfg) -> int:
    names = list(cfg.TASK.CHANNELS)
    if names != ["SR"]:
        raise ValueError('SR restoration requires TASK.CHANNELS: ["SR"] exclusively')
    scale = int(cfg.SR.UPSAMPLE)
    if scale not in (1, 2, 4) or scale != cfg.SR.UPSAMPLE:
        raise ValueError("SR.UPSAMPLE must be 1, 2, or 4")
    patch = tuple(map(int, cfg.DATA.PATCH_SIZE))
    if len(patch) != 4 or min(patch) < 1:
        raise ValueError("DATA.PATCH_SIZE must have positive ZYXC entries")
    zdown = list(cfg.MODEL.Z_DOWN)
    yxdown = list(cfg.MODEL.YX_DOWN)
    nstage = len(cfg.MODEL.FEATURE_MAPS) - 1
    if not (len(zdown) == len(yxdown) == nstage and
            len(cfg.MODEL.CONVNEXT_LAYERS) == nstage + 1):
        raise ValueError("MODEL downsampling/stage lengths are inconsistent")
    # Stem downsamples Z by max(Z_DOWN[0], STEM_SIZE) when Z_DOWN[0] > stem.
    stem = int(cfg.MODEL.CONVNEXT_STEM_K_SIZE)
    if stem < 1 or any(d < 1 for d in zdown + yxdown):
        raise ValueError("Model downsampling factors must be positive")
    divisible = (max(stem, zdown[0]) * math.prod(zdown),
                 stem * math.prod(yxdown), stem * math.prod(yxdown))
    # The actual stem Z stride is stem*max(int(zdown[0]/stem),1).
    divisible = (stem * max(int(zdown[0] / stem), 1) * math.prod(zdown),
                 divisible[1], divisible[2])
    if any(p % d for p, d in zip(patch[:3], divisible)):
        raise ValueError(f"Input patch {patch[:3]} must be divisible by {divisible}")
    sr_infer_geometry(cfg)  # One authoritative geometry for validation and inference.
    if len(cfg.SR.INFER.VOXEL_SIZE) not in (0, 3):
        raise ValueError("SR.INFER.VOXEL_SIZE must be [] or [z,y,x]")
    if len(cfg.SR.INFER.VOXEL_SIZE) == 3 and any(not np.isfinite(float(v)) or float(v) <= 0
                                                      for v in cfg.SR.INFER.VOXEL_SIZE):
        raise ValueError("SR.INFER.VOXEL_SIZE must be finite and positive")
    for key in ("RAW_RANGE", "GT_RANGE"):
        value = list(getattr(cfg.SR, key))
        if value and (len(value) != 2 or not np.isfinite(value).all() or value[0] >= value[1]):
            raise ValueError(f"SR.{key} must be [] or [min,max]")
    for name in ("LOSS_L1_WEIGHT", "LOSS_MSE_WEIGHT", "LOSS_EDGE_WEIGHT",
                 "AUGMENT_NOISE_STD", "AUGMENT_BLUR_SIGMA"):
        if float(getattr(cfg.SR, name)) < 0:
            raise ValueError(f"SR.{name} cannot be negative")
    if not any(float(getattr(cfg.SR, name)) > 0 for name in
               ("LOSS_L1_WEIGHT", "LOSS_MSE_WEIGHT", "LOSS_EDGE_WEIGHT")):
        raise ValueError("At least one SR loss weight must be positive")
    return scale


def image_intensity_range(img: np.ndarray, bounds: Sequence[float], name: str) -> tuple[float, float]:
    """Resolve explicit bounds or the extrema of one complete image."""
    arr = np.asarray(img)
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(f"{name} must have real numeric dtype; got {arr.dtype}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    lower, upper = float(arr.min()), float(arr.max())
    if not np.isfinite([lower, upper]).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if len(bounds) == 2:
        lo, hi = map(float, bounds)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"{name} bounds must be finite [min,max] with min < max")
        if lower < lo - 1e-4 * max(1.0, abs(lo)) or upper > hi + 1e-4 * max(1.0, abs(hi)):
            raise ValueError(f"{name} value range [{lower:g}, {upper:g}] is outside [{lo:g}, {hi:g}]")
        return lo, hi
    if bounds:
        raise ValueError(f"{name} bounds must be [] or [min,max]")
    return lower, upper


def normalize_intensity(img: np.ndarray, bounds: Sequence[float], name: str) -> np.ndarray:
    """Scale one whole image to [0,1], independently of other images and dtypes."""
    lo, hi = image_intensity_range(img, bounds, name)
    data = np.asarray(img, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{name} cannot be represented as finite float32 values")
    if lo == hi:
        return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0., 1.).astype(np.float32, copy=False)


def reader_intensity_range(reader, channels, block_size, bounds) -> tuple[float, float]:
    """Scan one inference volume in bounded blocks, then reuse its range for all patches."""
    if bounds:
        return tuple(map(float, bounds))
    lower, upper = float("inf"), -float("inf")
    shape = reader.shape[-3:]
    log_message("[SR] scanning this input image for its own intensity range")
    for z in range(0, shape[0], block_size[0]):
        for y in range(0, shape[1], block_size[1]):
            for x in range(0, shape[2], block_size[2]):
                for channel in channels:
                    block = np.asarray(reader[channel, z:min(z + block_size[0], shape[0]),
                                               y:min(y + block_size[1], shape[1]),
                                               x:min(x + block_size[2], shape[2])])
                    lo, hi = image_intensity_range(block, [], "inference raw")
                    lower, upper = min(lower, lo), max(upper, hi)
    if not np.isfinite([lower, upper]).all():
        raise ValueError("Inference image has no finite intensity range")
    log_message(f"[SR] this input image range: [{lower:g}, {upper:g}]")
    return lower, upper


def pair_sr_tiffs(raw_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match raw sample.tif to GT sample.SR.tif, detecting duplicates and gaps."""
    raw_files = get_tif_path_in_folder(raw_dir)
    gt_files = get_tif_path_in_folder(gt_dir)
    raws = {p.stem: p for p in raw_files}
    gts: dict[str, Path] = {}
    for path in gt_files:
        if not path.stem.endswith(".SR"):
            continue
        stem = path.stem[:-3]
        if stem in gts:
            raise ValueError(f"Duplicate GT for {stem}: {gts[stem]}, {path}")
        gts[stem] = path
    if not raws:
        raise ValueError(f"No raw TIFF files in {raw_dir}")
    if set(raws) != set(gts):
        raise ValueError(f"Raw/GT pairing mismatch: missing GT={sorted(set(raws)-set(gts))}, "
                         f"GT without raw={sorted(set(gts)-set(raws))}")
    return [(raws[stem], gts[stem]) for stem in sorted(raws)]


def align_sr_gt(raw: np.ndarray, gt: np.ndarray, scale: int,
                allow_rescale: bool) -> np.ndarray:
    """Align GT to the expected high-res shape; assumes registered identical FOV."""
    if raw.ndim != 4 or gt.ndim != 4:
        raise ValueError("Both raw and GT must be in CZYX order")
    if raw.shape[0] != gt.shape[0]:
        raise ValueError(f"Channel count differs: raw={raw.shape[0]}, GT={gt.shape[0]}")
    wanted = (raw.shape[0], raw.shape[1], raw.shape[2] * scale, raw.shape[3] * scale)
    if gt.shape == wanted:
        return np.asarray(gt, dtype=np.float32)
    if not allow_rescale:
        raise ValueError(f"GT shape {gt.shape} does not match expected {wanted}")
    if raw.shape[1] != gt.shape[1]:
        raise ValueError(f"GT Z={gt.shape[1]} differs from raw Z={raw.shape[1]}; "
                         "axial registration/resampling must be done explicitly")
    warnings.warn(f"Resampling GT XY from {gt.shape} to {wanted}; "
                  "verify raw and GT share the same physical field of view.", stacklevel=2)
    # Channel and Z are not mixed; anti-alias during downsampling only.
    return np.stack([
        np.stack([resize(gt[c, z], wanted[-2:], order=1, mode="reflect",
                         anti_aliasing=(gt.shape[-2] > wanted[-2] or
                                        gt.shape[-1] > wanted[-1]),
                         preserve_range=True).astype(np.float32)
                  for z in range(gt.shape[1])], axis=0)
        for c in range(gt.shape[0])], axis=0)


def crop_padded(img: np.ndarray, start: Sequence[int], patch: Sequence[int]) -> np.ndarray:
    """Read a fixed CZYX patch, reflecting beyond image boundaries safely."""
    shape = np.asarray(img.shape[-3:], dtype=int)
    start = np.asarray(start, dtype=int)
    stop = start + np.asarray(patch, dtype=int)
    low = np.maximum(start, 0)
    high = np.minimum(stop, shape)
    slices = (slice(None),) + tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    cut = np.asarray(img[slices])
    pad = [(0, 0)] + [(int(a), int(b)) for a, b in
                      zip(low - start, stop - high)]
    if any(a or b for a, b in pad):
        cut = np.pad(cut, pad, mode="reflect" if min(cut.shape[-3:]) > 1 else "edge")
    return np.ascontiguousarray(cut)


def _overlap_voxels(fracs, patch: Sequence[int]) -> tuple[int, int, int]:
    frac = np.asarray(fracs, dtype=float)
    if frac.shape != (3,) or not np.isfinite(frac).all() or np.any(frac < 0) or np.any(frac >= 1):
        raise ValueError("DATA.*.OVERLAP must be ZYX fractions in [0,1)")
    return tuple(np.rint(frac * np.asarray(patch, dtype=int)).astype(int))


def make_sr_patch_store(cfg, part: str, output: Path) -> Path:
    """Produce paired Zarr v3 NCZYX arrays with distinct raw/GT spatial shapes."""
    scale = validate_sr(cfg)
    patch = tuple(int(v) for v in cfg.DATA.PATCH_SIZE[:3])
    channels = int(cfg.DATA.PATCH_SIZE[-1])
    folder = getattr(cfg.DATA, part)
    pairs = pair_sr_tiffs(folder.PATH, folder.GT_PATH)
    coords_overlap = _overlap_voxels(folder.OVERLAP, patch)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    root = zarr.open_group(str(output), mode="w", zarr_format=3)
    hr_patch = (patch[0], patch[1] * scale, patch[2] * scale)
    codec = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    xs = root.create_array("raw", shape=(0, channels, *patch),
                           chunks=(1, 1, min(16, patch[0]),
                                   min(256, patch[1]), min(256, patch[2])),
                           compressors=[codec], dtype="float32")
    ys = root.create_array("label", shape=(0, channels, *hr_patch),
                           chunks=(1, 1, min(16, hr_patch[0]),
                                   min(256, hr_patch[1]), min(256, hr_patch[2])),
                           compressors=[codec], dtype="float32")
    count = 0
    image_ranges = []
    for raw_path, gt_path in pairs:
        raw = read_one_3D_tif(raw_path, "CZYX")
        gt = read_one_3D_tif(gt_path, "CZYX")
        if raw.shape[0] != channels:
            raise ValueError(f"{raw_path.name} contains {raw.shape[0]} channels, expected {channels}")
        raw_bounds = image_intensity_range(raw, cfg.SR.RAW_RANGE, "raw")
        gt_bounds = image_intensity_range(gt, cfg.SR.GT_RANGE, "GT")
        image_ranges.append({"raw": str(raw_path), "gt": str(gt_path),
                             "raw_range": list(raw_bounds), "gt_range": list(gt_bounds)})
        log_message(f"[SR] {part} {raw_path.name}: raw range {raw_bounds}; GT range {gt_bounds}")
        raw = normalize_intensity(raw, cfg.SR.RAW_RANGE, "raw")
        gt = normalize_intensity(gt, cfg.SR.GT_RANGE, "GT")
        gt = align_sr_gt(raw, gt, scale, bool(cfg.SR.RESCALE_GT))
        coordinates = list(patch_coordinates_iter(raw.shape[-3:], patch,
                                                  coords_overlap, (0, 0, 0)))
        if not coordinates:
            raise ValueError(f"Cannot extract any patch from {raw_path.name}")
        xs.resize((count + len(coordinates), channels, *patch))
        ys.resize((count + len(coordinates), channels, *hr_patch))
        for coord in coordinates:
            lo = coord[:, 0].astype(int)
            x = crop_padded(raw, lo, patch)
            y = crop_padded(gt, lo * np.asarray((1, scale, scale)), hr_patch)
            xs[count] = x
            ys[count] = y
            count += 1
    root.attrs["sknext"] = {"task": "SR", "scale_zyx": [1, scale, scale],
                            "patch_zyx": list(patch), "dtype": "float32",
                            "normalization": "per_image_minmax_v1",
                            "image_ranges": image_ranges,
                            "raw_range": list(cfg.SR.RAW_RANGE),
                            "gt_range": list(cfg.SR.GT_RANGE)}
    if count == 0:
        raise ValueError(f"No SR patches written to {output}")
    log_message(f"[SR] {part}: wrote {count} paired patches to {output}", flush=True)
    return output


class SRPatchDataset(Dataset):
    """Read aligned LR/HR patches; augmentation uses the shared AUGMENTOR config."""

    def __init__(self, path: Path, train: bool = False, cfg=None,
                 noise_std: float = 0.0, blur_sigma: float = 0.0):
        self.path = Path(path)
        self.root = None  # lazy reopen in worker processes
        self.train = bool(train)
        if self.train and cfg is None:
            raise ValueError("SRPatchDataset(training) requires cfg for the shared AUGMENTOR")
        self.augmentor = (PairedAugmentor(cfg, task="sr", scale=int(cfg.SR.UPSAMPLE),
                                          legacy_noise_std=noise_std,
                                          legacy_blur_sigma=blur_sigma)
                          if self.train else None)
        self._length = int(zarr.open_group(str(path), mode="r")["raw"].shape[0])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self.root is None:
            self.root = zarr.open_group(str(self.path), mode="r")
        x = np.asarray(self.root["raw"][int(idx)], dtype=np.float32)
        y = np.asarray(self.root["label"][int(idx)], dtype=np.float32)
        if self.augmentor is not None:
            # CZYX -> ZYXC, with scale-aware shared spatial coordinates.
            x, y = self.augmentor(np.moveaxis(x, 0, -1), np.moveaxis(y, 0, -1))
            x, y = np.moveaxis(x, -1, 0), np.moveaxis(y, -1, 0)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def sr_loss(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"SR shape mismatch: pred={tuple(pred.shape)}, gt={tuple(target.shape)}")
    # Use fp32 for quantitative fidelity even under mixed precision.
    pred, target = pred.float(), target.float()
    delta = pred - target
    loss = (float(cfg.SR.LOSS_L1_WEIGHT) * delta.abs().mean() +
            float(cfg.SR.LOSS_MSE_WEIGHT) * delta.square().mean())
    edge_weight = float(cfg.SR.LOSS_EDGE_WEIGHT)
    if edge_weight:
        for axis in (-1, -2, -3):
            if pred.shape[axis] > 1:
                loss = loss + edge_weight * (
                    torch.diff(pred, dim=axis) - torch.diff(target, dim=axis)).abs().mean() / 3
    return loss


def sr_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.float().clamp(0, 1) - target.float()).square().mean().item()
    return float(-10. * math.log10(max(mse, 1e-12)))


class _TiffReader:
    def __init__(self, path: Path):
        self.path = path
        try:
            source = tifffile.memmap(path)
        except (ValueError, TypeError, tifffile.TiffFileError):
            source = tifffile.imread(path)
        if source.ndim == 3:
            source = source[np.newaxis, ...]
        if source.ndim != 4:
            raise ValueError(f"TIFF must be ZYX or CZYX: {path}")
        self.data = source
        self.shape = tuple(source.shape)
        self.dtype = source.dtype
        self.voxel_size = None  # Non-OME TIFF has no trustworthy spatial calibration.

    def __getitem__(self, index):
        return self.data[index]

    def close(self):
        del self.data


class RestorationWorkflow:
    def __init__(self, cfg, device: torch.device, job_id: int):
        self.cfg = cfg
        self.device = torch.device(device)
        self.job_id = job_id
        self.scale = validate_sr(cfg)
        self.patch = tuple(int(i) for i in cfg.DATA.PATCH_SIZE[:3])
        self.ch = int(cfg.DATA.PATCH_SIZE[-1])
        self.workdir = Path(cfg.PATHS.RESULT_DIR.PATH_)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Path(cfg.PATHS.CHECKPOINT_DIR) / f"checkpoint_{int(job_id):02d}.pth"
        self.latest = self.checkpoint.with_name(self.checkpoint.stem + "_latest.pth")

    def build_model(self):
        cfg = self.cfg
        if str(cfg.MODEL.ARCHITECTURE).lower().replace("-", "_") not in {"unext_v2", "u_next_v2"}:
            raise ValueError("SR workflow currently supports only U_NeXt_V2")
        model = U_NeXt_V2(
            image_shape=tuple(map(int, cfg.DATA.PATCH_SIZE)),
            feature_maps=list(cfg.MODEL.FEATURE_MAPS), z_down=list(cfg.MODEL.Z_DOWN),
            yx_down=list(cfg.MODEL.YX_DOWN), cn_layers=list(cfg.MODEL.CONVNEXT_LAYERS),
            isotropy=cfg.MODEL.ISOTROPY, stem_k_size=int(cfg.MODEL.CONVNEXT_STEM_K_SIZE),
            stochastic_depth_prob=float(cfg.MODEL.CONVNEXT_SD_PROB),
            output_channels=[self.ch], output_channel_info=["SR"],
            explicit_activations=False, separated_decoders=False,
            upsampling_factor=(1, self.scale, self.scale) if self.scale > 1 else (),
            upsampling_position="post")
        from sknext.utils.parallel import parallel_model
        return parallel_model(model, self.device, cfg.TRAIN.BATCH_SIZE
                              if cfg.TRAIN.ENABLE else None)

    def _identity(self) -> str:
        cfg = self.cfg
        payload = {
            "normalization_policy": "per_image_minmax_v1",
            "scale": self.scale, "patch": self.patch, "channels": self.ch,
            "range_raw": list(cfg.SR.RAW_RANGE), "range_gt": list(cfg.SR.GT_RANGE),
            "features": list(cfg.MODEL.FEATURE_MAPS), "z_down": list(cfg.MODEL.Z_DOWN),
            "yx_down": list(cfg.MODEL.YX_DOWN), "layers": list(cfg.MODEL.CONVNEXT_LAYERS),
            "paths": [str(cfg.DATA.TRAIN.PATH), str(cfg.DATA.TRAIN.GT_PATH),
                      str(cfg.DATA.VAL.PATH), str(cfg.DATA.VAL.GT_PATH)],
            "loss": [float(cfg.SR.LOSS_L1_WEIGHT), float(cfg.SR.LOSS_MSE_WEIGHT),
                     float(cfg.SR.LOSS_EDGE_WEIGHT)],
            "batch": int(cfg.TRAIN.BATCH_SIZE), "lr": float(cfg.TRAIN.LR),
            "weight_decay": float(cfg.TRAIN.W_DECAY),
            "optimizer": str(cfg.TRAIN.OPTIMIZER),
            "scheduler": str(cfg.TRAIN.LR_SCHEDULER.NAME),
            "epochs": int(cfg.TRAIN.EPOCHS),
            "stem": int(cfg.MODEL.CONVNEXT_STEM_K_SIZE),
            "isotropy": list(cfg.MODEL.ISOTROPY) if not isinstance(cfg.MODEL.ISOTROPY, bool)
                         else cfg.MODEL.ISOTROPY,
            "sd_prob": float(cfg.MODEL.CONVNEXT_SD_PROB),
            "overlap": list(cfg.DATA.TRAIN.OVERLAP),
            "val_overlap": list(cfg.DATA.VAL.OVERLAP),
            "noise_std": float(cfg.SR.AUGMENT_NOISE_STD),
            "blur_sigma": float(cfg.SR.AUGMENT_BLUR_SIGMA),
            "augmentor": dict(cfg.AUGMENTOR),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _checkpoint_data(self, epoch, model, optimizer=None, scheduler=None, best=None):
        return {"model_state_dict": model_state_dict(model), "model_name": "U_NeXt_V2",
                "normalization_policy": "per_image_minmax_v1",
                "output_channel_names": ["SR"], "scale": self.scale,
                "input_channels": self.ch, "patch_size": list(self.patch),
                "raw_range": list(self.cfg.SR.RAW_RANGE),
                "gt_range": list(self.cfg.SR.GT_RANGE),
                "feature_maps": list(self.cfg.MODEL.FEATURE_MAPS),
                "z_down": list(self.cfg.MODEL.Z_DOWN),
                "yx_down": list(self.cfg.MODEL.YX_DOWN),
                "epoch": epoch, "best_val_loss": best, "identity": self._identity(),
                "rng_by_rank": getattr(self, "_rng_states", []),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scaler_state_dict": self.scaler.state_dict() if hasattr(self, "scaler") else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None}

    @staticmethod
    def _save(data, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(data, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _load(self, model, path: Path):
        state = read_checkpoint(path)
        self._validate_intensity_policy(state)
        for key, current in (("scale", self.scale), ("input_channels", self.ch),
                             ("patch_size", list(self.patch)),
                             ("raw_range", list(self.cfg.SR.RAW_RANGE)),
                             ("gt_range", list(self.cfg.SR.GT_RANGE)),
                             ("feature_maps", list(self.cfg.MODEL.FEATURE_MAPS)),
                             ("z_down", list(self.cfg.MODEL.Z_DOWN)),
                             ("yx_down", list(self.cfg.MODEL.YX_DOWN))):
            if state.get(key) != current:
                raise ValueError(f"SR checkpoint {key}: stored={state.get(key)}, requested={current}")
        load_model_weights(model, state, expected_architecture="U_NeXt_V2",
                           expected_channels=["SR"], allow_partial=False)
        return state

    def _validate_intensity_policy(self, state):
        """Do not silently reuse dataset-wide scaling with per-image normalization."""
        if state.get("normalization_policy") != "per_image_minmax_v1":
            raise ValueError("SR checkpoint predates per-image normalization. Start a fresh run "
                             "with MODEL.LOAD_CHECKPOINT=false and TRAIN.RESUME=false.")

    def train(self):
        cfg = self.cfg
        if bool(cfg.TRAIN.RESUME) and bool(cfg.MODEL.LOAD_CHECKPOINT):
            raise ValueError("TRAIN.RESUME and MODEL.LOAD_CHECKPOINT cannot both be true")
        if cfg.TRAIN.RESUME or cfg.MODEL.LOAD_CHECKPOINT:
            path = (self.latest if cfg.TRAIN.RESUME else
                    Path(cfg.MODEL.CHECKPOINT_INPUT_PATH) if cfg.MODEL.CHECKPOINT_INPUT_PATH else self.checkpoint)
            self._validate_intensity_policy(read_checkpoint(path))
        if primary():
            (self.workdir / "sr_intensity_ranges.json").write_text(json.dumps({
                "raw_range": list(cfg.SR.RAW_RANGE), "gt_range": list(cfg.SR.GT_RANGE),
                "normalization_policy": "per_image_minmax_v1",
            }, indent=2), encoding="utf-8")
        train_path = prepare_once(make_sr_patch_store, cfg, "TRAIN", self.workdir / "train_sr_patches.zarr")
        val_path = prepare_once(make_sr_patch_store, cfg, "VAL", self.workdir / "val_sr_patches.zarr")
        ds_train = SRPatchDataset(train_path, train=True, cfg=cfg,
                                  noise_std=float(cfg.SR.AUGMENT_NOISE_STD),
                                  blur_sigma=float(cfg.SR.AUGMENT_BLUR_SIGMA))
        ds_val = SRPatchDataset(val_path)
        batch = int(cfg.TRAIN.BATCH_SIZE)
        if batch < 1:
            raise ValueError("TRAIN.BATCH_SIZE must be positive")
        train_dl = make_loader(ds_train, cfg, self.device)
        val_dl = make_loader(ds_val, cfg, self.device, training=False)
        model = self.build_model()
        log_message(model)
        print_model_parameters(model)
        print_cuda_memory(self.device, prefix="[SR TRAIN] model loaded")
        name = str(cfg.TRAIN.OPTIMIZER).upper()
        if name == "ADAMW":
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.TRAIN.LR),
                                    weight_decay=float(cfg.TRAIN.W_DECAY), betas=tuple(cfg.TRAIN.OPT_BETAS))
        elif name == "ADAM":
            opt = torch.optim.Adam(model.parameters(), lr=float(cfg.TRAIN.LR),
                                   weight_decay=float(cfg.TRAIN.W_DECAY), betas=tuple(cfg.TRAIN.OPT_BETAS))
        elif name == "SGD":
            opt = torch.optim.SGD(model.parameters(), lr=float(cfg.TRAIN.LR),
                                  weight_decay=float(cfg.TRAIN.W_DECAY), momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer {name}")
        schedname = str(cfg.TRAIN.LR_SCHEDULER.NAME).lower()
        steps = len(train_dl)
        report_steps = loss_log_steps(steps)
        if schedname == "warmupcosine":
            scheduler = build_warmup_cosine_scheduler(
                opt, epochs=int(cfg.TRAIN.EPOCHS), steps_per_epoch=steps,
                warmup_epochs=min(int(cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS),
                                  int(cfg.TRAIN.EPOCHS) - 1),
                min_lr=float(cfg.TRAIN.LR_SCHEDULER.MIN_LR))
        elif schedname == "one_cycle":
            scheduler = build_onecycle_scheduler(opt, max_lr=float(cfg.TRAIN.LR),
                                                 epochs=int(cfg.TRAIN.EPOCHS), steps_per_epoch=steps)
        else:
            raise ValueError(f"Unsupported scheduler {schedname}")
        amp = self.device.type == "cuda" and bool(cfg.TRAIN.AMP)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)
        start, best, stagnation = 0, float("inf"), 0
        if cfg.TRAIN.RESUME:
            state = self._load(model, self.latest)
            if state.get("identity") != self._identity():
                raise ValueError("SR checkpoint configuration identity changed; use a new run ID")
            if state.get("optimizer_state_dict") is None or state.get("scheduler_state_dict") is None:
                raise ValueError("SR resume requires complete optimizer/scheduler states")
            opt.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            if state.get("scaler_state_dict"):
                self.scaler.load_state_dict(state["scaler_state_dict"])
            from sknext.utils.checkpoint import _restore_rng
            _restore_rng(state)
            start = int(state["epoch"])
            best = float(state["best_val_loss"])
        elif cfg.MODEL.LOAD_CHECKPOINT:
            path = Path(cfg.MODEL.CHECKPOINT_INPUT_PATH) if cfg.MODEL.CHECKPOINT_INPUT_PATH else self.checkpoint
            self._load(model, path)
        epochs = int(cfg.TRAIN.EPOCHS)
        loss_chart = TrainingLossChart(self.workdir / "sr_history.jsonl",
                                       Path(cfg.PATHS.RESULT_DIR.OUTPUT_CHART) / "SR_Loss.png", start) if primary() else NullWriter()
        writer = SummaryWriter(log_dir=str(self.workdir / "sr_tensorboard")) if primary() else NullWriter()
        for epoch in range(start, epochs):
            set_epoch(train_dl, epoch)
            model.train()
            train_loss = 0.0
            train_samples = 0
            for step, (x, y) in enumerate(train_dl, start=1):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=amp):
                    pred = model(x)
                    loss = sr_loss(pred.float(), y, cfg)
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                scheduler.step()
                step_loss = loss.item()
                train_loss += step_loss * len(x)
                train_samples += len(x)
                if step in report_steps:
                    log_message(f"[SR] [EPOCH{epoch+1:04d}/{epochs:04d}] [TRAIN] "
                          f"[BATCH{step:06d}/{steps:06d}] loss: {step_loss:.6f}, "
                          f"mean_loss: {train_loss/train_samples:.6f}, "
                          f"lr: {opt.param_groups[0]['lr']:.6g}", flush=True)
            model.eval()
            val_loss, val_psnr, val_samples = 0., 0., 0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    with torch.autocast("cuda", enabled=amp):
                        pred = validation_model(model)(x)
                        loss = sr_loss(pred.float(), y, cfg)
                    val_loss += loss.item() * len(x)
                    val_psnr += sr_psnr(pred, y) * len(x)
                    val_samples += len(x)
            train_loss, train_samples, val_loss, val_psnr, val_samples = reduce_sums(
                [train_loss, train_samples, val_loss, val_psnr, val_samples], self.device)
            v = val_loss / max(1, val_samples)
            metrics = {"train_loss": train_loss/max(1, train_samples), "val_loss": v,
                       "val_psnr_db": val_psnr/max(1, val_samples),
                       "learning_rate": opt.param_groups[0]["lr"]}
            writer.add_scalars("SR/Loss", {"train": metrics["train_loss"],
                                           "val": metrics["val_loss"]}, epoch+1)
            writer.add_scalar("SR/PSNR_dB/val", metrics["val_psnr_db"], epoch+1)
            writer.add_scalar("SR/Learning_Rate", metrics["learning_rate"], epoch+1)
            writer.flush()
            loss_chart.log(epoch+1, metrics)
            log_message(f"[SR] epoch={epoch+1}/{epochs} train_loss={metrics['train_loss']:.6f} "
                  f"val_loss={v:.6f} val_PSNR={metrics['val_psnr_db']:.2f}dB", flush=True)
            if (epoch+1) % 5 == 0:
                print_cuda_memory(self.device, prefix=f"[SR TRAIN] epoch {epoch+1}, after validation",
                                  all_devices=True)
            self._rng_states = training_rng_states()
            if v < best:
                best, stagnation = v, 0
                if primary():
                    self._save(self._checkpoint_data(epoch+1, model, best=best), self.checkpoint)
            else:
                stagnation += 1
                log_message(f"no improvement, Patience: {stagnation:04d}/{int(cfg.TRAIN.PATIENCE):04d}", flush=True)
            if primary():
                self._save(self._checkpoint_data(epoch+1, model, opt, scheduler, best), self.latest)
            if int(cfg.TRAIN.PATIENCE) > 0 and stagnation >= int(cfg.TRAIN.PATIENCE):
                log_message(f"[SR] early stopping at epoch {epoch+1}", flush=True)
                break
        writer.close()

    def _input_sources(self):
        src = Path(self.cfg.DATA.INFER.PATH)
        if src.is_dir() and not src.name.lower().endswith(".zarr"):
            files = get_tif_path_in_folder(src)
            if not files:
                raise ValueError(f"No TIFF files in infer folder {src}")
            return files
        if not src.exists():
            raise FileNotFoundError(src)
        return [src]

    def _output_paths(self, source: Path):
        """Resolve the SR store and its sibling resume log from GT_PATH."""
        value = self.cfg.DATA.INFER.GT_PATH
        if not value:
            raise ValueError("DATA.INFER.GT_PATH must specify the SR output")
        target = Path(value)
        input_path = Path(self.cfg.DATA.INFER.PATH)
        if target.suffix.lower() == ".zarr":
            if input_path.is_dir() and input_path.suffix.lower() != ".zarr":
                raise ValueError("TIFF folder inference requires DATA.INFER.GT_PATH to be an output directory")
            out_path = target
        else:
            out_path = target / f"{source.stem}_SR_{self.scale}x.ome.zarr"
        # Overwrite must never remove the input or a directory containing it.
        resolved_source, resolved_output = source.resolve(), out_path.resolve()
        if (resolved_source == resolved_output or resolved_output in resolved_source.parents
                or resolved_source in resolved_output.parents):
            raise ValueError("SR output must be separate from the inference input")
        name = out_path.name[:-len(".ome.zarr")] if out_path.name.lower().endswith(".ome.zarr") else out_path.stem
        return out_path, out_path.with_name(f"{name}.progress.json")

    def _open_reader(self, path):
        if path.suffix.lower() in (".tif", ".tiff"):
            return _TiffReader(path)
        kind = detect_path_type(path)
        if kind == "zarr":
            return ZarrIOManager(path, mode="r")
        if kind == "ims":
            return IMSReader(path)
        raise ValueError(f"Unsupported SR input {path} ({kind})")

    @staticmethod
    def _write_infer_progress(path: Path, manifest: str, done: int, total: int):
        """Write progress atomically only after corresponding Zarr tile completes."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps({"manifest": manifest, "done": done,
                                       "total": total}), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @torch.no_grad()
    def _predict_sr_block(self, model, reader, core_lo, core_hi, input_bounds, geometry):
        core_lo, core_hi = np.asarray(core_lo), np.asarray(core_hi)
        core_size = core_hi - core_lo
        shape = tuple(map(int, reader.shape[-3:]))
        physical_ch = list(self.cfg.DATA.INFER.CHANNEL)
        factor = (1, self.scale, self.scale)
        padding, patch_central = geometry['padding'], geometry['patch_central']
        block_margin = geometry['block_margin']
        # One image read per channel per *block*. The surrounding
        # margin contains all patch halos, except at volume edges.
        read_lo = np.maximum(0, core_lo - block_margin)
        read_hi = np.minimum(shape, core_hi + block_margin)
        block = np.stack([
            np.asarray(reader[int(ch), int(read_lo[0]):int(read_hi[0]),
                                   int(read_lo[1]):int(read_hi[1]),
                                   int(read_lo[2]):int(read_hi[2])])
            for ch in physical_ch], axis=0)
        block = (np.zeros_like(block, dtype=np.float32) if input_bounds[0] == input_bounds[1]
                 else normalize_intensity(block, input_bounds, "inference raw"))
        hr_size = tuple(int(n * f) for n, f in zip(core_size, factor))
        # A block buffer permits exactly ONE write to the output
        # store per central block, rather than one write per patch.
        restored = np.empty((self.ch, *hr_size), dtype=np.float32)
        from sknext.utils.parallel import inference_batch_size, configured_inference_batch
        batch_size = inference_batch_size(model, configured_inference_batch(self.cfg))
        pending = []

        def flush_patches():
            if not pending:
                return
            inputs = torch.from_numpy(np.stack([p[0] for p in pending])).to(self.device)
            predictions = model(inputs)
            if isinstance(predictions, dict):
                predictions = predictions["pred"]
            expected = (len(pending), self.ch, *tuple(p * f for p, f in zip(self.patch, factor)))
            if tuple(predictions.shape) != expected:
                raise RuntimeError(f"SR model output shape mismatch: {tuple(predictions.shape)}")
            predictions = predictions.float().cpu().numpy()
            for prediction, (_, local, valid) in zip(predictions, pending):
                hhr = tuple(int(p * f) for p, f in zip(padding, factor))
                nhr = tuple(int(v * f) for v, f in zip(valid, factor))
                central_prediction = prediction[:,
                    hhr[0]:hhr[0] + nhr[0],
                    hhr[1]:hhr[1] + nhr[1],
                    hhr[2]:hhr[2] + nhr[2]]
                if bool(self.cfg.SR.OUTPUT_CLIP):
                    central_prediction = np.clip(central_prediction, 0., 1.)
                dst_lo = tuple(int(q * f) for q, f in zip(local, factor))
                dst = tuple(slice(s, s + length) for s, length in zip(dst_lo, nhr))
                restored[(slice(None), *dst)] = central_prediction
            pending.clear()

        for dz in range(0, int(core_size[0]), patch_central[0]):
            for dy in range(0, int(core_size[1]), patch_central[1]):
                for dx in range(0, int(core_size[2]), patch_central[2]):
                    local = np.array((dz, dy, dx), dtype=np.int64)
                    valid = np.minimum(patch_central, core_size - local)
                    global_start = core_lo + local - padding
                    patch_start_in_block = global_start - read_lo
                    raw_patch = crop_padded(block, patch_start_in_block, self.patch)
                    pending.append((raw_patch, local, valid))
                    if len(pending) >= batch_size:
                        flush_patches()
        flush_patches()
        return restored

    def _infer_one(self, model, source: Path, pool=None):
        """One I/O read/write per central block; GPU inference on padded patches.

        Patch context is DATA.INFER.PADDING. Block read extent is controlled by
        BLOCK_FACTOR/BLOCK_CENTRAL_FACTOR, with a derived margin. Read only a
        bounded source block, tile its *central* region without overlapping
        writes, reconstruct an HR central block in RAM, then commit that block
        with one Zarr write and one resumable progress-record update.
        """
        out_path, log_path = self._output_paths(source)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reader = self._open_reader(source)
        out = None
        try:
            physical_ch = list(self.cfg.DATA.INFER.CHANNEL)
            if not physical_ch or len(physical_ch) != self.ch:
                raise ValueError(f"DATA.INFER.CHANNEL must list exactly {self.ch} source channels")
            if min(physical_ch) < 0 or max(physical_ch) >= reader.shape[0]:
                raise ValueError("DATA.INFER.CHANNEL index out of range")
            shape = tuple(map(int, reader.shape[-3:]))
            factor = (1, self.scale, self.scale)
            out_shape = tuple(a * b for a, b in zip(shape, factor))
            geometry = sr_infer_geometry(self.cfg)
            padding = geometry["padding"]
            patch_central = geometry["patch_central"]
            central_block = geometry["central_block"]
            block_margin = geometry["block_margin"]
            input_bounds = reader_intensity_range(reader, physical_ch, central_block,
                                                  self.cfg.SR.RAW_RANGE)
            nc = tuple(math.ceil(a / b) for a, b in zip(shape, central_block))
            total = math.prod(nc)
            stat = source.stat()
            ckpt = (Path(self.cfg.MODEL.CHECKPOINT_INPUT_PATH) if
                    self.cfg.MODEL.CHECKPOINT_INPUT_PATH else self.checkpoint)
            ckptstat = ckpt.stat()
            # Batch size is an execution setting, not a resume compatibility key.
            manifest_args = {
                "version": 3, "source": str(source.resolve()),
                "normalization_policy": "per_image_minmax_v1",
                "input_image_range": list(input_bounds),
                "source_stat": [stat.st_size, stat.st_mtime_ns],
                "checkpoint": str(ckpt.resolve()),
                "checkpoint_stat": [ckptstat.st_size, ckptstat.st_mtime_ns],
                "channels": physical_ch, "shape": shape, "scale": self.scale,
                "patch": self.patch, "infer_padding": padding,
                "block_size": geometry["block_size"],
                "central_block": central_block, "block_margin": block_margin,
                "raw_range": list(self.cfg.SR.RAW_RANGE),
                "gt_range": list(self.cfg.SR.GT_RANGE),
                "clip": bool(self.cfg.SR.OUTPUT_CLIP),
            }
            manifest = hashlib.sha256(json.dumps(manifest_args, sort_keys=True).encode()).hexdigest()
            resume = bool(self.cfg.SR.INFER.RESUME)
            overwrite = bool(self.cfg.SR.INFER.OVERWRITE)
            done = 0
            if out_path.exists() and resume and not overwrite:
                if not log_path.is_file():
                    raise FileExistsError(f"Existing SR output has no progress log: {out_path}")
                state = json.loads(log_path.read_text(encoding="utf-8"))
                if state.get("manifest") != manifest or state.get("total") != total:
                    raise ValueError("SR resume manifest mismatch. Check source/weights/block geometry or use a new output path")
                done = int(state["done"])
                if not 0 <= done <= total:
                    raise ValueError("Corrupted SR inference progress counter")
                out = ZarrIOManager(out_path, shape=(self.ch, *out_shape),
                                    dtype="float32", mode="r+")
            else:
                if out_path.exists():
                    if not overwrite:
                        raise FileExistsError(f"SR output already exists: {out_path}")
                    import shutil
                    shutil.rmtree(out_path)
                if log_path.exists():
                    if not overwrite:
                        raise FileExistsError(f"SR progress log exists without output: {log_path}")
                    log_path.unlink()
                v = list(self.cfg.SR.INFER.VOXEL_SIZE)
                if not v:
                    v = getattr(reader, "voxel_size", None)
                    if v is None or len(v) != 3:
                        warnings.warn(f"Missing spatial calibration for {source.name}; "
                                      "assuming unit voxel size. Set SR.INFER.VOXEL_SIZE.", stacklevel=2)
                        v = (1., 1., 1.)
                vout = tuple(float(a) / b for a, b in zip(v, factor))
                out = ZarrIOManager(out_path, shape=(self.ch, *out_shape), dtype="float32",
                                    chunks=(1, min(16, out_shape[0]),
                                            min(256, out_shape[1]), min(256, out_shape[2])),
                                    mode="a", voxel_size=vout, image_name="SkNeXt SR restored",
                                    channel_names=[f"SR_{i}" for i in range(self.ch)])
                out.root.attrs["sknext_normalization"] = {
                    "policy": "per_image_minmax_v1", "input_image_range": list(input_bounds)}
                self._write_infer_progress(log_path, manifest, 0, total)
            if done == total:
                log_message(f"[SR] already completed: {out_path}", flush=True)
                return

            def jobs():
                block_index = 0
                for zz in range(0, shape[0], central_block[0]):
                    for yy in range(0, shape[1], central_block[1]):
                        for xx in range(0, shape[2], central_block[2]):
                            core_lo = np.array((zz, yy, xx), dtype=np.int64)
                            core_hi = np.minimum(core_lo + central_block, shape)
                            if block_index >= done:
                                yield dict(id=block_index, source=str(source), start=core_lo, stop=core_hi,
                                           bounds=input_bounds, geometry=geometry)
                            block_index += 1

            results = (pool.map(jobs()) if pool is not None else
                       ((job, self._predict_sr_block(model, reader, job['start'], job['stop'],
                                                    input_bounds, geometry)) for job in jobs()))
            for job, restored in results:
                core_lo, block_index = job['start'], job['id']
                hr_size = restored.shape[1:]
                out_lo = tuple(int(c * f) for c, f in zip(core_lo, factor))
                dest = tuple(slice(s, s + length) for s, length in zip(out_lo, hr_size))
                out[(slice(None), *dest)] = restored
                block_index += 1
                self._write_infer_progress(log_path, manifest, block_index, total)
                if block_index == done + 1 or block_index % 10 == 0 or block_index == total:
                    log_message(f"[SR] [BLOCK{block_index:07d}/{total:07d}] [INFER] results saved, source: {source.name}", flush=True)
                if pool is None and block_index % 20 == 0:
                    print_cuda_memory(self.device, prefix=f"[SR INFER] block {block_index}/{total}",
                                      all_devices=True)
            log_message(f"[SR] restored volume: {out_path} CZYX={(self.ch, *out_shape)}", flush=True)
        finally:
            if out is not None:
                out.close()
            reader.close()

    def infer(self):
        from sknext.utils.inference_workers import BlockPool, parallel_inference_enabled
        from contextlib import ExitStack
        use_pool = parallel_inference_enabled(self.cfg, self.device)
        model = None
        if not use_pool:
            model = self.build_model()
            path = Path(self.cfg.MODEL.CHECKPOINT_INPUT_PATH) if self.cfg.MODEL.CHECKPOINT_INPUT_PATH else self.checkpoint
            self._load(model, path)
            model.eval()
            print_model_parameters(model)
        with ExitStack() as stack:
            pool = stack.enter_context(BlockPool(self.cfg, self.job_id, 'restoration')) if use_pool else None
            for source in self._input_sources():
                self._infer_one(model, source, pool)

    def run(self):
        if bool(self.cfg.TRAIN.ENABLE) and bool(self.cfg.INFER.ENABLE):
            raise ValueError("Enable either TRAIN or INFER, not both")
        if self.cfg.TRAIN.ENABLE:
            self.train()
        elif self.cfg.INFER.ENABLE:
            self.infer()
        else:
            raise ValueError("Set TRAIN.ENABLE or INFER.ENABLE to true for SR")
