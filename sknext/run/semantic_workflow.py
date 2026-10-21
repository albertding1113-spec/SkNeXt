"""Semantic-only 3D segmentation runner for SkNeXt.

Unified CLI (TASK.CHANNELS contains only S.<name>; mode comes from YAML)::

    sknext --config semantic_only.yaml --run_id 0 --gpu 0

Standalone CLI (also supported)::

    python -m sknext.run.semantic_workflow --config semantic_only.yaml --mode train --run_id 0
    python -m sknext.run.semantic_workflow --config semantic_only.yaml --mode infer --run_id 0

TASK.CHANNELS must exclusively contain S.<name> labels. Predictions are independent
sigmoid binary masks. No instance labels, skeletons, or watershed are used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from sknext.data.data_loader import ZarrPatchLoader
from sknext.data.datasetIO import (tif_list_to_zarr, IMSReader, ZarrIOManager,
                                   detect_path_type)
from sknext.data.imageIO import (get_tif_path_dict_in_folder,
                                 get_tif_path_in_folder, pair_raw_and_label_tiffs,
                                 patch_coordinates_iter)
from sknext.data.preprocessing import preprocess_img, reflect_padding_img
from sknext.model.unext_v2 import U_NeXt_V2
from sknext.model.loss import instance_segmentation_loss
from sknext.model.scheduler import build_warmup_cosine_scheduler, build_onecycle_scheduler
from sknext.utils.checkpoint import (read_checkpoint, load_model_weights,
                                     restore_training_checkpoint, training_checkpoint_paths)
from sknext.utils.model_info import print_cuda_memory
from sknext.utils.progress import loss_log_steps, log_message


def cfgget(cfg: dict, path: str, default=None):
    obj = cfg
    for key in path.split('.'):
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def checked_semantic_names(cfg: dict) -> list[str]:
    names = list(cfgget(cfg, 'TASK.CHANNELS', []))
    if not names or any(not isinstance(n, str) or not n.startswith('S.') or len(n) < 3 for n in names):
        raise ValueError('Semantic-only runner requires TASK.CHANNELS: [S.ER, S.mito, ...]')
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate semantic channels: {names}')
    return names


def get_thresholds(cfg: dict, num_channels: int) -> np.ndarray:
    value = cfgget(cfg, 'INFER.THRESHOLD', 0.5)
    if isinstance(value, (float, int)):
        value = [value] * num_channels
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (num_channels,) or not np.isfinite(value).all() or np.any((value <= 0) | (value >= 1)):
        raise ValueError('INFER.THRESHOLD must be a number or one number per channel, all in (0,1)')
    return value


def spatial_patch(cfg: dict) -> tuple[int, int, int]:
    patch = cfgget(cfg, 'DATA.PATCH_SIZE', [32, 128, 128, 1])
    if len(patch) != 4 or min(patch) <= 0:
        raise ValueError('DATA.PATCH_SIZE must be four positive ZYXC integers')
    return tuple(map(int, patch[:3]))


def overlap_voxels(cfg: dict, part: str) -> tuple[int, int, int]:
    fraction = np.asarray(cfgget(cfg, f'DATA.{part}.OVERLAP', [0., 0.25, 0.25]), dtype=float)
    if fraction.shape != (3,) or np.any(fraction < 0) or np.any(fraction >= 1):
        raise ValueError('DATA.*.OVERLAP must be fractions in [0, 1), in ZYX order')
    values = np.rint(fraction * np.asarray(spatial_patch(cfg))).astype(int)
    if np.any(values >= spatial_patch(cfg)):
        raise ValueError('Overlap in voxels must be smaller than patch')
    return tuple(map(int, values))


def normalize_options(cfg: dict) -> dict:
    nc = cfgget(cfg, 'DATA.NORMALIZATION', {})
    if str(nc.get('SCOPE', 'patch')).lower() != 'patch':
        raise ValueError("DATA.NORMALIZATION.SCOPE must be 'patch' for consistent training/inference")
    pc = nc.get('PERC_CLIP', {})
    return {
        'perc_clip': bool(pc.get('ENABLE', False)),
        'perc_clip_range': [float(pc.get('LOWER_PERC', 1)), float(pc.get('UPPER_PERC', 99.8))],
        'normalization': bool(nc.get('ENABLE', True)),
        'norm_type': nc.get('TYPE', 'scale_range'),
    }


def make_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    channels = checked_semantic_names(cfg)
    patch = cfgget(cfg, 'DATA.PATCH_SIZE', [32, 128, 128, 1])
    features = list(cfgget(cfg, 'MODEL.FEATURE_MAPS', [16,32,64,128,256]))
    layers = list(cfgget(cfg, 'MODEL.CONVNEXT_LAYERS', [2] * len(features)))
    zd = list(cfgget(cfg, 'MODEL.Z_DOWN', [1,1,2,2]))
    yxd = list(cfgget(cfg, 'MODEL.YX_DOWN', [2] * (len(features) - 1)))
    isotropy = cfgget(cfg, 'MODEL.ISOTROPY', True)
    if len(features) < 2 or len(layers) != len(features) or len(zd) != len(features)-1 or len(yxd) != len(features)-1:
        raise ValueError('MODEL.FEATURE_MAPS, CONVNEXT_LAYERS, Z_DOWN and YX_DOWN have inconsistent lengths')
    div = np.asarray((max(1, int(cfgget(cfg, 'MODEL.CONVNEXT_STEM_K_SIZE', 2))),) * 3, dtype=int)
    div[0] = max(int(zd[0]), int(div[0]))
    div *= np.prod(np.array([(z,y,y) for z,y in zip(zd,yxd)]), axis=0).astype(int)
    if np.any(np.asarray(patch[:3]) % div != 0):
        raise ValueError(f'Patch {patch[:3]} must be divisible by total downsample factors {div.tolist()}')
    model = U_NeXt_V2(
        image_shape=tuple(patch), feature_maps=features, cn_layers=layers,
        z_down=zd, yx_down=yxd, isotropy=isotropy,
        stem_k_size=int(cfgget(cfg, 'MODEL.CONVNEXT_STEM_K_SIZE', 2)),
        stochastic_depth_prob=float(cfgget(cfg, 'MODEL.CONVNEXT_SD_PROB', 0.1)),
        output_channels=[len(channels)], output_channel_info=['semantic'],
        separated_decoders=False, explicit_activations=False, contrast=False,
    )
    return model.to(device)


def checkpoint_path(cfg: dict, run_id: int) -> Path:
    override = cfgget(cfg, 'MODEL.CHECKPOINT', None)
    if override:
        return Path(override).expanduser()
    base = Path(cfgget(cfg, 'PATHS.RESULT_DIR.PATH', './result'))
    return base / 'results' / str(run_id) / 'checkpoints' / f'checkpoint_{run_id:02d}.pth'


def checkpoint_read_path(cfg: dict, run_id: int) -> Path:
    """Use a legacy input checkpoint without overwriting it during training."""
    path = cfgget(cfg, 'MODEL.CHECKPOINT_INPUT_PATH', '')
    return Path(path).expanduser() if path else checkpoint_path(cfg, run_id)


def load_model_checkpoint(model, cfg, run_id, device):
    """Restore weights from legacy semantic-only or ordinary PyTorch checkpoints."""
    path = checkpoint_read_path(cfg, run_id)
    state = read_checkpoint(path)
    report = load_model_weights(
        model, state,
        expected_architecture="U_NeXt_V2",
        expected_channels=checked_semantic_names(cfg),
        allow_partial=bool(cfgget(cfg, 'MODEL.CHECKPOINT_ALLOW_PARTIAL', False)))
    model.to(device)
    log_message(f"Loaded {report['loaded']} model tensors from {path} "
          f"(partial={report['partial']})", flush=True)
    return state


def _pair_training_files(cfg, part: str, names: list[str]):
    raw_path = Path(cfgget(cfg, f'DATA.{part}.PATH', ''))
    label_path = Path(cfgget(cfg, f'DATA.{part}.GT_PATH', ''))
    raw_files = get_tif_path_in_folder(raw_path)
    groups = get_tif_path_dict_in_folder(label_path, names)
    if not raw_files:
        raise ValueError(f'No raw TIFFs found in {raw_path}')
    raw_files, groups = pair_raw_and_label_tiffs(raw_files, groups)
    return raw_files, groups, raw_path


def _dataset_path(raw_dir: Path, part: str) -> Path:
    return raw_dir.parent / f'{part.lower()}_semantic_patches.zarr'


def prepare_patch_data(cfg: dict, part: str, names: list[str]) -> Path:
    files, groups, raw_dir = _pair_training_files(cfg, part, names)
    output = _dataset_path(raw_dir, part)
    filt = cfgget(cfg, 'DATA.TRAIN.FILTER', {}) if part == 'TRAIN' else {}
    fd = {
        'enable': bool(filt.get('ENABLE', False)),
        'props': list(filt.get('PROPS', [])),
        'values': list(filt.get('VALUES', [])),
        'signs': list(filt.get('SIGNS', [])),
    }
    channel_extra = cfgget(cfg, 'TASK.CHANNELS_EXTRA_OPTS', [{}])
    if isinstance(channel_extra, list):
        channel_extra = channel_extra[0] if channel_extra else {}
    tif_list_to_zarr(
        files, groups, output,
        patch_size=tuple(cfgget(cfg, 'DATA.PATCH_SIZE')),
        overlap=overlap_voxels(cfg, part),
        padding=tuple(cfgget(cfg, f'DATA.{part}.PADDING', [0,0,0])),
        preprocess_dict=normalize_options(cfg), channels=names,
        channel_extra_opts=channel_extra,
        filter_dict=fd, normalize_per_patch=True,
    )
    return output


def train(cfg: dict, run_id: int, device: torch.device) -> None:
    names = checked_semantic_names(cfg)
    weights = cfgget(cfg, 'TASK.CHANNEL_WEIGHTS', [1.] * len(names))
    if len(weights) != len(names) or any(float(v) <= 0 for v in weights):
        raise ValueError('TASK.CHANNEL_WEIGHTS must contain one positive weight per semantic channel')
    train_path = prepare_patch_data(cfg, 'TRAIN', names)
    val_path = prepare_patch_data(cfg, 'VAL', names)
    train_ds = ZarrPatchLoader(cfg, train_path, if_val=False)
    val_ds = ZarrPatchLoader(cfg, val_path, if_val=True)
    batch_size = int(cfgget(cfg, 'TRAIN.BATCH_SIZE', 1))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                         num_workers=int(cfgget(cfg, 'TRAIN.NUM_WORKERS', 0)), pin_memory=device.type == 'cuda')
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = make_model(cfg, device)
    print_cuda_memory(device, prefix="[SEMANTIC TRAIN] model loaded")
    resume = bool(cfgget(cfg, 'TRAIN.RESUME', False))
    if resume and cfgget(cfg, 'MODEL.LOAD_CHECKPOINT', False):
        raise ValueError("TRAIN.RESUME and MODEL.LOAD_CHECKPOINT are mutually exclusive")
    if not resume and cfgget(cfg, 'MODEL.LOAD_CHECKPOINT', False):
        load_model_checkpoint(model, cfg, run_id, device)
    loss_fn = instance_segmentation_loss(channel_weights=weights, channel_names=names,
                                         bce_weight=float(cfgget(cfg, 'TRAIN.BCE_WEIGHT', 1.0)),
                                         dice_weight=float(cfgget(cfg, 'TRAIN.DICE_WEIGHT', 1.0)),
                                         auto_balance=bool(cfgget(cfg, 'TRAIN.AUTO_BALANCE', True)))
    lr = float(cfgget(cfg, 'TRAIN.LR', 1e-4))
    if str(cfgget(cfg, 'TRAIN.OPTIMIZER', 'ADAMW')).upper() != 'ADAMW':
        raise ValueError('Semantic-only runner currently supports TRAIN.OPTIMIZER: ADAMW')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(cfgget(cfg, 'TRAIN.W_DECAY', 0.01)),
                                  betas=tuple(cfgget(cfg, 'TRAIN.OPT_BETAS', [0.9,0.999])))
    epochs = int(cfgget(cfg, 'TRAIN.EPOCHS', 100))
    steps = len(loader)
    report_steps = loss_log_steps(steps)
    sch_name = str(cfgget(cfg, 'TRAIN.LR_SCHEDULER.NAME', 'warmupcosine')).lower()
    if sch_name in {'warmupcosine', 'warmup_cosine'}:
        scheduler = build_warmup_cosine_scheduler(
            optimizer, total_steps=epochs*steps, steps_per_epoch=steps,
            warmup_epochs=int(cfgget(cfg, 'TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS', 5)),
            min_lr=float(cfgget(cfg, 'TRAIN.LR_SCHEDULER.MIN_LR', 1e-6)),
        )
    elif sch_name in {'one_cycle', 'onecycle'}:
        scheduler = build_onecycle_scheduler(optimizer, max_lr=lr, total_steps=epochs*steps)
    else:
        raise ValueError(f'Unsupported LR scheduler: {sch_name}')
    amp = device.type == 'cuda' and bool(cfgget(cfg, 'TRAIN.AMP', True))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ckpt = checkpoint_path(cfg, run_id)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, stagnant = float('inf'), 0
    patience = int(cfgget(cfg, 'TRAIN.PATIENCE', 20))
    latest = ckpt.with_name(ckpt.stem + '_latest.pth')
    model_identity_cfg = dict(cfgget(cfg, 'MODEL', {}))
    model_identity_cfg.pop('CHECKPOINT_ALLOW_PARTIAL', None)
    model_identity_cfg.pop('CHECKPOINT_INPUT_PATH', None)
    identity = hashlib.sha256(json.dumps({
        'model': model_identity_cfg, 'channels': names, 'weights': list(weights),
        'normalization': normalize_options(cfg), 'augmentation': cfgget(cfg, 'AUGMENTOR', {}),
        'patch': list(cfgget(cfg, 'DATA.PATCH_SIZE', [])),
        'training': {key: cfgget(cfg, 'TRAIN.'+key) for key in
                     ['BATCH_SIZE','EPOCHS','LR','W_DECAY','OPT_BETAS','OPTIMIZER','LR_SCHEDULER',
                      'BCE_WEIGHT','DICE_WEIGHT','AUTO_BALANCE']},
        'train_path': str(train_path.resolve()), 'val_path': str(val_path.resolve()),
        'train_samples': len(train_ds), 'val_samples':len(val_ds),
        'input_files': [
            (str(file.resolve()),file.stat().st_size,file.stat().st_mtime_ns)
            for location in (cfgget(cfg,'DATA.TRAIN.PATH'), cfgget(cfg,'DATA.TRAIN.GT_PATH'),
                             cfgget(cfg,'DATA.VAL.PATH'), cfgget(cfg,'DATA.VAL.GT_PATH'))
            for file in sorted(Path(location).iterdir())
            if file.is_file() and file.suffix.lower() in {'.tif', '.tiff'}
        ],
        'steps':steps}, sort_keys=True, default=str).encode()).hexdigest()
    start_epoch = 0
    if resume:
        input_latest, input_best = training_checkpoint_paths(checkpoint_read_path(cfg, run_id))
        restoration = restore_training_checkpoint(
            model, optimizer, scheduler,
            latest_path=input_latest,
            best_path=input_best,
            identity=identity, epochs=epochs,
            expected_architecture='U_NeXt_V2',
            expected_channels=names,
            allow_partial=bool(cfgget(cfg, 'MODEL.CHECKPOINT_ALLOW_PARTIAL', False)),
            scaler=scaler)
        start_epoch = restoration['epoch']
        best = restoration['best_val_loss']
        stagnant = restoration['epochs_without_improvement']
        log_message(f"Semantic training {restoration['mode']} from "
              f"{restoration['source']}, next epoch {start_epoch + 1}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        train_sum = 0.
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits = model(inputs)
                if isinstance(logits, dict):
                    logits = logits['pred']
                loss = loss_fn(logits.float(), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step_loss = float(loss.detach())
            train_sum += step_loss
            if step in report_steps:
                log_message(f"[SEMANTIC] [EPOCH{epoch+1:04d}/{epochs:04d}] [TRAIN] "
                      f"[BATCH{step:06d}/{steps:06d}] loss: {step_loss:.6f}, "
                      f"mean_loss: {train_sum/step:.6f}, "
                      f"lr: {optimizer.param_groups[0]['lr']:.6g}", flush=True)
            if step == 1:
                print_cuda_memory(device, prefix=f"[SEMANTIC TRAIN] epoch {epoch+1}, first batch")
        model.eval()
        valid_sum, count = 0., 0
        intersections = np.zeros(len(names), dtype=np.float64)
        unions = np.zeros(len(names), dtype=np.float64)
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    logits = model(inputs)
                    if isinstance(logits, dict): logits = logits['pred']
                    loss = loss_fn(logits.float(), targets)
                valid_sum += float(loss) * len(inputs)
                count += len(inputs)
                binary = torch.sigmoid(logits.float()) >= 0.5
                truth = targets >= 0.5
                dims = (0, 2, 3, 4)
                intersections += (binary & truth).sum(dim=dims).cpu().numpy()
                unions += (binary | truth).sum(dim=dims).cpu().numpy()
        mean_val = valid_sum / max(1, count)
        ious = np.divide(intersections, unions, out=np.ones_like(intersections), where=unions>0)
        log_message(f'Epoch {epoch+1:4d}/{epochs}: train={train_sum/max(1, steps):.5f}, '
              f'val={mean_val:.5f}, IoU={dict(zip(names, np.round(ious, 4)))}', flush=True)
        print_cuda_memory(device, prefix=f"[SEMANTIC TRAIN] epoch {epoch+1}, after validation")
        if mean_val < best:
            best, stagnant = mean_val, 0
            torch.save({'model_state_dict': model.state_dict(), 'semantic_channels': names,
                        'epoch': epoch+1, 'val_loss': best, 'input_channels': int(cfgget(cfg, 'DATA.PATCH_SIZE')[-1]),
                        'architecture': 'U_NeXt_V2_semantic_only'}, ckpt)
            log_message(f'Checkpoint saved: {ckpt}', flush=True)
        else:
            stagnant += 1
        np_rng = np.random.get_state()
        state = {'format_version':2, 'epoch':epoch+1, 'identity':identity,
                 'model_state_dict':model.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),
                 'scheduler_state_dict':scheduler.state_dict(),
                 'scaler_state_dict':scaler.state_dict(),
                 'best':best, 'stagnant':stagnant,
                 'rng_python':random.getstate(),
                 'rng_numpy':(np_rng[0], np_rng[1].tolist(), np_rng[2], np_rng[3], np_rng[4]),
                 'rng_torch':torch.get_rng_state(),
                 'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        tmp = latest.with_name('.' + latest.name + '.tmp')
        try:
            torch.save(state, tmp)
            os.replace(tmp, latest)
        finally:
            if tmp.exists(): tmp.unlink()
        if patience > 0 and stagnant >= patience:
            log_message('Early stopping', flush=True)
            break


def _input_reader(path: Path, channel_indices: list[int]):
    kind = detect_path_type(path)
    if kind == 'ims':
        # Logical reader channels now follow requested channel order.
        return IMSReader(path, channel=channel_indices), list(range(len(channel_indices)))
    if kind == 'zarr':
        return ZarrIOManager(path, mode='r'), channel_indices
    raise ValueError(f'Inference expects .ims or CZYX OME-Zarr, got {path}')


def _window(patch: tuple[int, int, int]) -> np.ndarray:
    axes = [np.maximum(np.hanning(length).astype('float32'), 0.05) if length > 2
            else np.ones(length, dtype=np.float32) for length in patch]
    return axes[0][:,None,None] * axes[1][None,:,None] * axes[2][None,None,:]


def _prediction_model_result(model, tensor):
    out = model(tensor)
    if isinstance(out, dict):
        out = out['pred']
    return out


def _predict_block(model, raw: np.ndarray, cfg: dict, device: torch.device,
                   names: list[str], amp: bool) -> np.ndarray:
    """Infer one bounded CZYX block and blend overlapping patch probabilities."""
    patch_size = spatial_patch(cfg)
    sz = raw.shape[1:]
    accum = np.zeros((len(names), *sz), dtype=np.float32)
    divisor = np.zeros(sz, dtype=np.float32)
    win = _window(patch_size)
    overlap = overlap_voxels(cfg, 'INFER')
    coords = list(patch_coordinates_iter(sz, patch_size, overlap, (0,0,0)))
    batch_size = max(1, int(cfgget(cfg, 'TRAIN.BATCH_SIZE', 1)))
    norm = normalize_options(cfg)
    skip_empty = bool(cfgget(cfg, 'INFER.SKIP_ZERO_PATCHES', True))
    for batch_start in range(0, len(coords), batch_size):
        coords_batch = coords[batch_start:batch_start+batch_size]
        tensor_patches = []
        nonempty_indices = []
        for i, coord in enumerate(coords_batch):
            z0,z1 = coord[0]; y0,y1=coord[1]; x0,x1=coord[2]
            raw_patch = raw[:,z0:z1,y0:y1,x0:x1]
            if skip_empty and not np.any(raw_patch):
                continue
            raw_patch = reflect_padding_img(raw_patch, patch_size)
            raw_patch = preprocess_img(raw_patch, norm).astype(np.float32, copy=False)
            tensor_patches.append(raw_patch)
            nonempty_indices.append(i)
        if nonempty_indices:
            xx = torch.from_numpy(np.stack(tensor_patches)).to(device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                probabilities = torch.sigmoid(_prediction_model_result(model, xx).float()).cpu().numpy()
            del xx
        else:
            probabilities = np.zeros((0,len(names),*patch_size),dtype=np.float32)
        results = {i: probabilities[j] for j,i in enumerate(nonempty_indices)}
        for i, coord in enumerate(coords_batch):
            z0,z1=coord[0]; y0,y1=coord[1]; x0,x1=coord[2]
            actual = (int(z1-z0), int(y1-y0), int(x1-x0))
            weight = win[:actual[0], :actual[1], :actual[2]]
            if i in results:
                prob = results[i][:,:actual[0],:actual[1],:actual[2]]
                accum[:, z0:z1,y0:y1,x0:x1] += prob * weight[None]
            divisor[z0:z1,y0:y1,x0:x1] += weight
    return accum / np.maximum(divisor[None], 1e-8)


def _resolved_output(cfg: dict, run_id: int) -> Path:
    value = cfgget(cfg, 'DATA.INFER.GT_PATH', '')
    if not value:
        raise ValueError('DATA.INFER.GT_PATH must specify the semantic segmentation output')
    path = Path(value)
    return path if path.name.endswith('.zarr') else path / f'semantic_{run_id}.ome.zarr'


def infer(cfg: dict, run_id: int, device: torch.device, overwrite: bool=False) -> None:
    names = checked_semantic_names(cfg)
    model = make_model(cfg, device)
    load_model_checkpoint(model, cfg, run_id, device)
    model.eval()
    print_cuda_memory(device, prefix="[SEMANTIC INFER] model loaded")
    amp = device.type == 'cuda' and bool(cfgget(cfg, 'INFER.AMP', True))
    source = Path(cfgget(cfg, 'DATA.INFER.PATH', ''))
    channels = list(map(int, cfgget(cfg, 'DATA.INFER.CHANNEL', [0])))
    if len(channels) != int(cfgget(cfg, 'DATA.PATCH_SIZE')[-1]):
        raise ValueError('DATA.INFER.CHANNEL length must equal DATA.PATCH_SIZE[-1]')
    output = _resolved_output(cfg, run_id)
    if output.resolve() == source.resolve():
        raise ValueError('DATA.INFER.GT_PATH must not overwrite the input data store')
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.parent / (output.name + '.progress.json')
    reader, logical_channels = _input_reader(source, channels)
    try:
        zyx = tuple(map(int, reader.shape[1:]))
        core_factor = np.asarray(cfgget(cfg, 'DATA.INFER.BLOCK_CENTRAL_FACTOR', [1,2,2]), dtype=int)
        if core_factor.shape != (3,) or np.any(core_factor < 1):
            raise ValueError('DATA.INFER.BLOCK_CENTRAL_FACTOR must have three positive integers')
        core_size = tuple((np.asarray(spatial_patch(cfg)) * core_factor).tolist())
        halo = tuple(map(int, cfgget(cfg, 'DATA.INFER.HALO', []) or
                                [max(1,x//2) for x in spatial_patch(cfg)]))
        if len(halo) != 3 or min(halo) < 0:
            raise ValueError('DATA.INFER.HALO must contain nonnegative ZYX widths')
        num_core = tuple(math.ceil(n / c) for n,c in zip(zyx, core_size))
        run_meta = {'source': str(source.resolve()), 'source_shape':list(zyx), 'channels':names,
                    'format_version': 2,
                    'patch_size':list(spatial_patch(cfg)), 'core_size':list(core_size),
                    'halo':list(halo), 'input_channels':channels,
                    'normalization': normalize_options(cfg),
                    'overlap_voxels': list(overlap_voxels(cfg, 'INFER')),
                    'skip_zero_patches': bool(cfgget(cfg, 'INFER.SKIP_ZERO_PATCHES', True)),
                    'amp': amp,
                    'batch_size': max(1, int(cfgget(cfg, 'TRAIN.BATCH_SIZE', 1))),
                    'checkpoint':str(checkpoint_read_path(cfg, run_id).resolve()),
                    'checkpoint_size':checkpoint_read_path(cfg, run_id).stat().st_size,
                    'checkpoint_mtime_ns':checkpoint_read_path(cfg, run_id).stat().st_mtime_ns,
                    'thresholds':get_thresholds(cfg,len(names)).tolist(),
                    'save_probabilities':bool(cfgget(cfg, 'INFER.SAVE_PROBABILITIES', False))}
        if overwrite:
            if output.exists():
                from sknext.data.datasetIO import remove_zarr_if_exists
                remove_zarr_if_exists(output)
            if progress.exists(): progress.unlink()
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved.get('meta') != run_meta:
                raise RuntimeError('Cannot resume: inference configuration / checkpoint differs from progress metadata')
            if not output.exists():
                raise RuntimeError('Progress exists but output data is missing')
            completed = set(saved.get('completed', []))
        else:
            if output.exists():
                raise FileExistsError(f'Output already exists without progress metadata: {output}; use --overwrite')
            completed = set()
        thresholds = get_thresholds(cfg, len(names))
        # Keep probabilities in a *separate* OME-Zarr to avoid overloading labels.
        save_prob = bool(cfgget(cfg, 'INFER.SAVE_PROBABILITIES', False))
        prob_path = output.parent / (output.name[:-5] + '_prob.zarr')
        if save_prob and completed and not prob_path.exists():
            raise FileNotFoundError('Cannot resume probability output: probability store is missing')
        if save_prob and prob_path.exists() and not progress.exists() and not overwrite:
            raise FileExistsError(f'Probability store exists without progress: {prob_path}')
        if save_prob and overwrite and prob_path.exists():
            from sknext.data.datasetIO import remove_zarr_if_exists
            remove_zarr_if_exists(prob_path)
        # output store remains CZYX, but contains NO instance ID channel.
        chunks = (1,min(16,zyx[0]),min(256,zyx[1]),min(256,zyx[2]))
        with ZarrIOManager(output, shape=(len(names),*zyx), dtype='uint8', chunks=chunks,
                           mode='a', channel_names=[ch[2:] for ch in names]) as dst:
            dst.root.attrs['sknext_semantic_channels'] = names
            prob_writer = (ZarrIOManager(prob_path,shape=(len(names),*zyx),dtype='uint16',
                                          mode='a',chunks=chunks,
                                          channel_names=[ch[2:] for ch in names])
                           if save_prob else None)
            try:
                total = math.prod(num_core)
                block_id = 0
                for zz in range(0, zyx[0], core_size[0]):
                    for yy in range(0, zyx[1], core_size[1]):
                        for xx in range(0, zyx[2], core_size[2]):
                            if block_id in completed:
                                block_id += 1
                                continue
                            start = np.array([zz,yy,xx], dtype=int)
                            stop = np.minimum(start + core_size, zyx)
                            low = np.maximum(start - halo, 0)
                            high = np.minimum(stop + halo, zyx)
                            key = tuple(slice(int(a),int(b)) for a,b in zip(low,high))
                            block = np.stack([reader[(ch,*key)] for ch in logical_channels], axis=0)
                            probs = _predict_block(model, block, cfg, device, names, amp)
                            local = tuple(slice(int(a-b),int(c-b)) for a,c,b in zip(start,stop,low))
                            write_key = (slice(None), *[slice(int(a),int(b)) for a,b in zip(start,stop)])
                            center_probs = probs[(slice(None),*local)]
                            masks = (center_probs >= thresholds[:,None,None,None]).astype('uint8')
                            dst[write_key] = masks
                            if prob_writer is not None:
                                prob_writer[write_key] = np.rint(center_probs * 65535.).astype('uint16')
                            completed.add(block_id)
                            # Update progress atomically: a crash cannot truncate its JSON.
                            temporary = progress.with_name(progress.name + '.tmp')
                            temporary.write_text(json.dumps({'meta':run_meta, 'completed':sorted(completed)},indent=2))
                            os.replace(temporary, progress)
                            log_message(f'[BLOCK{block_id+1:07d}/{total:07d}] [INFER] results saved, '
                                  f'core={start.tolist()}:{stop.tolist()}',flush=True)
                            if len(completed) == 1 or (block_id+1) % 10 == 0 or block_id+1 == total:
                                print_cuda_memory(device, prefix=f"[SEMANTIC INFER] block {block_id+1}/{total}")
                            block_id += 1
            finally:
                if prob_writer is not None: prob_writer.close()
        log_message(f'Semantic-only results saved to: {output}',flush=True)
    finally:
        reader.close()


def run(cfg: dict, run_id: int, device: torch.device, mode: str = 'auto',
        overwrite: bool = False):
    """Shared entry point for sknext and the standalone semantic CLI."""
    checked_semantic_names(cfg)
    if mode == 'auto':
        training_enabled = bool(cfgget(cfg, 'TRAIN.ENABLE', False))
        inference_enabled = bool(cfgget(cfg, 'INFER.ENABLE', False))
        if training_enabled == inference_enabled:
            raise ValueError('Choose exactly one: TRAIN.ENABLE=true or INFER.ENABLE=true, or pass --mode')
        mode = 'train' if training_enabled else 'infer'
    if mode not in {'train', 'infer'}:
        raise ValueError(f'Unknown semantic workflow mode: {mode}')
    log_message(f'Semantic-only workflow: {mode}, device={device}', flush=True)
    if mode == 'train':
        return train(cfg, run_id, device)
    return infer(cfg, run_id, device, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description='SkNeXt semantic-only training/inference (no SWC, no watershed)')
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['auto','train','infer'],default='auto',
                        help='auto reads TRAIN.ENABLE / INFER.ENABLE in the YAML')
    parser.add_argument('--run_id',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--overwrite',action='store_true',help='Overwrite preexisting inference output')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    if not isinstance(cfg,dict): raise ValueError('YAML must contain a mapping')
    device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    run(cfg, args.run_id, device, mode=args.mode, overwrite=args.overwrite)


if __name__=='__main__':
    main()
