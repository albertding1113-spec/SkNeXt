"""Backward-compatible, explicitly verified SkNeXt checkpoint restoration.

Missing *metadata* is tolerated. Missing or incompatible model tensors require
opt-in partial loading, because inference with random output weights is unsafe.
The absence of a trusted, complete training state means warm-start, NOT resume.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import copy
import random
import warnings

import numpy as np
import torch


MODEL_KEYS = ("model_state_dict", "state_dict", "model_weights", "network_state_dict",
              "weights", "model", "net")


def read_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load a trusted local checkpoint using PyTorch's safer weights-only mode."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load checkpoint {path} in weights-only mode. Legacy checkpoints "
            "containing custom Python objects must be converted in a trusted "
            "environment; unsafe pickle loading is deliberately disabled."
        ) from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, got {type(data).__name__}")
    return data


def _is_weights(mapping: Any) -> bool:
    return (isinstance(mapping, Mapping) and bool(mapping) and
            all(isinstance(k, str) and isinstance(v, torch.Tensor)
                for k, v in mapping.items()))


def extract_weights(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Accept bare model state_dicts or commonly used wrapper key names."""
    if _is_weights(checkpoint):
        return checkpoint
    for name in MODEL_KEYS:
        if name in checkpoint and _is_weights(checkpoint[name]):
            return checkpoint[name]
    raise ValueError("Checkpoint does not contain a valid tensor model state_dict "
                     f"(accepted wrapper keys: {MODEL_KEYS}).")


def _normalize_prefixes(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(weights)
    for prefix in ("module.", "_orig_mod."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix):]: value for key, value in result.items()}
    return result


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    expected_architecture: str | None = None,
    expected_channels: list[str] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Load weights with precise compatibility checks and an actionable report.

    Missing metadata is not treated as an error. When partial loading is enabled,
    only matching tensor names/shapes are loaded; all omissions are warned.
    """
    if expected_architecture is not None:
        saved_arch = checkpoint.get("model_name", checkpoint.get("architecture"))
        if saved_arch is None:
            warnings.warn("Checkpoint lacks architecture metadata; verifying weight tensors instead.",
                          RuntimeWarning, stacklevel=2)
        else:
            canon = lambda x: str(x).lower().replace("-", "").replace("_", "")
            # A legacy semantic-only U_NeXt_V2_semantic_only is architecturally U-NeXt V2.
            if canon(saved_arch).removesuffix("semanticonly") != canon(expected_architecture):
                raise ValueError(f"Checkpoint architecture {saved_arch!r} differs from "
                                 f"configured {expected_architecture!r}")
    if expected_channels is not None:
        saved_channels = checkpoint.get("output_channel_names", checkpoint.get("semantic_channels"))
        if saved_channels is None:
            warnings.warn("Checkpoint lacks channel-order metadata. Confirm TASK.CHANNELS "
                          "matches training, especially if heads have the same shape.",
                          RuntimeWarning, stacklevel=2)
        elif list(saved_channels) != list(expected_channels):
            raise ValueError(f"Checkpoint channel order {list(saved_channels)} differs from "
                             f"configured {list(expected_channels)}")
    target = model.module if hasattr(model, "module") else model
    weights = _normalize_prefixes(extract_weights(checkpoint))
    current = target.state_dict()
    mismatched = {key: (tuple(value.shape), tuple(current[key].shape))
                  for key, value in weights.items()
                  if key in current and tuple(value.shape) != tuple(current[key].shape)}
    unexpected = sorted(set(weights) - set(current))
    missing = sorted(set(current) - set(weights))
    if (missing or unexpected or mismatched) and not allow_partial:
        raise RuntimeError("Model weights are incompatible. "
                           f"Missing keys ({len(missing)}): {missing[:8]}; "
                           f"unexpected ({len(unexpected)}): {unexpected[:8]}; "
                           f"shape differences ({len(mismatched)}): "
                           f"{dict(list(mismatched.items())[:8])}. "
                           "Set MODEL.CHECKPOINT_ALLOW_PARTIAL: true only if intentionally "
                           "initializing missing layers from scratch.")
    if allow_partial:
        filtered = {k: v for k, v in weights.items() if k in current and k not in mismatched}
        if not filtered:
            raise RuntimeError("No model weights match; refusing to run with a fully random model")
        if missing or unexpected or mismatched:
            warnings.warn("PARTIAL CHECKPOINT LOAD: "
                          f"{len(missing)} absent keys, {len(unexpected)} extra keys, "
                          f"{len(mismatched)} shape mismatches. Unloaded model layers "
                          "retain random initialization; inference quality is not validated.",
                          RuntimeWarning, stacklevel=2)
        target.load_state_dict(filtered, strict=False)
    else:
        filtered = weights
        target.load_state_dict(filtered, strict=True)
    return {"loaded": len(filtered), "missing": missing,
            "unexpected": unexpected, "shape_mismatch": mismatched,
            "partial": bool(missing or unexpected or mismatched)}


def _restore_rng(state: Mapping[str, Any]) -> None:
    """Restore available RNG fields; never require them for continuation."""
    if state.get('rng_by_rank'):
        from sknext.utils.execution import rank, world_size
        states = state['rng_by_rank']
        if len(states) != world_size():
            warnings.warn('GPU count changed; stochastic training will not be bitwise reproducible.',
                          RuntimeWarning, stacklevel=2)
        state = states[rank() % len(states)]
    if state.get("rng_python") is not None:
        random.setstate(state["rng_python"])
    if state.get("rng_numpy") is not None:
        rng = state["rng_numpy"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32),
                             rng[2], rng[3], rng[4]))
    if state.get("rng_torch") is not None:
        torch.set_rng_state(state["rng_torch"])
    if torch.cuda.is_available() and state.get("rng_cuda"):
        if state.get('rng_cuda_local'):
            torch.cuda.set_rng_state(state['rng_cuda'][0])
        else:
            torch.cuda.set_rng_state_all(state["rng_cuda"])


def training_checkpoint_paths(input_path: str | Path) -> tuple[Path, Path]:
    """Resolve (latest, best) whether the supplied file is a best or latest checkpoint."""
    path = Path(input_path)
    if path.stem.endswith('_latest'):
        latest = path
        best = path.with_name(path.stem[:-len('_latest')] + path.suffix)
    else:
        best = path
        latest = path.with_name(path.stem + '_latest' + path.suffix)
    return latest, best


def restore_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    latest_path: str | Path,
    best_path: str | Path,
    identity: str,
    epochs: int,
    expected_architecture: str | None = None,
    expected_channels: list[str] | None = None,
    allow_partial: bool = False,
    scaler: Any = None,
) -> dict[str, Any]:
    """Resume exactly where possible; otherwise warm-start from available weights.

    If checkpoint is incomplete, optimizer/scheduler/scaler and epoch stay fresh.
    If the checkpoint carries a *different* identity, fail rather than accidentally
    resuming with changed data or hyperparameters. RNG values are optional and
    warnings describe any consequent loss of bitwise reproducibility.
    """
    latest_path, best_path = Path(latest_path), Path(best_path)
    selected = latest_path if latest_path.is_file() else best_path
    if not selected.is_file():
        raise FileNotFoundError(f"Neither resume checkpoint {latest_path} nor legacy "
                                f"best weights {best_path} exists")
    state = read_checkpoint(selected)
    saved_id = state.get("identity")
    if selected == latest_path and saved_id is not None and saved_id != identity:
        raise ValueError("Training checkpoint/configuration/data identity mismatch: refusing "
                         "to resume or overwrite results with changed settings. Use a fresh run ID.")
    # Do not load optimizer state unless ALL key elements needed for exact
    # epoch/learning-rate continuity exist and the identity can be verified.
    complete = (selected == latest_path and saved_id == identity and
                all(state.get(k) is not None for k in
                    ("optimizer_state_dict", "scheduler_state_dict", "epoch")))
    model_report = load_model_weights(model, state,
                                      expected_architecture=expected_architecture,
                                      expected_channels=expected_channels,
                                      allow_partial=allow_partial)
    if model_report["partial"]:
        complete = False   # random-initialized layers invalidate optimizer continuity
    if complete:
        epoch = int(state["epoch"])
        if not 0 <= epoch <= epochs:
            raise ValueError(f"Checkpoint epoch {epoch} is invalid for {epochs} total epochs")
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        scheduler_snapshot = copy.deepcopy(scheduler.state_dict())
        scaler_snapshot = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            if scaler is not None and state.get("scaler_state_dict") is not None:
                scaler.load_state_dict(state["scaler_state_dict"])
            elif scaler is not None and scaler.is_enabled():
                warnings.warn("Checkpoint has no AMP scaler state; using a fresh GradScaler.",
                              RuntimeWarning, stacklevel=2)
        except Exception as exc:
            optimizer.load_state_dict(optimizer_snapshot)
            scheduler.load_state_dict(scheduler_snapshot)
            if scaler is not None:
                scaler.load_state_dict(scaler_snapshot)
            complete = False
            warnings.warn(f"Optimizer/scheduler restoration failed ({exc}); "
                          "falling back to weight-only warm start.", RuntimeWarning, stacklevel=2)
    if not complete:
        warnings.warn(f"Checkpoint {selected.name} has no verified complete training "
                      "state; model weights loaded, optimizer/scheduler start fresh "
                      "at epoch 0. This is not an exact resume.",
                      RuntimeWarning, stacklevel=2)
        return {"mode": "warm_start", "epoch": 0, "source": str(selected),
                "best_val_loss": float("inf"), "best_val_metrics": {},
                "epochs_without_improvement": 0, "chart_history": {},
                "weights": model_report}
    if any(key not in state for key in ("rng_torch", "rng_numpy", "rng_python")):
        warnings.warn("Checkpoint is missing RNG states; resumed optimization is not "
                      "bitwise reproducible.", RuntimeWarning, stacklevel=2)
    else:
        try:
            _restore_rng(state)
        except Exception as exc:
            warnings.warn(f"Could not restore all RNG states ({exc}); continuing with "
                          "current RNG state.", RuntimeWarning, stacklevel=2)
    return {"mode": "resume", "epoch": epoch, "source": str(selected),
            "best_val_loss": float(state.get("best_val_loss", state.get("best", float("inf")))),
            "best_val_metrics": dict(state.get("best_val_metrics") or {}),
            "epochs_without_improvement": int(state.get("epochs_without_improvement", state.get("stagnant", 0))),
            "chart_history": dict(state.get("chart_history") or {}),
            "weights": model_report}
