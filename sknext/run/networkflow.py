from pathlib import Path
import torch
import os
import re
import math
import shutil
import traceback
from datetime import datetime
import zarr
from numcodecs import Blosc
import matplotlib
import json
import hashlib
import random
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Iterable, Sequence
from sknext.run.base_workflow import Base_Workflow
from sknext.data.imageIO import get_tif_path_in_folder, get_tif_path_dict_in_folder, pair_raw_and_label_tiffs, patch_coordinates_iter, get_coord_after_padding, central_block_coordinates_iter
from sknext.data.datasetIO import tif_list_to_zarr, detect_path_type, IMSReader, ZarrIOManager
from sknext.data.data_loader import ZarrPatchLoader
from sknext.data.label import watershed_with_sk
from sknext.data.preprocessing import reflect_padding_img, preprocess_img
from sknext.data.postprocessing import post_processing_instance, post_processing_semantic
from sknext.model.unext_v2 import U_NeXt_V2
from sknext.model.scheduler import build_onecycle_scheduler, build_warmup_cosine_scheduler
from sknext.utils.utils import get_cfg_value, time_str
from sknext.model.loss import instance_segmentation_loss, instance_segmentation_metrics
from sknext.utils.model_info import print_model_parameters, profile_forward_memory
from sknext.utils.checkpoint import (read_checkpoint, load_model_weights,
                                     restore_training_checkpoint, training_checkpoint_paths)


class Segmentation_Workflow(Base_Workflow):
    """Train a U-NeXt model and infer with or without SWC-based segmentation.

    Extends shared configuration handling with patch preparation, checkpoint
    I/O, loss/metric tracking, overlap blending, and OME-Zarr output.
    """
    def __init__(self, cfg, device:torch.device, job_id:int):
        """Initialize shared settings and epoch chart tracking for a segmentation run.

        cfg supplies workflow options, device selects model execution, and job_id
        identifies outputs. Creates the configured chart directory.
        """
        super(Segmentation_Workflow, self).__init__(cfg, device, job_id)
        self.best_val_loss = float("inf")
        # Store the same scalar values that are written to SummaryWriter.
        # Structure: {"Loss": {"train_loss": [(epoch, value), ...], "val_loss": [(epoch, value), ...]}}
        self._chart_history: dict[str, dict[str, list[tuple[int, float]]]] = {}
        Path(self.chart_dir).mkdir(parents=True, exist_ok=True)
        print(f"{time_str()} Init Segmentation.", flush=True)

    def _log_scalars(self, main_tag: str, scalars: dict[str, float], epoch: int):
        """Write scalars to TensorBoard and retain the same values for plotting."""
        scalar_values = {name: float(value) for name, value in scalars.items()}
        self.log_writer.add_scalars(main_tag, scalar_values, epoch)
        tag_history = self._chart_history.setdefault(main_tag, {})
        for series_name, value in scalar_values.items():
            tag_history.setdefault(series_name, []).append((int(epoch), value))

    @staticmethod
    def _chart_filename(main_tag: str) -> str:
        """Convert a TensorBoard tag into a safe PNG filename."""
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", main_tag.strip())
        safe_name = safe_name.strip("._") or "chart"
        return f"{safe_name}.png"

    def save_log_charts(self, current_epoch: int):
        """Save one line chart for each scalar group written through ``_log_scalars``."""
        self.log_writer.flush()
        chart_dir = Path(self.chart_dir)
        chart_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        for main_tag, series_dict in self._chart_history.items():
            if not series_dict: continue
            fig, ax = plt.subplots(figsize=(9, 6))
            has_data = False
            for series_name, points in series_dict.items():
                if not points: continue
                # Keep the final value if the same epoch is logged more than once.
                values_by_epoch = {epoch: value for epoch, value in points}
                epochs = sorted(values_by_epoch)
                values = [values_by_epoch[epoch] for epoch in epochs]
                ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.5, label=series_name)
                has_data = True
            if not has_data:
                plt.close(fig)
                continue
            ax.set_title(f"{main_tag} (through epoch {current_epoch})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(main_tag)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            fig.tight_layout()
            chart_path = chart_dir / self._chart_filename(main_tag)
            fig.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(chart_path)

    def save_patch_as_zarr(self):
        """Create paired training and validation stores from configured TIFF directories.

        Resolves instance/semantic label files, crops patches, and writes raw and
        label arrays. Patch filtering applies to training data only.
        """
        self.train_tif_list = get_tif_path_in_folder(self.train_path)
        self.train_gt_tif_dict = get_tif_path_dict_in_folder(self.train_gt_path, self.channels)
        self.train_tif_list, self.train_gt_tif_dict = pair_raw_and_label_tiffs(self.train_tif_list, self.train_gt_tif_dict)
        self.val_tif_list = get_tif_path_in_folder(self.val_path)
        self.val_gt_tif_dict = get_tif_path_dict_in_folder(self.val_gt_path, self.channels)
        self.val_tif_list, self.val_gt_tif_dict = pair_raw_and_label_tiffs(self.val_tif_list, self.val_gt_tif_dict)
        self.train_zarr_path = Path(self.train_path).parent / "train.ome.zarr"
        self.val_zarr_path = Path(self.val_path).parent / "val.ome.zarr"
        tif_list_to_zarr(self.train_tif_list, self.train_gt_tif_dict, self.train_zarr_path, self.patch_size,
                         self.train_overlap, self.train_padding, self.preprocess_dict, self.channels,
                         self.channels_extra_opts, filter_dict=self.filter_dict, normalize_per_patch=True)
        # Validation data is intentionally not filtered, otherwise the validation
        # distribution and reported metrics would be biased toward retained patches.
        tif_list_to_zarr(self.val_tif_list, self.val_gt_tif_dict, self.val_zarr_path, self.patch_size,
                         self.val_overlap, self.val_padding, self.preprocess_dict, self.channels,
                         self.channels_extra_opts, filter_dict=None, normalize_per_patch=True)

    def define_activations_and_channels(self):
        """Expand task channels into model output names and initialize linear heads.

        Affinity targets expand into separate Z, Y, and X offset channels.
        Stores the output channel count, names, and activation identifiers.
        """
        self.out_channel_num = 0
        self.output_channel_names = []
        for channel in self.channels:
            if channel == "A":
                a_opts = self.channels_extra_opts.get("A", {"z_affinities": [1], "y_affinities": [1], "x_affinities": [1]})
                z_aff = a_opts.get("z_affinities", [1])
                y_aff = a_opts.get("y_affinities", [1])
                x_aff = a_opts.get("x_affinities", [1])
                assert len(z_aff) == len(y_aff) == len(x_aff), "z/y/x affinities should have the same length."
                self.out_channel_num += 3 * len(z_aff)
                self.output_channel_names += [f"A_z_{d}" for d in z_aff] + [f"A_y_{d}" for d in y_aff] + [f"A_x_{d}" for d in x_aff]
            else:
                self.out_channel_num += 1
                self.output_channel_names += [channel]
        self.head_activations = list(["linear"] * self.out_channel_num)
        self.instance_indices = [i for i, name in enumerate(self.output_channel_names)
                                 if not name.startswith("S.")]
        self.semantic_indices = [i for i, name in enumerate(self.output_channel_names)
                                 if name.startswith("S.")]
        if not self.output_channel_names or len(set(self.output_channel_names)) != len(self.output_channel_names):
            raise ValueError("Model output channel names must be nonempty and unique")

    def set_loss_metrics(self):
        """Create the weighted BCE/Dice loss and channel-wise IoU evaluator.

        Requires output_channel_names from define_activations_and_channels and
        moves the loss module to the configured device.
        """
        self.loss_func = instance_segmentation_loss(channel_weights=self.channel_weights,
                                                    channel_names=self.output_channel_names,
                                                    bce_weight=1.0,
                                                    dice_weight=1.0,
                                                    auto_balance=True)
        self.loss_func.to(self.device)
        self.metric = instance_segmentation_metrics(self.output_channel_names)

    def set_optimizer_scheduler(self):
        """Build the configured SGD/Adam/AdamW optimizer and step-based LR scheduler.

        Requires an initialized model and train_steps_per_epoch. Unsupported
        optimizer or scheduler names fail assertions.
        """
        assert self.optimizer_name in ["SGD", "ADAM", "ADAMW"], "Unknown optimizer."
        if self.optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.w_decay)
        elif self.optimizer_name == "ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.opt_betas,weight_decay=self.w_decay)
        elif self.optimizer_name == "ADAMW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=self.opt_betas,weight_decay=self.w_decay)
        assert self.lr_scheduler_name in ["one_cycle", "warmupcosine"], "Unknown scheduler."
        if self.lr_scheduler_name == "one_cycle":
            self.scheduler = build_onecycle_scheduler(self.optimizer,
                                                      max_lr=self.lr,
                                                      epochs=self.epochs,
                                                      steps_per_epoch=self.train_steps_per_epoch,
                                                      pct_start=0.1)
        elif self.lr_scheduler_name == "warmupcosine":
            self.scheduler = build_warmup_cosine_scheduler(self.optimizer,
                                                           epochs=self.epochs,
                                                           steps_per_epoch=self.train_steps_per_epoch,
                                                           warmup_epochs=self.warmup_cosine_decay_epochs,
                                                           min_lr=self.min_lr)

    def set_model(self):
        """Construct U-NeXt V2 with configured geometry and task output channels.

        Moves the model to the selected device and chooses train/eval mode from
        the workflow flags. Only the supported U-NeXt V2 architecture aliases are accepted.
        """
        self.define_activations_and_channels()
        assert self.model_name in ["unext_v2", "u_next_v2", "unext-v2"], f"Unsupported model: {self.model_name}"
        self.model = U_NeXt_V2(image_shape=self.patch_size,
                               feature_maps=self.feature_maps,
                               upsample_layer="convtranspose",
                               z_down=self.z_down,
                               yx_down=self.yx_down,
                               output_channels=[self.out_channel_num],
                               separated_decoders=False,
                               output_channel_info=["+".join(self.output_channel_names)],
                               explicit_activations=False,
                               head_activations=list(self.head_activations),
                               stochastic_depth_prob=self.sd_prob,
                               cn_layers=self.convnext_layers,
                               isotropy=self.isotropy,
                               stem_k_size=self.convnext_stem_k_size,
                               contrast=False,
                               return_one_tensor=False)
        self.model.to(self.device)
        if self.cfg.TRAIN.ENABLE:
            self.model.train()
        else:
            self.model.eval()

    def create_skeleton(self):
        """Load SWCs only when skeleton-guided inference is explicitly enabled."""
        from sknext.skeleton.skeleton import SkeletonManager
        self.skeleton_manager = SkeletonManager(self.skeleton_path)

    @property
    def checkpoint_input_file(self):
        """The existing file to read; never redirects saved checkpoints."""
        override = get_cfg_value(self.cfg, "MODEL.CHECKPOINT_INPUT_PATH", "")
        return Path(override).expanduser() if override else Path(self.checkpoint_file)

    def load_checkpoint(self):
        """Load old or new model checkpoints for inference or fresh fine-tuning.

        Optional checkpoint metadata never blocks weight loading. Tensor mismatches
        require MODEL.CHECKPOINT_ALLOW_PARTIAL=True and are reported explicitly.
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Call set_model() before load_checkpoint()")
        path = self.checkpoint_input_file
        state = read_checkpoint(path)
        report = load_model_weights(
            self.model, state,
            expected_architecture=self.model_name,
            expected_channels=list(self.output_channel_names),
            allow_partial=bool(get_cfg_value(self.cfg, "MODEL.CHECKPOINT_ALLOW_PARTIAL", False)))
        self.model.to(self.device)
        print(f"{time_str()} Loaded {report['loaded']} tensors from {path} "
              f"(partial={report['partial']})", flush=True)
        return report

    @staticmethod
    def _atomic_torch_save(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(data, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists(): tmp.unlink()

    def _training_identity(self):
        """Hash the model, data, optimization and schedule contract for exact resume."""
        # Loading policy is not part of model architecture. Removing this newly
        # introduced key retains exact-resume identities of older v2 checkpoints.
        identity_model_cfg = self.cfg.MODEL.clone()
        for key in ("CHECKPOINT_ALLOW_PARTIAL", "CHECKPOINT_INPUT_PATH"):
            if key in identity_model_cfg:
                del identity_model_cfg[key]
        fields = {"model": identity_model_cfg, "task": self.cfg.TASK,
                  "normalization": self.cfg.DATA.NORMALIZATION,
                  "patch_size": list(self.patch_size),
                  "train": {"optimizer": self.optimizer_name, "lr": self.lr,
                            "weight_decay": self.w_decay, "betas": list(self.opt_betas),
                            "epochs": self.epochs, "batch_size": self.batch_size,
                            "scheduler": self.cfg.TRAIN.LR_SCHEDULER},
                  "training_inputs": [str(Path(self.train_path).resolve()),
                                      str(Path(self.train_gt_path).resolve()),
                                      str(Path(self.val_path).resolve()),
                                      str(Path(self.val_gt_path).resolve())],
                  "train_overlap": list(map(int, self.train_overlap)),
                  "val_overlap": list(map(int, self.val_overlap)),
                  "train_padding": list(self.train_padding), "val_padding": list(self.val_padding),
                  "filter": self.filter_dict, "augmentor": self.cfg.AUGMENTOR,
                  "training_patches": self.train_patches,
                  "validation_patches": self.val_patches,
                  "input_file_signatures": [
                      (str(Path(path).resolve()), Path(path).stat().st_size,
                       Path(path).stat().st_mtime_ns)
                      for group in (self.train_tif_list, self.val_tif_list,
                                    *self.train_gt_tif_dict.values(),
                                    *self.val_gt_tif_dict.values())
                      for path in group
                  ]}
        # YACS nodes stringify deterministically without relying on pickle.
        blob = json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def latest_checkpoint_file(self):
        path = Path(self.checkpoint_file)
        return path.with_name(path.stem + "_latest.pth")

    def save_checkpoint(self):
        """Atomically write best weights for inference and warm-start compatibility."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        self._atomic_torch_save({"model_state_dict": weights,
                                 "model_name": self.model_name,
                                 "output_channel_names": self.output_channel_names}, self.checkpoint_file)
        print(f"{time_str()} Best model saved to {self.checkpoint_file}", flush=True)

    def save_training_state(self, epoch, best_val_metrics, epochs_without_improvement):
        """Persist an exact epoch-boundary continuation state, separately from best weights."""
        numpy_rng = np.random.get_state()
        model = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "format_version": 2, "epoch": int(epoch), "identity": self._training_identity(),
            "model_name": self.model_name, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": float(self.best_val_loss),
            "best_val_metrics": dict(best_val_metrics),
            "epochs_without_improvement": int(epochs_without_improvement),
            "chart_history": self._chart_history,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "rng_python": random.getstate(),
            "rng_numpy": (numpy_rng[0], numpy_rng[1].tolist(),
                          numpy_rng[2], numpy_rng[3], numpy_rng[4]),
        }
        self._atomic_torch_save(state, self.latest_checkpoint_file)

    def load_training_state(self):
        """Resume an exact epoch state or fall back to legacy weight-only warm start.

        Full restoration requires compatible optimizer, scheduler, epoch and a
        matching training identity. If these are missing, restart training at
        epoch zero with fresh optimizer/scheduler instead of pretending to resume.
        """
        latest, best = training_checkpoint_paths(self.checkpoint_input_file)
        result = restore_training_checkpoint(
            self.model, self.optimizer, self.scheduler,
            latest_path=latest,
            best_path=best,
            identity=self._training_identity(), epochs=self.epochs,
            expected_architecture=self.model_name,
            expected_channels=list(self.output_channel_names),
            allow_partial=bool(get_cfg_value(self.cfg, "MODEL.CHECKPOINT_ALLOW_PARTIAL", False)))
        self.best_val_loss = result["best_val_loss"]
        self._chart_history = result["chart_history"]
        print(f"{time_str()} Training checkpoint: {result['mode']}; "
              f"source={result['source']}; starting epoch={result['epoch'] + 1}",
              flush=True)
        return (result["epoch"], result["best_val_metrics"],
                result["epochs_without_improvement"])

    def create_reader(self):
        """Open the configured inference source as an OME-Zarr or Imaris reader.

        Computes output chunk dimensions from patch size and stores infer_reader.
        Unsupported source formats raise FileNotFoundError.
        """
        self.chunk = (1, self.patch_size[0]*2, self.patch_size[1]*4, self.patch_size[2]*4)
        self.infer_path = Path(self.infer_path)
        assert self.infer_path.exists(), "Infer path does not exist."
        file_type = detect_path_type(self.infer_path)
        if file_type == "zarr":
            self.infer_reader = ZarrIOManager(self.infer_path, mode="r")
        elif file_type == "ims":
            self.infer_reader = IMSReader(self.infer_path)
        else: raise FileNotFoundError(f"file type {file_type} is not supported.")

    def _resolve_infer_output_paths(self):
        """Resolve separate segmentation and probability OME-Zarr destinations."""
        segmentation = Path(self.infer_gt_path)
        if segmentation.suffix.lower() != ".zarr":
            segmentation = segmentation / f"result{self.job_id}.ome.zarr"

        requested_raw = str(self.cfg.DATA.INFER.RAW_OUTPUT_PATH).strip()
        if requested_raw:
            raw = Path(requested_raw)
            if raw.suffix.lower() != ".zarr":
                raw = raw / f"result{self.job_id}_raw.ome.zarr"
        elif segmentation.name.lower().endswith(".ome.zarr"):
            raw = segmentation.with_name(segmentation.name[:-len(".ome.zarr")] + "_raw.ome.zarr")
        else:
            raw = segmentation.with_name(segmentation.stem + "_raw.zarr")

        if segmentation.resolve() == raw.resolve():
            raise ValueError("uint16 and float32 output paths must be different OME-Zarr stores.")
        if self.save_segmentation and segmentation.resolve() == self.infer_path.resolve():
            raise ValueError("The uint16 output path must not overwrite the inference input.")
        if self.save_raw_predictions and raw.resolve() == self.infer_path.resolve():
            raise ValueError("The float32 output path must not overwrite the inference input.")
        self.infer_gt_path = segmentation
        self.infer_raw_path = raw

    def _configure_infer_output(self):
        """Determine output destinations without creating or modifying stores."""
        self.output_mode = str(self.cfg.INFER.OUTPUT_MODE).lower().strip()
        if self.output_mode not in {"uint16", "float32", "both"}:
            raise ValueError("INFER.OUTPUT_MODE must be 'uint16', 'float32', or 'both'.")
        self.save_segmentation = self.output_mode in {"uint16", "both"}
        self.save_raw_predictions = self.output_mode in {"float32", "both"}
        self.segmentation_dtype = (
            "uint32" if (self.save_segmentation and self.instance_num and not self.skeleton_flag)
            else "uint16"
        )
        self._resolve_infer_output_paths()

    def create_writer(self):
        """Create selected label and/or float32 OME-Zarr output stores.

        Float32 channels correspond one-to-one to ``output_channel_names``
        (F/P/C/affinities/S.*) and contain overlap-blended sigmoid probabilities.
        """
        self.infer_writer = None
        self.infer_raw_writer = None
        self._configure_infer_output()

        spatial_shape = tuple(int(x) for x in self.infer_reader.shape[1:4])
        if self.save_segmentation:
            self.infer_writer = ZarrIOManager(
                self.infer_gt_path,
                shape=(self.postpro_channel_num, *spatial_shape),
                dtype=self.segmentation_dtype,
                chunks=self.chunk,
                mode="a",
                channel_names=self.postpro_channel_info,
            )
            self.infer_writer.root.attrs["sknext_instance_id_mode"] = (
                "skeleton_index_plus_one" if self.skeleton_flag else "global_automatic_watershed_uint32"
            )
        if self.save_raw_predictions:
            self.infer_raw_writer = ZarrIOManager(
                self.infer_raw_path,
                shape=(self.out_channel_num, *spatial_shape),
                dtype="float32",
                chunks=self.chunk,
                mode="a",
                channel_names=self.output_channel_names,
                channel_windows=[(0.0, 1.0)] * self.out_channel_num,
                image_name="SkNeXt sigmoid probability maps",
            )
            # Distinguish probabilities from pre-sigmoid logits in stored metadata.
            self.infer_raw_writer.root.attrs["sknext_prediction"] = {
                "kind": "sigmoid_probability",
                "dtype": "float32",
                "channel_names": list(self.output_channel_names),
                "overlap_blending": "weighted_mean",
                "value_range": [0.0, 1.0],
            }

    def define_postprocess_channels(self):
        """Derive output layout from explicit model-channel indices (never slices)."""
        if not hasattr(self, "output_channel_names"):
            self.define_activations_and_channels()
        self.instance_num = len(self.instance_indices)
        self.semantic_num = len(self.semantic_indices)
        self.postpro_channel_info = (["Infer_instance"] if self.instance_num else []) + [
            self.output_channel_names[i][2:] for i in self.semantic_indices
        ]
        self.postpro_channel_num = len(self.postpro_channel_info)

    def validate_watershed_config(self):
        """Fail before opening output stores if watershed would be unconstrained."""
        if not self.save_segmentation or not self.instance_num:
            return
        available = [self.output_channel_names[i] for i in self.instance_indices]
        if not isinstance(self.watershed_topographic_channel, str) or self.watershed_topographic_channel not in available:
            raise ValueError(f"Watershed TOPOGRAPHIC_SURFACE_CHANNEL={self.watershed_topographic_channel!r} "
                             f"is not among instance channels {available}")
        for name, choices, thresholds in (
            ("SEED_CHANNELS", self.watershed_seed_channels, self.watershed_seed_channels_thresh),
            ("GROWTH_MASK_CHANNELS", self.watershed_growth_mask_channels, self.watershed_growth_mask_channels_thresh),
        ):
            if not isinstance(choices, (list, tuple)):
                raise TypeError(f"Watershed {name} must be a list")
            if len(set(choices)) != len(choices) or any(ch not in available for ch in choices):
                raise ValueError(f"Watershed {name}={choices!r} contains duplicated or unavailable channels; "
                                 f"available={available}")
            if thresholds and len(thresholds) != len(choices):
                raise ValueError(f"Watershed {name}_THRESH requires one threshold per channel")
            for threshold in thresholds:
                if threshold != "auto":
                    try: value = float(threshold)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"Invalid watershed threshold {threshold!r}") from error
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Watershed threshold outside [0,1]: {threshold!r}")
        if not self.watershed_growth_mask_channels:
            raise ValueError("Watershed GROWTH_MASK_CHANNELS must be nonempty to prevent background growth")
        if not self.skeleton_flag and not self.watershed_seed_channels:
            raise ValueError("SWC-free watershed requires SEED_CHANNELS")

    def close_reader_writer(self):
        """Close initialized OME-Zarr writer(s) and the inference reader."""
        for name in ("infer_writer", "infer_raw_writer", "infer_reader"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()

    def train(self):
        """Prepare patch stores, optimize the model, and validate after each epoch.

        Optionally loads existing model weights, records losses/metrics and LR,
        exports charts every five epochs, saves improved validation checkpoints,
        and stops after the configured patience without improvement.
        """
        print(f"{time_str()} preparing training data", flush=True)
        self.save_patch_as_zarr()
        print(f"{time_str()} preparing data loader", flush=True)
        self.train_patch_loader = ZarrPatchLoader(self.cfg, self.train_zarr_path, if_val=False)
        self.val_patch_loader = ZarrPatchLoader(self.cfg, self.val_zarr_path, if_val=True)
        self.train_patches = len(self.train_patch_loader)
        self.val_patches = len(self.val_patch_loader)
        self.train_steps_per_epoch = math.ceil(self.train_patches / self.batch_size)
        self.val_steps_per_epoch = math.ceil(self.val_patches / self.batch_size)
        print(f"{time_str()} preparing model", flush=True)
        self.set_model()
        print_model_parameters(self.model)
        profile_forward_memory(self.model, input_shape=(self.batch_size,self.patch_size[-1],*self.patch_size[:-1],), device=self.device)
        torch.cuda.empty_cache()
        print(f"{time_str()} preparing optimizer and scheduler", flush=True)
        self.set_optimizer_scheduler()
        print(f"{time_str()} preparing loss function and metrics", flush=True)
        self.set_loss_metrics()
        resume = bool(self.cfg.TRAIN.RESUME)
        if resume and self.load_checkpoint_flag:
            raise ValueError("TRAIN.RESUME and MODEL.LOAD_CHECKPOINT are mutually exclusive")
        if resume:
            start_epoch, best_val_metrics, epochs_without_improvement = self.load_training_state()
        else:
            start_epoch, best_val_metrics, epochs_without_improvement = 0, {}, 0
            if self.load_checkpoint_flag:
                self.load_checkpoint()
        best_val_loss = self.best_val_loss

        shuffle_train = bool(get_cfg_value(self.cfg,"AUGMENTOR.SHUFFLE_TRAIN_DATA_EACH_EPOCH",True,))
        shuffle_val = bool(get_cfg_value(self.cfg,"AUGMENTOR.SHUFFLE_VAL_DATA_EACH_EPOCH",False,))
        print(f"{time_str()} patch order: train_shuffle={shuffle_train}, val_shuffle={shuffle_val}", flush=True)

        for epoch in range(start_epoch, self.epochs):
            epoch_number = epoch + 1
            print(f"{time_str()} [EPOCH{epoch_number:04d}]", flush=True)
            # Generate a fresh permutation at the beginning of every epoch.
            # Every training patch is still used exactly once per epoch.
            train_indices = self.train_patch_loader.get_epoch_indices(shuffle=shuffle_train,)
            self.model.train()
            epoch_train_loss = 0
            epoch_train_metrics = {}  # {"F" 0.0:, "P": 0.0}
            epoch_train_patches = 0
            for step in range(self.train_steps_per_epoch):
                # clear optimizer grad before starting everything
                self.optimizer.zero_grad()
                start_ = step * self.batch_size
                end_ = min((step + 1) * self.batch_size, self.train_patches)
                batch_indices = train_indices[start_:end_]
                img, mask = self.train_patch_loader[batch_indices]
                img = img.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                pred = self.model(img)
                step_loss = self.loss_func(pred, mask)
                step_metrics = self.metric(pred, mask)
                step_patches = end_ - start_
                # update epoch loss and metrics
                epoch_train_loss += float(step_loss.item()) * step_patches
                for key, value in step_metrics.items():
                    epoch_train_metrics[key] = epoch_train_metrics.get(key, 0.0) + float(value) * step_patches
                epoch_train_patches += step_patches
                # print step loss and metrics
                if (step+1)%math.ceil(self.train_steps_per_epoch/20) == 0:
                    metric_str = " ".join(f"{key} channel: {value:.6f}," for key, value in step_metrics.items())
                    print(f"{time_str()} [EPOCH{epoch_number:04d}] [TRAIN] [{epoch_train_patches:06d}/{self.train_patches:06d}], loss: {step_loss: 6f}, {metric_str}, lr: {self.optimizer.param_groups[0]['lr']:.6f}", flush=True)
                # backward
                step_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            # calculate epoch loss and metrics
            epoch_train_loss = epoch_train_loss / epoch_train_patches
            for key, value in epoch_train_metrics.items():
                epoch_train_metrics[key] = epoch_train_metrics.get(key, 0.0) / epoch_train_patches
            # print epoch loss and metrics
            metric_str = " ".join(f"{key} channel: {value:.6f}," for key, value in epoch_train_metrics.items())
            print(f"{time_str()} [EPOCH{epoch_number:04d}] [TRAIN] loss: {epoch_train_loss:.6f}, {metric_str}", flush=True)

            # validation
            val_indices = self.val_patch_loader.get_epoch_indices(shuffle=shuffle_val,)
            self.model.eval()
            epoch_val_loss = 0
            epoch_val_metrics = {}
            epoch_val_patches = 0
            with torch.no_grad():
                for step in range(self.val_steps_per_epoch):
                    start_ = step * self.batch_size
                    end_ = min((step + 1) * self.batch_size, self.val_patches)
                    batch_indices = val_indices[start_:end_]
                    img, mask = self.val_patch_loader[batch_indices]
                    img = img.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)
                    pred = self.model(img)
                    step_loss = self.loss_func(pred, mask)
                    step_metrics = self.metric(pred, mask)
                    step_patches = end_ - start_
                    # update epoch loss and metrics
                    epoch_val_loss += float(step_loss.item()) * step_patches
                    for key, value in step_metrics.items():
                        epoch_val_metrics[key] = epoch_val_metrics.get(key, 0.0) + float(value) * step_patches
                    epoch_val_patches += step_patches
                    # print step loss and metrics
                    if (step+1) % math.ceil(self.val_steps_per_epoch / 10) == 0:
                        metric_str = " ".join(f"{key} channel: {value:.6f}," for key, value in step_metrics.items())
                        print(f"{time_str()} [EPOCH{epoch_number:04d}] [VAL] [{epoch_val_patches:06d}/{self.val_patches:06d}], loss: {step_loss: 6f}, {metric_str}", flush=True)
            # calculate epoch loss and metrics
            epoch_val_loss = epoch_val_loss / epoch_val_patches
            for key, value in epoch_val_metrics.items():
                epoch_val_metrics[key] = epoch_val_metrics.get(key, 0.0) / epoch_val_patches
            # print epoch loss and metric
            metric_str = " ".join(f"{key} channel: {value:.6f}," for key, value in epoch_val_metrics.items())
            print(f"{time_str()} [EPOCH{epoch_number:04d}] [VAL] loss: {epoch_val_loss:6f}, {metric_str}", flush=True)
            # Write epoch-level values to TensorBoard and chart history.
            self._log_scalars("Loss",{"train_loss": epoch_train_loss, "val_loss": epoch_val_loss}, epoch_number)
            for key in epoch_train_metrics.keys():
                self._log_scalars(f"{key} channel",{"train_metric": epoch_train_metrics[key], "val_metric": epoch_val_metrics[key]}, epoch_number)
            self._log_scalars( "Learning Rate",{"lr": self.optimizer.param_groups[0]["lr"]},  epoch_number)
            # Export all current log charts every five completed epochs.
            if epoch_number % 5 == 0:
                self.save_log_charts(epoch_number)
            # update best_loss and patience
            if epoch_val_loss < best_val_loss:
                metric_str = " ".join(f"{key} channel: {value:.6f}," for key, value in epoch_val_metrics.items())
                print(f"{time_str()} Model improved from {best_val_loss:.6f} to {epoch_val_loss:.6f}, {metric_str}", flush=True)
                epochs_without_improvement = 0
                best_val_loss = epoch_val_loss
                self.best_val_loss = best_val_loss
                best_val_metrics = epoch_val_metrics
                self.save_checkpoint()
            else:
                epochs_without_improvement += 1
                print(f"{time_str()} no improvement, Patience: {epochs_without_improvement:04d}/{self.patience:04d}", flush=True)
            # Save last epoch even when validation did not improve (required for exact resume).
            self.save_training_state(epoch_number, best_val_metrics, epochs_without_improvement)
            if self.patience > 0 and epochs_without_improvement >= self.patience:
                print(f"{time_str()} Early stopping after epoch {epoch_number}. "
                      f"Best validation loss: {best_val_loss:.6f}", flush=True)
                return

    @torch.no_grad()
    def infer(self):
        """Run full-volume or SWC-selective inference with label/float32 outputs.

        Completed blocks are logged only after every selected output has been
        written, so that a resumed run regenerates an interrupted block in both
        stores. Float32-only mode does not perform watershed or postprocessing.
        """
        print(f"{time_str()} preparing model", flush=True)
        self.set_model()
        print_model_parameters(self.model)
        profile_forward_memory(self.model, input_shape=(self.batch_size, self.patch_size[-1], *self.patch_size[:-1],),
                               device=self.device)
        torch.cuda.empty_cache()
        self.load_checkpoint()
        self.model.eval()
        self.define_postprocess_channels()
        # Preflight the watershed before any reader or writer is opened.
        self.infer_path = Path(self.infer_path)
        self._configure_infer_output()
        self.validate_watershed_config()
        print(f"{time_str()} validating inference source, outputs and resume metadata", flush=True)
        self.create_reader()
        infer_completed = False
        try:
            coord_iter = central_block_coordinates_iter(self.infer_reader.shape[1:4], self.c_block_size)
            self.total_block_num = int(np.prod(np.ceil(
                np.array(self.infer_reader.shape[1:4], dtype="float64") / np.array(self.c_block_size, dtype="float64"))))
            self.raw_all_blocks = bool(self.cfg.INFER.RAW_ALL_BLOCKS) and self.save_raw_predictions
            self.infer_manifest = self._build_infer_manifest()
            # In SWC-free mode, this ID is advanced only after complete blocks.
            # A successful block commit writes the corresponding state to the log.
            self.next_instance_id = 1
            self.load_infer_log()
            if self.skeleton_flag and (self.save_segmentation or not self.raw_all_blocks):
                self.create_skeleton()
            self.create_writer()
            if self.save_segmentation and hasattr(self, "skeleton_manager"):
                self.infer_writer.root.attrs["sknext_swc_id_mapping"] = self.skeleton_manager.validate_instance_mapping()

            for block_num, c_coord in enumerate(coord_iter):
                if block_num < self.inferred_block_num:
                    continue
                # The *only* block halo is calculated in Base_Workflow from
                # DATA.INFER.BLOCK_FACTOR and DATA.INFER.BLOCK_CENTRAL_FACTOR.
                # Both prediction/watershed and cross-block ID matching use
                # these same coordinates; no separate segmentation halo exists.
                coord, coord_in_block = get_coord_after_padding(
                    self.infer_reader.shape[1:4], c_coord, self.block_padding)
                if hasattr(self, "skeleton_manager"):
                    cropped_skeleton = self.skeleton_manager.crop_skeletons(coord)
                else:
                    cropped_skeleton = []
                should_predict = (not self.skeleton_flag) or bool(len(cropped_skeleton)) or self.raw_all_blocks
                if should_predict:
                    prediction_mode = f"{len(cropped_skeleton)} skeletons" if self.skeleton_flag else "whole-volume, no SWC"
                    print(f"{time_str()} [BLOCK{block_num:07d}/{self.total_block_num:07d}] "
                          f"[INFER] predicting ({prediction_mode})", flush=True)
                    seg_result, raw_result = self._infer_one_block(coord, cropped_skeleton)
                    if self.save_segmentation and self.instance_num and not self.skeleton_flag:
                        # Relabel local watershed IDs to unique uint32 IDs, reusing
                        # a previous ID only for reliable matches in processed halos.
                        seg_result = self._stitch_instance_block(
                            seg_result, coord, coord_in_block, block_num)
                    central_slices = (
                        slice(None),
                        slice(int(coord_in_block[0, 0]), int(coord_in_block[0, 1])),
                        slice(int(coord_in_block[1, 0]), int(coord_in_block[1, 1])),
                        slice(int(coord_in_block[2, 0]), int(coord_in_block[2, 1])),
                    )
                    start = (0, int(c_coord[0, 0]), int(c_coord[1, 0]), int(c_coord[2, 0]))
                    if self.save_segmentation:
                        self.infer_writer.write_block(seg_result[central_slices], start=start)
                    if self.save_raw_predictions:
                        self.infer_raw_writer.write_block(raw_result[central_slices], start=start)
                    print(f"{time_str()} [BLOCK{block_num:07d}/{self.total_block_num:07d}] "
                          "[INFER] requested results saved", flush=True)
                # A block is committed only after all its selected output writes.
                self.inferred_block_num = block_num + 1
                self.save_infer_log()
            infer_completed = True
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            traceback.print_exc()
            raise
        finally:
            try:
                if infer_completed:
                    factors = [(1, 2, 2), (2, 4, 4), (4, 8, 8), (8, 16, 16), (16, 32, 32)]
                    if self.infer_writer is not None:
                        print(f"{time_str()} building segmentation label pyramid", flush=True)
                        self.infer_writer.build_pyramid(factors=factors, mode="nearest")
                    if self.infer_raw_writer is not None:
                        print(f"{time_str()} building float32 probability pyramid", flush=True)
                        self.infer_raw_writer.build_pyramid(factors=factors, mode="mean")
            finally:
                self.close_reader_writer()

    def _infer_one_block(self, coord, skeleton):
        """Predict and post-process one spatial block using overlapping patches.

        Args:
            coord: Global (3, 2) half-open ZYX bounds of the block to read.
            skeleton: Skeletons cropped to this block for watershed seeding.

        Returns:
            (CZYX label segmentation or None, float32 CZYX probabilities or None).
            Overlapping probabilities are blended before either output is made.
            In float32-only mode, skeletonization and watershed are not called.
        """
        block_size = np.array((self.patch_size[3], *(coord[:, 1] - coord[:, 0])), dtype=np.int64)
        block_raw = np.zeros(tuple(block_size), dtype=self.infer_reader.dtype)  # CZYX
        for i in range(self.patch_size[3]):
            block_raw[i, ...] = self.infer_reader[self.infer_channel[i], coord[0, 0]:coord[0, 1], coord[1, 0]:coord[1, 1], coord[2, 0]:coord[2, 1],]
        # Keep original intensity until each patch is padded and normalized.
        # Accumulate weighted probabilities instead of allowing a later patch to overwrite an earlier prediction in overlapping regions.
        block_pred_sum = np.zeros((self.out_channel_num, *block_size[1:4]), dtype=np.float32,)
        # The same spatial blending weight is used for every output channel.
        block_weight_sum = np.zeros(tuple(block_size[1:4]), dtype=np.float32)
        all_patch_coords = list(
            patch_coordinates_iter(block_size[1:4], self.patch_size[0:3], self.infer_overlap, self.infer_padding,))
        if not all_patch_coords:
            raise RuntimeError(f"No inference patches were generated for block shape {tuple(block_size[1:4])}.")
        # Determine the real overlap on both sides of every patch. This is
        # derived from the generated coordinates rather than assuming that the
        # requested overlap is always exact; the final end-aligned patch can
        # have a larger overlap than intermediate patches.
        axis_intervals = []
        axis_interval_indices = []
        for axis in range(3):
            intervals = sorted(
                {(int(p_coord[axis, 0]), int(p_coord[axis, 1])) for p_coord in all_patch_coords},
                key=lambda interval: (interval[0], interval[1]),)
            axis_intervals.append(intervals)
            axis_interval_indices.append({interval: index for index, interval in enumerate(intervals)})
        patch_overlap_widths = []
        for p_coord in all_patch_coords:
            overlap_widths = np.zeros((3, 2), dtype=np.int64)
            for axis in range(3):
                current_interval = (int(p_coord[axis, 0]), int(p_coord[axis, 1]),)
                interval_index = axis_interval_indices[axis][current_interval]
                intervals = axis_intervals[axis]
                if interval_index > 0:
                    previous_interval = intervals[interval_index - 1]
                    overlap_widths[axis, 0] = max(0, previous_interval[1] - current_interval[0],)
                if interval_index + 1 < len(intervals):
                    next_interval = intervals[interval_index + 1]
                    overlap_widths[axis, 1] = max(0, current_interval[1] - next_interval[0],)
            patch_overlap_widths.append(overlap_widths)

        for batch_start in range(0, len(all_patch_coords), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(all_patch_coords),)
            block_pred_sum, block_weight_sum = self._infer_one_batch(
                block_pred_sum,
                block_weight_sum,
                block_raw,
                all_patch_coords[batch_start:batch_end],
                patch_overlap_widths[batch_start:batch_end],)
        uncovered_mask = block_weight_sum <= 0
        if np.any(uncovered_mask):
            uncovered_voxels = int(np.count_nonzero(uncovered_mask))
            raise RuntimeError(f"Patch blending left {uncovered_voxels} block voxels without any prediction weight. Check PATCH_SIZE, OVERLAP and PADDING.")
        block_pred = block_pred_sum / block_weight_sum[None, ...]
        # Numerical roundoff during weighted accumulation can move probabilities
        # a few ULPs outside the sigmoid range.
        np.clip(block_pred, 0.0, 1.0, out=block_pred)
        # Float32 is the final overlap-blended probability output, not logits.
        raw_result = block_pred if self.save_raw_predictions else None
        seg_result = None
        if self.save_segmentation:
            seg_result = self._postprocess_block_predictions(block_pred, coord, skeleton)
        return seg_result, raw_result

    def _postprocess_block_predictions(self, block_pred, coord, skeleton):
        """Convert float32 probabilities into an instance + semantic label block."""
        result_dtype = np.dtype(self.segmentation_dtype)
        block_result = np.zeros((self.postpro_channel_num, *block_pred.shape[1:4]), dtype=result_dtype)
        print(f"{time_str()} [BLOCK{self.inferred_block_num:07d}/{self.total_block_num:07d}] "
              "[INFER] start post-processing", flush=True)
        start_id = 0
        if self.instance_num:
            # SWC-guided mode retains the original selection/ID semantics.
            # SWC-free mode always watersheds from learned F/P/C seed maps.
            if not self.skeleton_flag or len(skeleton) > 0:
                skeleton_label = (
                    self.skeleton_manager.create_cropped_skeleton_mask(skeleton, coord)
                    if self.skeleton_flag else None
                )
                block_result[0, ...] = watershed_with_sk(
                    block_pred[self.instance_indices, ...],
                    [self.output_channel_names[i] for i in self.instance_indices],
                    self.watershed_seed_channels,
                    self.watershed_seed_channels_thresh,
                    self.watershed_topographic_channel,
                    self.watershed_growth_mask_channels,
                    self.watershed_growth_mask_channels_thresh,
                    skeleton_label=skeleton_label,
                    output_dtype=result_dtype,
                )
                block_result[0, ...] = post_processing_instance(
                    block_result[0, ...], self.postprocess_dict)
            start_id = 1
        if self.semantic_num:
            block_result[start_id:, ...] = (block_pred[self.semantic_indices, ...] > 0.5).astype(result_dtype)
            block_result[start_id:, ...] = post_processing_semantic(
                block_result[start_id:, ...], self.postprocess_dict)
        return block_result

    def _stitch_instance_block(self, result, coord, central_coord, block_num):
        """Give the central region globally unique IDs and stitch across halos.

        No SWC IDs are used. The current block's halo comes directly from the
        inference block bounds computed with ``self.block_padding``; it is NOT
        a second, independently configured instance-segmentation halo.
        Only overlaps between this halo and **already committed** neighboring
        central blocks can reuse an existing ID; other local watershed instances
        receive fresh monotonic uint32 IDs. The matching decision is one-to-one
        per block to reduce false merges. Since it is local and greedy, this is
        not a guarantee of correct long-range neuron identities.
        """
        result = np.asarray(result)
        local_labels = result[0]
        center_slices = tuple(slice(int(a), int(b)) for a, b in central_coord)
        central_ids = np.unique(local_labels[center_slices])
        central_ids = central_ids[central_ids > 0]
        if central_ids.size == 0:
            return result

        max_uint32 = np.iinfo(np.uint32).max
        if self.next_instance_id > max_uint32:
            raise OverflowError("Global instance counter exceeded uint32 capacity")
        central_ids_set = set(map(int, central_ids))
        old_to_new: dict[int, int] = {}

        # Full-volume blocks are visited in Z, then Y, then X order. A voxel's
        # central-block index therefore determines whether it has been committed.
        # This includes zero-valued background, which must count against the
        # matching fraction. Crucially it excludes any stale data from an
        # interrupted write of the current central block.
        central_size = np.asarray(self.c_block_size, dtype=np.int64)
        block_grid = (
            np.asarray(self.infer_reader.shape[1:4], dtype=np.int64)
            + central_size - 1
        ) // central_size
        block_zyx = np.unravel_index(int(block_num), tuple(block_grid))
        axis_index = [
            np.arange(int(coord[axis, 0]), int(coord[axis, 1]), dtype=np.int64)
            // int(self.c_block_size[axis]) for axis in range(3)
        ]
        already_processed = (
            (axis_index[0][:, None, None] < block_zyx[0])
            | ((axis_index[0][:, None, None] == block_zyx[0])
               & (axis_index[1][None, :, None] < block_zyx[1]))
            | ((axis_index[0][:, None, None] == block_zyx[0])
               & (axis_index[1][None, :, None] == block_zyx[1])
               & (axis_index[2][None, None, :] < block_zyx[2]))
        )

        # `already_processed` is the part of the inference halo that lies in
        # previously committed central blocks. At the first block it is empty.
        if np.any(already_processed):
            reference = np.asarray(self.infer_writer[
                0,
                int(coord[0, 0]):int(coord[0, 1]),
                int(coord[1, 0]):int(coord[1, 1]),
                int(coord[2, 0]):int(coord[2, 1]),
            ])
            # The denominator covers both previously labeled and previously
            # background voxels, minimizing spurious matches to tiny overlaps.
            old_area_local = local_labels[already_processed]
            old_area_local = old_area_local[old_area_local > 0]
            eligible_ids, eligible_counts = np.unique(old_area_local, return_counts=True)
            local_sizes = dict(zip(map(int, eligible_ids), map(int, eligible_counts)))
            valid = already_processed & (local_labels > 0) & (reference > 0)
            if np.any(valid):
                codes = (local_labels[valid].astype(np.uint64) << np.uint64(32)) | reference[valid].astype(np.uint64)
                pair_codes, pair_counts = np.unique(codes, return_counts=True)
                candidates = []
                min_voxels = int(self.cfg.INFER.STITCH_MIN_OVERLAP_VOXELS)
                min_fraction = float(self.cfg.INFER.STITCH_MIN_FRACTION)
                if min_voxels < 1 or not 0.0 < min_fraction <= 1.0:
                    raise ValueError("Invalid stitching thresholds: MIN_OVERLAP_VOXELS >= 1 and 0 < MIN_FRACTION <= 1 required")
                for code, count in zip(pair_codes, pair_counts):
                    local_id = int(code >> np.uint64(32))
                    global_id = int(code & np.uint64(max_uint32))
                    if local_id not in central_ids_set:
                        continue
                    fraction = int(count) / local_sizes[local_id]
                    if int(count) >= min_voxels and fraction >= min_fraction:
                        candidates.append((int(count), fraction, local_id, global_id))
                # Each local region can match at most one existing global ID,
                # and each previously assigned global ID at most one region.
                used_global = set()
                for count, fraction, local_id, global_id in sorted(candidates, reverse=True):
                    if local_id not in old_to_new and global_id not in used_global:
                        old_to_new[local_id] = global_id
                        used_global.add(global_id)

        for local_id in map(int, central_ids):
            if local_id not in old_to_new:
                if self.next_instance_id > max_uint32:
                    raise OverflowError("Global instance counter exceeded uint32 capacity")
                old_to_new[local_id] = int(self.next_instance_id)
                self.next_instance_id += 1

        # Only the central region is committed. Avoid a full-volume LUT sized by
        # the maximum marker ID by applying a sorted search over present labels.
        centers = local_labels[center_slices]
        sorted_ids = np.array(sorted(old_to_new), dtype=np.uint32)
        new_ids = np.array([old_to_new[int(i)] for i in sorted_ids], dtype=np.uint32)
        remapped = np.zeros(centers.shape, dtype=np.uint32)
        foreground = centers > 0
        remapped[foreground] = new_ids[np.searchsorted(sorted_ids, centers[foreground])]
        result[0][center_slices] = remapped
        return result

    def _infer_one_batch(
        self,
        block_pred_sum,
        block_weight_sum,
        block_raw,
        p_coord_list,
        p_overlap_widths,
    ):
        """Accumulate weighted patch probabilities into a block's blending buffers.

        Args:
            block_pred_sum: Mutable float32 CZYX weighted probability accumulator.
            block_weight_sum: Mutable ZYX spatial weight accumulator.
            block_raw: Preprocessed CZYX source block.
            p_coord_list: Block-local (3, 2) ZYX bounds for each patch in the batch.
            p_overlap_widths: Per-patch (3, 2) low/high overlap widths in voxels.

        Returns:
            The two updated accumulation buffers. Short patches are padded to
            model input size and cropped back before accumulation.
        """
        current_batch_size = len(p_coord_list)
        if current_batch_size == 0:
            return block_pred_sum, block_weight_sum
        if current_batch_size != len(p_overlap_widths):
            raise ValueError("p_coord_list and p_overlap_widths must have the same length.")
        patch_raw = np.zeros((current_batch_size, self.patch_size[3], *self.patch_size[0:3],), dtype=np.float32,)  # BCZYX: retain normalized fractional values
        for i, p_coord in enumerate(p_coord_list):
            patch = block_raw[
                :, p_coord[0, 0]:p_coord[0, 1], p_coord[1, 0]:p_coord[1, 1],p_coord[2, 0]:p_coord[2, 1],]  # CZYX
            if patch.shape[1:4] != patch_raw.shape[2:5]:
                patch = reflect_padding_img(patch, patch_raw.shape[2:5])
            patch_raw[i, ...] = preprocess_img(patch, self.preprocess_dict)

        patch_tensor = torch.from_numpy(patch_raw.astype(np.float32, copy=False)).to(self.device, non_blocking=True)
        patch_pred = self.model(patch_tensor)  # B(out_channel_num)ZYX
        patch_pred = torch.sigmoid(patch_pred).cpu().numpy().astype(np.float32, copy=False,)

        def make_axis_weight(
            axis_length: int,
            left_overlap: int,
            right_overlap: int,
        ) -> np.ndarray:
            """Create complementary linear feathering weights for one axis."""
            axis_weight = np.ones(axis_length, dtype=np.float32)
            left_overlap = min(max(int(left_overlap), 0), axis_length)
            right_overlap = min(max(int(right_overlap), 0), axis_length)
            if left_overlap > 0:
                # Excluding exact 0/1 endpoints prevents zero-weight voxels and
                # still produces complementary ramps for adjacent patches.
                left_ramp = np.linspace(
                    0.0,
                    1.0,
                    left_overlap + 2,
                    dtype=np.float32,
                )[1:-1]
                axis_weight[:left_overlap] *= left_ramp
            if right_overlap > 0:
                right_ramp = np.linspace(
                    1.0,
                    0.0,
                    right_overlap + 2,
                    dtype=np.float32,
                )[1:-1]
                axis_weight[-right_overlap:] *= right_ramp
            return axis_weight

        for i, (p_coord, overlap_widths) in enumerate(zip(p_coord_list, p_overlap_widths)):
            patch_shape = (int(p_coord[0, 1] - p_coord[0, 0]), int(p_coord[1, 1] - p_coord[1, 0]), int(p_coord[2, 1] - p_coord[2, 0]),)
            z_weight = make_axis_weight(patch_shape[0], overlap_widths[0, 0], overlap_widths[0, 1],)
            y_weight = make_axis_weight(patch_shape[1], overlap_widths[1, 0], overlap_widths[1, 1],)
            x_weight = make_axis_weight(patch_shape[2],overlap_widths[2, 0], overlap_widths[2, 1],)
            patch_weight = (z_weight[:, None, None] * y_weight[None, :, None] * x_weight[None, None, :])
            block_slices = (
                slice(int(p_coord[0, 0]), int(p_coord[0, 1])),
                slice(int(p_coord[1, 0]), int(p_coord[1, 1])),
                slice(int(p_coord[2, 0]), int(p_coord[2, 1])),
            )
            valid_patch_pred = patch_pred[i, :, :patch_shape[0], :patch_shape[1], :patch_shape[2],]
            block_pred_sum[(slice(None), *block_slices)] += valid_patch_pred * patch_weight[None, ...]
            block_weight_sum[block_slices] += patch_weight
        return block_pred_sum, block_weight_sum

    @staticmethod
    def _file_identity(path):
        path = Path(path).resolve()
        stats = path.stat()
        return {"path": str(path), "size": stats.st_size,
                "mtime_ns": stats.st_mtime_ns}

    def _build_infer_manifest(self):
        """Fingerprint parameters and input identities that determine inference output."""
        source = Path(self.infer_path).resolve()
        if source.is_file():
            input_metadata = self._file_identity(source)
        else:
            # Avoid walking a PB volume: identify the root metadata only.
            input_metadata = {"path": str(source),
                              "metadata_files": [self._file_identity(p) for p in
                              (source / ".zattrs", source / ".zgroup", source / "zarr.json",
                               source / ".zmetadata") if p.is_file()]}
        checkpoint = self.checkpoint_input_file
        checksum = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                checksum.update(chunk)
        skeletons = []
        if self.skeleton_flag and (self.save_segmentation or not self.raw_all_blocks):
            directory = Path(self.skeleton_path)
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing SWC directory: {directory}")
            skeletons = [self._file_identity(path) for path in sorted(directory.glob("*.swc"))]
            if not skeletons:
                raise ValueError("Skeleton-guided inference requires SWC files")
        # Loading-only options must not invalidate already running inference jobs.
        architecture_cfg = self.cfg.MODEL.clone()
        for key in ("CHECKPOINT_ALLOW_PARTIAL", "CHECKPOINT_INPUT_PATH"):
            if key in architecture_cfg:
                del architecture_cfg[key]
        fields = {"format_version": 2, "job_id": int(self.job_id),
                  "input": input_metadata, "input_shape": list(map(int, self.infer_reader.shape)),
                  "checkpoint_sha256": checksum.hexdigest(),
                  "skeletons": skeletons,
                  "output_mode": self.output_mode,
                  "segmentation_path": str(self.infer_gt_path.resolve()) if self.save_segmentation else None,
                  "probability_path": str(self.infer_raw_path.resolve()) if self.save_raw_predictions else None,
                  "selected_raw_channels": list(self.infer_channel),
                  "model": str(architecture_cfg), "task": str(self.cfg.TASK),
                  "normalization": str(self.cfg.DATA.NORMALIZATION),
                  "patch_size": list(self.patch_size),
                  "infer_overlap": list(map(int, self.infer_overlap)),
                  "infer_padding": list(self.infer_padding),
                  "block_size": list(map(int, self.block_size)),
                  "central_block_size": list(map(int, self.c_block_size)),
                  "watershed": {"seed": list(self.watershed_seed_channels),
                                 "seed_thresholds": list(self.watershed_seed_channels_thresh),
                                 "surface": self.watershed_topographic_channel,
                                 "growth_mask": list(self.watershed_growth_mask_channels),
                                 "growth_thresholds": list(self.watershed_growth_mask_channels_thresh)},
                  "postprocessing": self.postprocess_dict,
                  "skeleton_enabled": bool(self.skeleton_flag),
                  "raw_all_blocks": bool(self.raw_all_blocks),
                  "stitch_min_overlap_voxels": int(self.cfg.INFER.STITCH_MIN_OVERLAP_VOXELS),
                  "stitch_min_fraction": float(self.cfg.INFER.STITCH_MIN_FRACTION),
                  "output_channel_names": list(self.output_channel_names)}
        return fields

    def save_infer_log(self) -> Path:
        """Atomically write the current block progress and output-store path to JSON.

        Returns:
            Path of the per-job progress file under infer_log_path. A temporary
            file is flushed and synced before replacement, then cleaned up.
        """
        self.infer_log_path = Path(self.cfg.DATA.INFER.INFER_LOG)
        self.infer_log_path.mkdir(parents=True, exist_ok=True)
        log_path = self.infer_log_path / f"infer_log_{int(self.job_id):02d}.json"
        payload = {
            "job_id": int(self.job_id),
            "manifest": self.infer_manifest,
            "inferred_block_num": self.inferred_block_num,
            "output_mode": self.output_mode,
            "raw_all_blocks": bool(self.cfg.INFER.RAW_ALL_BLOCKS) and self.save_raw_predictions,
            "skeleton_enabled": bool(self.skeleton_flag),
            "next_instance_id": int(self.next_instance_id),
            "segmentation_dtype": self.segmentation_dtype,
            "ome_zarr_path": str(self.infer_gt_path) if self.save_segmentation else None,
            "raw_ome_zarr_path": str(self.infer_raw_path) if self.save_raw_predictions else None,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),}
        temporary_path = log_path.with_name(f".{log_path.name}.{os.getpid()}.tmp")
        try:
            with temporary_path.open(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
            ) as file:
                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, log_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return log_path

    def load_infer_log(self):
        """Load saved inference progress.

        Returns ``(0, None)`` when no inference log exists. A malformed or
        incompatible log raises an exception instead of silently restarting
        from zero, because doing so could overwrite an existing partial
        result.

        Returns
        -------
        tuple[int, pathlib.Path or None]
            ``(inferred_block_num, ome_zarr_path)``.
        """
        self.inferred_block_num = 0
        log_path = Path(self.infer_log_path) / f"infer_log_{int(self.job_id):02d}.json"
        if not log_path.exists():
            # Never append into unknown output stores without a valid manifest.
            for required, store in ((self.save_segmentation, self.infer_gt_path),
                                    (self.save_raw_predictions, self.infer_raw_path)):
                if required and Path(store).exists():
                    raise FileExistsError(f"Output store exists without a valid inference progress log: {store}")
            return
        if not log_path.is_file():
            raise ValueError(f"Inference log path exists but is not a file: {log_path}")
        try:
            with log_path.open(mode="r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Inference log is not valid JSON: {log_path}") from error
        except OSError as error:
            raise RuntimeError(f"Unable to read inference log: {log_path}") from error
        if payload.get("manifest") != self.infer_manifest:
            raise ValueError("Inference resume manifest differs from current source, "
                             "checkpoint, normalization, channels or inference parameters. "
                             "Use a new job_id/output path; legacy logs without a manifest are not safe to resume.")
        saved_job_id = payload.get("job_id")
        if isinstance(saved_job_id, bool) or not isinstance(saved_job_id, int) or saved_job_id != int(self.job_id):
            raise ValueError(f"Inference log belongs to a different job: saved job_id={saved_job_id!r}, current job_id={self.job_id!r}.")
        inferred_block_num = payload.get("inferred_block_num")
        if isinstance(inferred_block_num, bool) or not isinstance(inferred_block_num, int) or inferred_block_num < 0:
            raise ValueError( f"Inference log contains an invalid inferred_block_num: {inferred_block_num!r}.")
        saved_mode = payload.get("output_mode", "uint16")  # legacy segmentation-only logs
        if saved_mode != self.output_mode:
            raise ValueError(
                f"Inference log output_mode={saved_mode!r} does not match "
                f"configured OUTPUT_MODE={self.output_mode!r}. Use a new job_id/log."
            )
        saved_all_blocks = payload.get("raw_all_blocks", False)
        configured_all_blocks = bool(self.cfg.INFER.RAW_ALL_BLOCKS) and self.save_raw_predictions
        if not isinstance(saved_all_blocks, bool) or saved_all_blocks != configured_all_blocks:
            raise ValueError(
                "Inference log raw_all_blocks does not match the current configuration. "
                "Use a new job_id/log when changing INFER.RAW_ALL_BLOCKS."
            )
        if inferred_block_num > self.total_block_num:
            raise ValueError("Inference log progress exceeds the current image block count.")
        # Legacy progress files predate these keys and can only be resumed in
        # their original SWC-enabled uint16 mode.
        old_skeleton_flag = payload.get("skeleton_enabled", True)
        if not isinstance(old_skeleton_flag, bool) or old_skeleton_flag != bool(self.skeleton_flag):
            raise ValueError("Inference log skeleton mode differs from configuration. Use a new job ID/log.")
        old_dtype = payload.get("segmentation_dtype", "uint16")
        if old_dtype != self.segmentation_dtype:
            raise ValueError("Inference log segmentation dtype differs from configuration. Use a new job ID/log.")
        if self.save_segmentation and self.instance_num and not self.skeleton_flag:
            next_id = payload.get("next_instance_id")
            if isinstance(next_id, bool) or not isinstance(next_id, int) or not 1 <= next_id <= np.iinfo(np.uint32).max + 1:
                raise ValueError("SWC-free inference log needs a valid next_instance_id for resuming.")
            self.next_instance_id = next_id
        requested_stores = (
            ("ome_zarr_path", self.infer_gt_path, self.save_segmentation),
            ("raw_ome_zarr_path", self.infer_raw_path, self.save_raw_predictions),
        )
        for key, expected_path, required in requested_stores:
            if not required:
                continue
            saved_path = payload.get(key)
            if not isinstance(saved_path, str) or not saved_path.strip():
                raise ValueError(f"Inference log missing required {key}.")
            if not Path(saved_path).is_dir() or not Path(saved_path).samefile(expected_path):
                raise ValueError(f"Inference log {key} differs from the configured output store.")
        self.inferred_block_num = inferred_block_num


    def run(self):
        """Run training when enabled, otherwise inference when enabled, then close logs."""
        if self.cfg.TRAIN.ENABLE:
            print(f"{time_str()} start instance segmentation training", flush=True)
            self.train()
        elif self.cfg.INFER.ENABLE:
            print(f"{time_str()} start instance segmentation inference", flush=True)
            self.infer()
        self.log_writer.close()
