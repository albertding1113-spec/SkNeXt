from pathlib import Path
import torch
import os
import re
import math
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sknext.run.base_workflow import Base_Workflow
from sknext.data.imageIO import get_tif_path_in_folder, get_tif_path_dict_in_folder
from sknext.data.datasetIO import tif_list_to_zarr
from sknext.data.data_loader import ZarrPatchLoader
from sknext.model.unext_v2 import U_NeXt_V2
from sknext.model.scheduler import build_onecycle_scheduler, build_warmup_cosine_scheduler
from sknext.utils.utils import get_cfg_value, time_str
from sknext.model.loss import instance_segmentation_loss, instance_segmentation_metrics
from sknext.utils.model_info import print_model_parameters, profile_forward_memory
from tifffile import tifffile


class Segmentation_Workflow(Base_Workflow):
    def __init__(self, cfg, device:torch.device, job_id:int):
        super(Segmentation_Workflow, self).__init__(cfg, device, job_id)
        self.channels = self.cfg.TASK.CHANNELS
        self.channels_extra_opts = cfg.TASK.CHANNELS_EXTRA_OPTS[0] if len(cfg.TASK.CHANNELS_EXTRA_OPTS[0]) else {}
        self.channel_weights = cfg.TASK.CHANNEL_WEIGHTS
        self.watershed_seed_channels = cfg.TASK.WATERSHED.SEED_CHANNELS
        self.watershed_seed_channels_thresh = cfg.TASK.WATERSHED.SEED_CHANNELS_THRESH
        self.watershed_topographic_channel = cfg.TASK.WATERSHED.TOPOGRAPHIC_SURFACE_CHANNEL
        self.watershed_growth_mask_channels = cfg.TASK.WATERSHED.GROWTH_MASK_CHANNELS
        self.watershed_growth_mask_channels_thresh = cfg.TASK.WATERSHED.GROWTH_MASK_CHANNELS_THRESH
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
        self.train_tif_list = get_tif_path_in_folder(self.train_path)
        self.train_gt_tif_dict = get_tif_path_dict_in_folder(self.train_gt_path, self.channels)
        self.val_tif_list = get_tif_path_in_folder(self.val_path)
        self.val_gt_tif_dict = get_tif_path_dict_in_folder(self.val_gt_path, self.channels)
        self.train_zarr_path = Path(self.train_path).parent / "train.ome.zarr"
        self.val_zarr_path = Path(self.val_path).parent / "val.ome.zarr"
        tif_list_to_zarr(self.train_tif_list, self.train_gt_tif_dict, self.train_zarr_path, self.patch_size,
                         self.train_overlap, self.train_padding, self.preprocess_dict, self.channels,
                         self.channels_extra_opts)
        tif_list_to_zarr(self.val_tif_list, self.val_gt_tif_dict, self.val_zarr_path, self.patch_size,
                         self.val_overlap, self.val_padding, self.preprocess_dict, self.channels,
                         self.channels_extra_opts)

    def define_activations_and_channels(self):
        self.out_channel_num = 0
        self.output_channel_info = []
        for channel in self.channels:
            if channel == "A":
                a_opts = self.channels_extra_opts.get("A", {"z_affinities": [1], "y_affinities": [1], "x_affinities": [1]})
                z_aff = a_opts.get("z_affinities", [1])
                y_aff = a_opts.get("y_affinities", [1])
                x_aff = a_opts.get("x_affinities", [1])
                assert len(z_aff) == len(y_aff) == len(x_aff), "z/y/x affinities should have the same length."
                self.out_channel_num += 3 * len(z_aff)
                self.output_channel_info += [f"A_z_{d}" for d in z_aff] + [f"A_y_{d}" for d in y_aff] + [f"A_x_{d}" for d in x_aff]
            else:
                self.out_channel_num += 1
                self.output_channel_info += [channel]
        self.head_activations = list(["linear"] * self.out_channel_num)

    def set_loss_metrics(self):
        self.loss_func = instance_segmentation_loss(channel_weights=self.channel_weights,
                                                    channel_names=self.output_channel_info,
                                                    bce_weight=1.0,
                                                    dice_weight=1.0,
                                                    auto_balance=True)
        self.loss_func.to(self.device)
        self.metric = instance_segmentation_metrics(self.output_channel_info)

    def set_optimizer_scheduler(self):
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
        self.define_activations_and_channels()
        assert self.model_name in ["unext_v2", "u_next_v2", "unext-v2"], f"Unsupported model: {self.model_name}"
        self.model = U_NeXt_V2(image_shape=self.patch_size,
                               feature_maps=self.feature_maps,
                               upsample_layer="convtranspose",
                               z_down=self.z_down,
                               yx_down=self.yx_down,
                               output_channels=[self.out_channel_num],
                               separated_decoders=False,
                               output_channel_info=["+".join(self.output_channel_info)],
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

    def load_checkpoint(self):
        assert hasattr(self, "model") and self.model is not None, (
            "self.model is None. Please call set_model() before loading checkpoint.")
        checkpoint_file = Path(self.checkpoint_file)
        assert checkpoint_file.exists(), "Checkpoint file does not exist."
        assert checkpoint_file.is_file(), "Checkpoint path is not a file."
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        model_name = checkpoint["model_name"]
        assert self.model_name == model_name, "Checkpoint file does not match."
        state_dict = checkpoint["model_state_dict"]
        # compatible with multi-GPU
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        if len(state_dict) > 0 and all(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        incompatible_keys = model_to_load.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        print(f"{time_str()} Model checkpoint loaded from: {self.checkpoint_file}", flush=True)

    def save_checkpoint(self):
        assert hasattr(self, "model") and self.model is not None, (
            "self.model is None. Please call set_model() before saving checkpoint.")
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {"model_state_dict": {key: value.detach().cpu()
                                           for key, value in model_to_save.state_dict().items()},
                      "model_name": self.model_name,}
        torch.save(checkpoint, self.checkpoint_file)
        print(f"{time_str()} Model checkpoint saved to: {self.checkpoint_file}", flush=True)

    def train(self):
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
        print(f"{time_str()} preparing optimizer and scheduler", flush=True)
        self.set_optimizer_scheduler()
        print(f"{time_str()} preparing loss function and metrics", flush=True)
        self.set_loss_metrics()
        if self.load_checkpoint_flag:
            self.load_checkpoint()

        best_val_loss = self.best_val_loss  # float('inf')
        best_val_metrics = {}
        epochs_without_improvement = 0
        for epoch in range(self.epochs):
            epoch_number = epoch + 1
            print(f"{time_str()} [EPOCH{epoch_number:04d}]", flush=True)
            self.model.train()
            epoch_train_loss = 0
            epoch_train_metrics = {}  # {"F" 0.0:, "P": 0.0}
            epoch_train_patches = 0
            for step in range(self.train_steps_per_epoch):
                # clear optimizer grad before starting everything
                self.optimizer.zero_grad()
                start_ = step * self.batch_size
                end_ = min((step + 1) * self.batch_size, self.train_patches)
                img, mask = self.train_patch_loader[start_:end_]
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
            self.model.eval()
            epoch_val_loss = 0
            epoch_val_metrics = {}
            epoch_val_patches = 0
            with torch.no_grad():
                for step in range(self.val_steps_per_epoch):
                    start_ = step * self.batch_size
                    end_ = min((step + 1) * self.batch_size, self.val_patches)
                    img, mask = self.val_patch_loader[start_:end_]
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
                if epochs_without_improvement >= self.patience:
                    metric_str = " ".join(f"{key} channel: {value:.6f}," for key, value in best_val_metrics.items())
                    print(f"{time_str()} Out of patience, stopping training. Best validation loss: {best_val_loss:.6f}", flush=True)
                    # Preserve the latest partial interval before stopping.
                    return

    def infer(self):
        pass

    def run(self):
        if self.cfg.TRAIN.ENABLE:
            print(f"{time_str()} start instance segmentation training", flush=True)
            self.train()
        elif self.cfg.INFER.ENABLE:
            print(f"{time_str()} start instance segmentation inference", flush=True)
            self.infer()
        self.log_writer.close()
