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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Iterable, Sequence
from sknext.run.base_workflow import Base_Workflow
from sknext.data.imageIO import get_tif_path_in_folder, get_tif_path_dict_in_folder, patch_coordinates_iter, get_coord_after_padding, central_block_coordinates_iter
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
from sknext.skeleton.skeleton import SkeletonManager


class Segmentation_Workflow(Base_Workflow):
    def __init__(self, cfg, device:torch.device, job_id:int):
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

    def set_loss_metrics(self):
        self.loss_func = instance_segmentation_loss(channel_weights=self.channel_weights,
                                                    channel_names=self.output_channel_names,
                                                    bce_weight=1.0,
                                                    dice_weight=1.0,
                                                    auto_balance=True)
        self.loss_func.to(self.device)
        self.metric = instance_segmentation_metrics(self.output_channel_names)

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
        self.skeleton_manager = SkeletonManager(self.skeleton_path)

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

    def create_reader(self):
        self.chunk = (1, self.patch_size[0]*2, self.patch_size[1]*4, self.patch_size[2]*4)
        self.infer_path = Path(self.infer_path)
        assert self.infer_path.exists(), "Infer path does not exist."
        file_type = detect_path_type(self.infer_path)
        if file_type == "zarr":
            self.infer_reader = ZarrIOManager(self.infer_path, mode="r")
        elif file_type == "ims":
            self.infer_reader = IMSReader(self.infer_path)
        else: raise FileNotFoundError(f"file type {file_type} is not supported.")

    def create_writer(self):
        self.infer_gt_path = Path(self.infer_gt_path)
        if self.infer_gt_path.exists():
            file_type = detect_path_type(self.infer_gt_path)
            if file_type == "zarr": pass
            elif file_type == "dir":
                self.infer_gt_path = self.infer_gt_path / f"result{self.job_id}.ome.zarr"
        else:
            self.infer_gt_path.mkdir(parents=True, exist_ok=True)
            if self.infer_gt_path.suffix == ".zarr": pass
            elif self.infer_gt_path.is_dir():
                self.infer_gt_path = self.infer_gt_path / f"result{self.job_id}.ome.zarr"
        self.infer_writer = ZarrIOManager(self.infer_gt_path,
                                          shape=(self.postpro_channel_num, *self.infer_reader.shape[1:4]),
                                          dtype="uint16",
                                          chunks=self.chunk,
                                          mode="a",
                                          channel_names=self.postpro_channel_info)


    def define_postprocess_channels(self):
        self.postpro_channel_num = 0
        self.postpro_channel_info = []
        self.instance_num = 0
        self.semantic_num = 0
        for channel in self.channels:
            if channel in ["F", "C", "P"]:
                self.instance_num += 1
            if channel in ["A"]:
                a_opts = self.channels_extra_opts.get("A",{"z_affinities": [1], "y_affinities": [1], "x_affinities": [1]})
                z_aff = a_opts.get("z_affinities", [1])
                y_aff = a_opts.get("y_affinities", [1])
                x_aff = a_opts.get("x_affinities", [1])
                assert len(z_aff) == len(y_aff) == len(x_aff), "z/y/x affinities should have the same length."
                self.instance_num += 3 * len(z_aff)
            if "S." in channel:
                self.semantic_num += 1
                self.postpro_channel_num += 1
                self.postpro_channel_info.append(channel.replace("S.", ""))
        if self.instance_num:
            self.postpro_channel_num += 1
            self.postpro_channel_info = tuple(["Infer_instance",] + self.postpro_channel_info)

    def close_reader_writer(self):
        self.infer_reader.close()
        self.infer_writer.close()

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
        torch.cuda.empty_cache()
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

    @torch.no_grad()
    def infer(self):
        # create and load model
        print(f"{time_str()} preparing model", flush=True)
        self.set_model()
        print_model_parameters(self.model)
        profile_forward_memory(self.model, input_shape=(self.batch_size, self.patch_size[-1], *self.patch_size[:-1],),
                               device=self.device)
        torch.cuda.empty_cache()
        self.load_checkpoint()
        self.model.eval()
        self.define_postprocess_channels()
        print(f"{time_str()} creating reader, writer and skeleton_manager", flush=True)
        # create reader
        self.create_reader()
        # create writer
        self.create_writer()
        # create coordinate_iter
        coord_iter = central_block_coordinates_iter(self.infer_reader.shape[1:4], self.c_block_size)
        self.total_block_num = int(np.prod(np.ceil(
            np.array(self.infer_reader.shape[1:4], dtype="float32") / np.array(self.c_block_size, dtype="float32"))))
        # create skeleton-manager
        self.create_skeleton()
        # load saved infer log
        self.load_infer_log()

        try:
            block_num = -1
            infer_completed = False
            # start predict
            for c_coord in coord_iter:
                block_num += 1
                if block_num < self.inferred_block_num: continue
                coord, coord_in_block = get_coord_after_padding(self.infer_reader.shape[1:4], c_coord, self.block_padding)
                cropped_skeleton = self.skeleton_manager.crop_skeletons(coord) # navis.NeuronList
                if len(cropped_skeleton) > 0:
                    print(f"{time_str()} [BLOCK{self.inferred_block_num:07d}/{self.total_block_num:07d}] [INFER] {len(cropped_skeleton):04d} skeleton(s) in this block, start inferring", flush=True)
                    block_result = self._infer_one_block(coord, cropped_skeleton)
                    self.infer_writer.write_block(
                        block_result[:, coord_in_block[0,0]:coord_in_block[0,1], coord_in_block[1,0]:coord_in_block[1,1], coord_in_block[2,0]:coord_in_block[2,1]],
                        start=(0, c_coord[0,0], c_coord[1,0], c_coord[2, 0]))
                    print(f"{time_str()} [BLOCK{self.inferred_block_num:07d}/{self.total_block_num:07d}] [INFER] results saved", flush=True)
                self.inferred_block_num = block_num + 1
                self.save_infer_log()
            infer_completed = True
        except Exception as e:
            print(f"Error type：{type(e).__name__}")
            print(f"Error message：{e}")
            traceback.print_exc()
            raise
        finally:
            try:
                if infer_completed:
                    # build pyramid
                    print(f"{time_str()}, All inference completed, start building pyramid", flush=True)
                    self.infer_writer.build_pyramid(factors=[(1, 2, 2), (2, 4, 4), (4, 8, 8), (8, 16, 16), (16, 32, 32)], mode="nearest")
            finally:
                # close reader and writer
                self.close_reader_writer()

    def _infer_one_block(self, coord, skeleton):
        block_size = np.array((self.patch_size[3], *(coord[:, 1] - coord[:, 0])), dtype=np.int64)
        block_raw = np.zeros(tuple(block_size), dtype=self.infer_reader.dtype)  # CZYX
        for i in range(self.patch_size[3]):
            block_raw[i, ...] = self.infer_reader[self.infer_channel[i], coord[0, 0]:coord[0, 1], coord[1, 0]:coord[1, 1], coord[2, 0]:coord[2, 1],]
        block_raw = preprocess_img(block_raw, self.preprocess_dict)
        # Accumulate weighted probabilities instead of allowing a later patch to overwrite an earlier prediction in overlapping regions.
        block_pred_sum = np.zeros((self.out_channel_num, *block_size[1:4]), dtype=np.float32,)
        # The same spatial blending weight is used for every output channel.
        block_weight_sum = np.zeros(tuple(block_size[1:4]), dtype=np.float32)
        block_result = np.zeros((self.postpro_channel_num, *block_size[1:4]),dtype=np.uint16,)
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
        # start post_processing
        print(f"{time_str()} [BLOCK{self.inferred_block_num:07d}/{self.total_block_num:07d}] [INFER] start post-processing",flush=True)
        skeleton_label = self.skeleton_manager.create_cropped_skeleton_mask(skeleton, coord)
        _start_id = 0
        if self.instance_num:
            block_result[_start_id, ...] = watershed_with_sk(
                        block_pred[0:self.instance_num, ...],
                        self.output_channel_names[0:self.instance_num],
                        self.watershed_seed_channels,
                        self.watershed_seed_channels_thresh,
                        self.watershed_topographic_channel,
                        self.watershed_growth_mask_channels,
                        self.watershed_growth_mask_channels_thresh,
                        skeleton_label=skeleton_label)
            block_result[_start_id, ...] = post_processing_instance(block_result[_start_id, ...], self.postprocess_dict)
            _start_id += 1
        if self.semantic_num:
            block_result[_start_id:, ...] = np.uint16(block_pred[self.instance_num:, ...] > 0.5)
            block_result[_start_id:, ...] = post_processing_semantic(block_result[_start_id:, ...], self.postprocess_dict)
        return block_result

    def _infer_one_batch(
        self,
        block_pred_sum,
        block_weight_sum,
        block_raw,
        p_coord_list,
        p_overlap_widths,
    ):
        current_batch_size = len(p_coord_list)
        if current_batch_size == 0:
            return block_pred_sum, block_weight_sum
        if current_batch_size != len(p_overlap_widths):
            raise ValueError("p_coord_list and p_overlap_widths must have the same length.")
        patch_raw = np.zeros((current_batch_size, self.patch_size[3], *self.patch_size[0:3],), dtype=block_raw.dtype,)  # BCZYX
        for i, p_coord in enumerate(p_coord_list):
            patch = block_raw[
                :, p_coord[0, 0]:p_coord[0, 1], p_coord[1, 0]:p_coord[1, 1],p_coord[2, 0]:p_coord[2, 1],]  # CZYX
            if patch.shape[1:4] != patch_raw.shape[2:5]:
                patch = reflect_padding_img(patch, patch_raw.shape[2:5])
            patch_raw[i, ...] = patch

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

    def save_infer_log(self) -> Path:
        self.infer_log_path = Path(self.cfg.DATA.INFER.INFER_LOG)
        self.infer_log_path.mkdir(parents=True, exist_ok=True)
        log_path = self.infer_log_path / f"infer_log_{int(self.job_id):02d}.json"
        payload = {
            "job_id": int(self.job_id),
            "inferred_block_num": self.inferred_block_num,
            "ome_zarr_path": str(self.infer_gt_path),
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
            return
        if not log_path.is_file():
            return
        try:
            with log_path.open(mode="r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Inference log is not valid JSON: {log_path}") from error
        except OSError as error:
            raise RuntimeError(f"Unable to read inference log: {log_path}") from error
        saved_job_id = payload.get("job_id")
        if isinstance(saved_job_id, bool) or not isinstance(saved_job_id, int) or saved_job_id != int(self.job_id):
            raise ValueError(f"Inference log belongs to a different job: saved job_id={saved_job_id!r}, current job_id={self.job_id!r}.")
        inferred_block_num = payload.get("inferred_block_num")
        if isinstance(inferred_block_num, bool) or not isinstance(inferred_block_num, int) or inferred_block_num < 0:
            raise ValueError( f"Inference log contains an invalid inferred_block_num: {inferred_block_num!r}.")
        self.inferred_block_num = inferred_block_num
        infer_gt_path = payload.get("ome_zarr_path")
        if not isinstance(infer_gt_path, str) or not infer_gt_path.strip():
            raise ValueError("Inference log contains an invalid ome_zarr_path.")
        infer_gt_path = Path(infer_gt_path)
        assert infer_gt_path.samefile(self.infer_gt_path), "zarr path in config file and log file are different."


    def run(self):
        if self.cfg.TRAIN.ENABLE:
            print(f"{time_str()} start instance segmentation training", flush=True)
            self.train()
        elif self.cfg.INFER.ENABLE:
            print(f"{time_str()} start instance segmentation inference", flush=True)
            self.infer()
        self.log_writer.close()
