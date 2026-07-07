from typing import Dict
import torch
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from pathlib import Path
from tensorboardX import SummaryWriter


class Base_Workflow(ABC):
    def __init__(self, cfg, device:torch.device, job_id:int):
        self.cfg = cfg
        self.device = device
        self.job_id = job_id
        self.result_dir = cfg.PATHS.RESULT_DIR.PATH_
        self.checkpoint_dir = cfg.PATHS.CHECKPOINT_DIR
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_{self.job_id:02d}.pth")
        self.log_dir = cfg.PATHS.RESULT_DIR.OUTPUT_LOG
        self.log_writer = SummaryWriter(log_dir=self.log_dir)
        self.chart_dir = cfg.PATHS.RESULT_DIR.OUTPUT_CHART
        self.patch_size = tuple(cfg.DATA.PATCH_SIZE)
        self.norm_type = cfg.DATA.NORMALIZATION.TYPE
        self.reflect_to_complete_shape = cfg.DATA.REFLECT_TO_COMPLETE_SHAPE
        self.perc_clip_flag = cfg.DATA.NORMALIZATION.PERC_CLIP.ENABLE
        self.perc_clip_range = [cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC, cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC]
        self.normalization = True
        self.norm_type = cfg.DATA.NORMALIZATION.TYPE
        self.train_path = cfg.DATA.TRAIN.PATH
        self.train_gt_path = cfg.DATA.TRAIN.GT_PATH
        self.train_overlap = tuple(cfg.DATA.TRAIN.OVERLAP)
        self.train_overlap = np.int64(np.round(np.array(self.train_overlap) * np.array(self.patch_size[0:3])))
        self.train_padding = tuple(cfg.DATA.TRAIN.PADDING)
        self.val_path = cfg.DATA.VAL.PATH
        self.val_gt_path = cfg.DATA.VAL.GT_PATH
        self.val_overlap = tuple(cfg.DATA.VAL.OVERLAP)
        self.val_overlap = np.int64(np.round(np.array(self.val_overlap) * np.array(self.patch_size[0:3])))
        self.val_padding = tuple(cfg.DATA.VAL.PADDING)
        self.infer_path = cfg.DATA.INFER.PATH
        self.infer_gt_path = cfg.DATA.INFER.GT_PATH
        self.infer_overlap = tuple(cfg.DATA.INFER.OVERLAP)
        self.infer_overlap = np.int64(np.round(np.array(self.infer_overlap) * np.array(self.patch_size[0:3])))
        self.infer_padding = tuple(cfg.DATA.INFER.PADDING)
        self.infer_axes_order = cfg.DATA.INFER.INPUT_AXES_ORDER
        self.infer_channel = list(cfg.DATA.INFER.CHANNEL)
        self.skeleton_flag = cfg.INFER.SKELETON.ENABLE
        self.skeleton_path = cfg.DATA.INFER.SKELETON_PATH
        self.augmentor_flag = cfg.AUGMENTOR.ENABLE
        self.augm_prob = cfg.AUGMENTOR.DA_PROB
        self.get_preprocess_dict()
        self.get_postprocess_dict()
        self.model_name = cfg.MODEL.ARCHITECTURE
        self.feature_maps = cfg.MODEL.FEATURE_MAPS
        self.z_down = cfg.MODEL.Z_DOWN
        self.yx_down = cfg.MODEL.YX_DOWN
        self.isotropy = cfg.MODEL.ISOTROPY
        self.convnext_layers = cfg.MODEL.CONVNEXT_LAYERS
        self.convnext_stem_k_size = cfg.MODEL.CONVNEXT_STEM_K_SIZE
        self.load_checkpoint_flag = cfg.MODEL.LOAD_CHECKPOINT
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.optimizer_name = cfg.TRAIN.OPTIMIZER
        self.sd_prob = cfg.MODEL.CONVNEXT_SD_PROB
        self.lr = cfg.TRAIN.LR
        self.w_decay = cfg.TRAIN.W_DECAY
        self.opt_betas = cfg.TRAIN.OPT_BETAS
        self.epochs = cfg.TRAIN.EPOCHS
        self.patience = cfg.TRAIN.PATIENCE
        self.lr_scheduler_name = cfg.TRAIN.LR_SCHEDULER.NAME
        self.min_lr = cfg.TRAIN.LR_SCHEDULER.MIN_LR
        self.warmup_cosine_decay_epochs = cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS
        self.infer_log_path = cfg.PATHS.RESULT_DIR.INFER_LOG


    def get_preprocess_dict(self):
        preprocess_dict = {}
        preprocess_dict["perc_clip"] = self.perc_clip_flag
        preprocess_dict["perc_clip_range"] = self.perc_clip_range
        preprocess_dict["normalization"] = self.normalization
        preprocess_dict["norm_type"] = self.norm_type
        self.preprocess_dict = preprocess_dict

    def get_postprocess_dict(self):
        if not self.cfg.INFER.POST_PROCESSING.ENABLE:
            self.postprocess_dict = {}
            return
        operations = list(self.cfg.INFER.POST_PROCESSING.OPERATIONS)
        values = list(self.cfg.INFER.POST_PROCESSING.VALUES)
        if len(operations) != len(values):
            raise ValueError("INFER.POST_PROCESSING.OPERATIONS and VALUES must have the same length.")
        postprocess_dict = {}
        for operation, value in zip(operations, values):
            if operation == "fill_holes":
                postprocess_dict["fill_holes"] = True
            elif operation == "remove_small":
                postprocess_dict["remove_small"] = int(value)
            elif operation == "remove_large":
                postprocess_dict["remove_large"] = int(value)
            else:
                raise ValueError(f"Unsupported post-processing operation: {operation}")
        self.postprocess_dict = postprocess_dict

    @abstractmethod
    def save_patch_as_zarr(self):
        pass

    @abstractmethod
    def set_model(self):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def run(self):
        pass