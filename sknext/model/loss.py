import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from torchmetrics import JaccardIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_msssim import SSIM
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union, Sequence


def jaccard_index_numpy(y_true, y_pred):
    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)
    return jac


def weight_binary_ratio(target: torch.Tensor, min_ratio: float = 5e-2) -> torch.Tensor:
    """
    Generate a voxel-wise weight map for one binary target channel.

    target shape:
        (B, 1, Z, Y, X) or (B, 1, Y, X)

    Logic:
        - If foreground is rare, increase foreground weight.
        - If background is rare, increase background weight.
        - This is useful for contour / skeleton / affinity channels.
    """
    target = target.float()

    if torch.max(target) == torch.min(target):
        return torch.ones_like(target, dtype=torch.float32)

    label = (target != 0).float()
    fg_ratio = label.mean().clamp(min=min_ratio, max=1.0 - min_ratio)

    high = torch.maximum(fg_ratio, 1.0 - fg_ratio)
    low = torch.minimum(fg_ratio, 1.0 - fg_ratio)
    weight_factor = high / low

    # 如果前景占比更大，说明背景更少，给背景高权重
    # 如果前景占比更小，说明前景更少，给前景高权重
    if fg_ratio > 1.0 - fg_ratio:
        label = 1.0 - label  # here the label meeans the ground should have higher weight

    weight = weight_factor * label + (1.0 - label)
    return weight.float()


class instance_segmentation_loss(nn.Module):
    """
    Multi-channel binary instance segmentation loss.

    Suitable for:
        F: foreground
        C: contour
        P: point / skeleton / centroid
        A: affinity maps

    pred:
        raw logits, shape (B, C, Z, Y, X)

    target:
        binary target, shape (B, C, Z, Y, X)
    """

    def __init__(
        self,
        channel_weights: Optional[Sequence[float]] = None,
        channel_names: Optional[Sequence[str]] = None,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        auto_balance: bool = True,
        eps: float = 1e-6,
        return_dict: bool = False,
    ):
        super().__init__()

        assert bce_weight >= 0, "bce_weight must be >= 0."
        assert dice_weight >= 0, "dice_weight must be >= 0."
        assert bce_weight + dice_weight > 0, "At least one of BCE or Dice must be enabled."

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.auto_balance = auto_balance
        self.eps = eps
        self.return_dict = return_dict

        if channel_weights is None:
            self.channel_weights = None
        else:
            channel_weights = torch.tensor(channel_weights, dtype=torch.float32)
            self.register_buffer("channel_weights", channel_weights)

        self.channel_names = list(channel_names) if channel_names is not None else None

    def _dice_loss_one_channel(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_logits:
            (B, 1, Z, Y, X)

        target:
            (B, 1, Z, Y, X)
        """
        pred_prob = torch.sigmoid(pred_logits)

        reduce_dims = tuple(range(2, pred_prob.ndim))

        intersection = torch.sum(pred_prob * target, dim=reduce_dims)
        denominator = torch.sum(pred_prob, dim=reduce_dims) + torch.sum(target, dim=reduce_dims)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()

    def _bce_loss_one_channel(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.auto_balance:
            weight_map = weight_binary_ratio(target)
            return F.binary_cross_entropy_with_logits(
                pred_logits,
                target,
                weight=weight_map,
                reduction="mean",
            )
        else:
            return F.binary_cross_entropy_with_logits(
                pred_logits,
                target,
                reduction="mean",
            )

    def forward(
        self,
        pred: torch.Tensor | dict,
        target: torch.Tensor,
    ) -> torch.Tensor | dict:
        """
        pred:
            Tensor or {"pred": Tensor}

        target:
            Tensor, expected shape (B, C, Z, Y, X)
        """
        if isinstance(pred, dict):
            pred = pred["pred"]

        if target.dtype != torch.float32:
            target = target.float()

        # 兼容 DataLoader 直接返回的 BZYXC 格式
        # 如果 target 是 (B, Z, Y, X, C)，自动转成 (B, C, Z, Y, X)
        if target.ndim == pred.ndim and target.shape[1] != pred.shape[1] and target.shape[-1] == pred.shape[1]:
            target = target.permute(0, 4, 1, 2, 3).contiguous()

        assert pred.shape == target.shape, (
            f"pred and target must have the same shape. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

        assert pred.ndim in [4, 5], (
            "pred must be 4D for 2D segmentation or 5D for 3D segmentation. "
            f"Got shape: {tuple(pred.shape)}"
        )

        n_channels = pred.shape[1]

        if self.channel_weights is None:
            channel_weights = torch.ones(n_channels, device=pred.device, dtype=pred.dtype)
        else:
            assert len(self.channel_weights) == n_channels, (
                f"channel_weights length must match output channels. "
                f"Got {len(self.channel_weights)} weights but pred has {n_channels} channels."
            )
            channel_weights = self.channel_weights.to(device=pred.device, dtype=pred.dtype)

        total_loss = pred.new_tensor(0.0)
        loss_items = {}

        for c in range(n_channels):
            pred_c = pred[:, c:c + 1]
            target_c = target[:, c:c + 1]

            bce = self._bce_loss_one_channel(pred_c, target_c)
            dice = self._dice_loss_one_channel(pred_c, target_c)

            one_loss = self.bce_weight * bce + self.dice_weight * dice
            weighted_loss = channel_weights[c] * one_loss

            total_loss = total_loss + weighted_loss

            if self.return_dict:
                name = self.channel_names[c] if self.channel_names is not None else f"ch{c}"
                loss_items[f"{name}_bce"] = bce.detach()
                loss_items[f"{name}_dice"] = dice.detach()
                loss_items[f"{name}_loss"] = one_loss.detach()

        total_loss = total_loss / channel_weights.sum().clamp_min(self.eps)

        if self.return_dict:
            loss_items["loss"] = total_loss
            return loss_items

        return total_loss


class instance_segmentation_metrics:
    def __init__(
        self,
        channel_names: list[str],
        threshold: float = 0.5,
        from_logits: bool = True,
        zero_division: float = 1.0,
    ):
        """
        Calculate Jaccard index / IoU for each instance segmentation channel.

        Parameters
        ----------
        channel_names:
            Names of output channels, e.g. ["F", "C", "P"] or ["A_z_1", "A_y_1", "A_x_1"].

        threshold:
            Threshold used to binarize prediction probabilities.

        from_logits:
            If True, pred is treated as raw logits and sigmoid will be applied.
            If False, pred is treated as probability map.

        zero_division:
            Value returned when union == 0.
            Usually:
                1.0 means empty prediction and empty target are considered perfectly correct.
                0.0 means empty-empty case is treated as zero IoU.
        """
        assert len(channel_names) > 0, "channel_names cannot be empty."
        assert 0.0 <= threshold <= 1.0, "threshold must be in [0, 1]."
        assert zero_division in [0.0, 1.0], "zero_division should be 0.0 or 1.0."
        self.channel_names = channel_names
        self.threshold = threshold
        self.from_logits = from_logits
        self.zero_division = zero_division

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor | dict, target: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        pred:
            Raw logits or probability map.

            Expected shape:
                (B, C, Z, Y, X)

            If pred is dict, use pred["pred"].

        target:
            Binary target.

            Expected shape:
                (B, C, Z, Y, X)

        Returns
        -------
        dict:
            Jaccard index of each channel and mean Jaccard index.
        """
        if isinstance(pred, dict):
            pred = pred["pred"]
        assert isinstance(pred, torch.Tensor), "pred must be a torch.Tensor or dict containing key 'pred'."
        assert isinstance(target, torch.Tensor), "target must be a torch.Tensor."
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}")
        assert pred.ndim in [4, 5], (
            "pred must be 4D for 2D segmentation or 5D for 3D segmentation. "
            f"Got pred shape: {tuple(pred.shape)}")
        batch_size, channel_num = pred.shape[0], pred.shape[1]
        assert channel_num == len(self.channel_names), (
            f"Number of channels in pred does not match channel_names. "
            f"pred has {channel_num} channels, but channel_names has {len(self.channel_names)} names.")
        if self.from_logits:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        pred_bin = pred_prob > self.threshold
        target_bin = target > 0.5
        metrics = {}
        jaccard_list = []
        # reduce over B, Z, Y, X / or B, Y, X
        reduce_dims = tuple(dim for dim in range(pred_bin.ndim) if dim != 1)
        intersection = torch.sum(pred_bin & target_bin, dim=reduce_dims).float()
        union = torch.sum(pred_bin | target_bin, dim=reduce_dims).float()
        for c, channel_name in enumerate(self.channel_names):
            if union[c] == 0:
                jaccard = pred.new_tensor(self.zero_division, dtype=torch.float32)
            else:
                jaccard = intersection[c] / union[c]
            metrics[f"{channel_name}"] = float(jaccard.item())
            jaccard_list.append(jaccard)
        return metrics
