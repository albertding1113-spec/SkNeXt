import os
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from yacs.config import CfgNode as CN
from typing import Any, Sequence


def set_seed(seed:int=42):
    """
        Set random seed for reproducibility.

        Parameters
        ----------
        seed : int
            Random seed.

        deterministic : bool
            If True, use deterministic CUDA algorithms as much as possible.
            This improves reproducibility but may reduce training speed.
        """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def get_cfg_value(
    cfg: CN,
    key: str | Sequence[str],
    default: Any = None,
    sep: str = "."
) -> Any:
    """
    support：
    1. "DATA.PATCH_SIZE"
    2. ["DATA", "PATCH_SIZE"]
    """
    if cfg is None or key is None:
        return default
    if isinstance(key, str):
        keys = key.split(sep)
    else:
        keys = list(key)
    try:
        value = cfg
        for k in keys:
            if isinstance(value, CN):
                value = getattr(value, k)
            elif isinstance(value, dict):
                value = value[k]
            else:
                return default
        return value
    except (AttributeError, KeyError, TypeError):
        return default


def time_str():
    now = datetime.now()
    return f"[{now.year:04d}.{now.month:02d}.{now.day:02d}||{now.hour:02d}:{now.minute:02d}:{now.second:02d}]"


