import torch
import numpy as np
from pathlib import Path
import zarr
import random
from zarr.storage import LocalStore
from sknext.data.augmentor import *
from sknext.utils.utils import get_cfg_value


class ZarrPatchLoader:
    def __init__(self, cfg,
                 zarr_path: str | Path,
                 dataset_name: list[str] | tuple[str] = ("raw", "label"),
                 if_val: bool = False):
        self.cfg = cfg
        self.zarr_path = Path(zarr_path)
        assert self.zarr_path.exists(), "zarr_path does not exist."
        self.store = LocalStore(str(self.zarr_path))
        self.root = zarr.open_group(store=self.store, mode="r")
        self.dataset_name = dataset_name
        self.if_val = if_val
        self.arrays = {}
        for name in self.dataset_name:
            if name not in self.root:
                raise KeyError(f"Dataset '{name}' not found in zarr file. Available datasets: {list(self.root.array_keys())}")
            arr = self.root[name]
            self.arrays[name] = arr
        self.num_patches = self.arrays[dataset_name[0]].shape[0]
        self.augmentor_flag = get_cfg_value(self.cfg, "AUGMENTOR.ENABLE", True)
        self.da_prob = get_cfg_value(cfg, "AUGMENTOR.DA_PROB", 0.5)
        #augmentor
        self.cut_out = get_cfg_value(self.cfg, "AUGMENTOR.CUT_OUT", False)
        self.g_blur = get_cfg_value(self.cfg, "AUGMENTOR.G_BLUR", False)
        self.g_sigma = get_cfg_value(self.cfg, "AUGMENTOR.G_SIGMA", (1.0, 2.0))
        self.median_blur = get_cfg_value(self.cfg, "AUGMENTOR.MEDIAN_BLUR", False)
        self.mb_kernel = get_cfg_value(self.cfg, "AUGMENTOR.MB_KERNEL", (3, 7))
        self.gamma_contrast = get_cfg_value(self.cfg, "AUGMENTOR.GAMMA_CONTRAST", False)
        self.gc_gamma = get_cfg_value(self.cfg, "AUGMENTOR.GC_GAMMA", (0.8, 1.5))
        self.brightness = get_cfg_value(self.cfg, "AUGMENTOR.BRIGHTNESS", False)
        self.brightness_factor = get_cfg_value(self.cfg, "AUGMENTOR.BRIGHTNESS_FACTOR", (-0.1, 0.1))
        self.contrast = get_cfg_value(self.cfg, "AUGMENTOR.CONTRAST", False)
        self.contrast_factor = get_cfg_value(self.cfg, "AUGMENTOR.CONTRAST_FACTOR", (-0.1, 0.1))
        self.affine_mode = get_cfg_value(self.cfg, "AUGMENTOR.AFFINE_MODE", "reflect")
        self.random_rot = get_cfg_value(self.cfg, "AUGMENTOR.RANDOM_ROT", False)
        self.random_rot_range = get_cfg_value(self.cfg, "AUGMENTOR.RANDOM_ROT_RANGE", (-180, 180))
        self.shear = get_cfg_value(self.cfg, "AUGMENTOR.SHEAR", False)
        self.shear_range = get_cfg_value(self.cfg, "AUGMENTOR.SHEAR_RANGE", (-20, 20))
        self.zoom = get_cfg_value(self.cfg, "AUGMENTOR.ZOOM", False)
        self.zoom_range = get_cfg_value(self.cfg, "AUGMENTOR.ZOOM_RANGE", (0.9, 1.1))
        self.zoom_in_z = get_cfg_value(self.cfg, "AUGMENTOR.ZOOM_IN_Z", False)
        self.shift = get_cfg_value(self.cfg, "AUGMENTOR.SHIFT", False)
        self.shift_range = get_cfg_value(self.cfg, "AUGMENTOR.SHIFT_RANGE", (0.1, 0.2))
        self.elastic = get_cfg_value(self.cfg, "AUGMENTOR.ELASTIC", False)
        self.e_alpha = get_cfg_value(self.cfg, "AUGMENTOR.E_ALPHA", (12, 16))
        self.e_sigma = get_cfg_value(self.cfg, "AUGMENTOR.E_SIGMA", 4)
        self.e_mode = get_cfg_value(self.cfg, "AUGMENTOR.E_MODE", "reflect")
        self.hflip = get_cfg_value(self.cfg, "AUGMENTOR.HFLIP", False)
        self.vflip = get_cfg_value(self.cfg, "AUGMENTOR.VFLIP", False)
        self.zflip = get_cfg_value(self.cfg, "AUGMENTOR.ZFLIP", False)

    def __len__(self) -> int:
        return self.num_patches

    def load_one_pair_patch(self, id:int) -> tuple[np.ndarray, np.ndarray]:
        if id < 0 or id >= self.num_patches:
            raise IndexError(f"Patch id out of range: {id}, valid range is [0, {self.num_patches - 1}]")
        raw_arr = self.arrays[self.dataset_name[0]]
        raw_patch = np.asarray(raw_arr[id, :, :, :, :])
        raw_patch = np.transpose(raw_patch, (1, 2, 3, 0))  # CZYX -> ZYXC
        label_arr = self.arrays[self.dataset_name[1]]
        label_patch = np.asarray(label_arr[id, :, :, :, :])
        label_patch = np.transpose(label_patch, (1, 2, 3, 0))   # CZYX -> ZYXC
        if self.augmentor_flag and not self.if_val:
            raw_patch, label_patch = self.patch_augment(raw_patch, label_patch)
        raw_patch = np.transpose(raw_patch, (3, 0, 1, 2))  # ZYXC -> CZYX
        label_patch = np.transpose(label_patch, (3, 0, 1, 2))  # ZYXC -> CZYX
        return raw_patch, label_patch

    def patch_augment(self, patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray]:
        if self.g_blur and random.uniform(0, 1) < self.da_prob / 5:
            patch, mask = cutout(patch, mask)
        if self.g_blur and random.uniform(0, 1) < self.da_prob:
            patch = gaussian_blur(patch, self.g_sigma)
        if self.median_blur and random.uniform(0, 1) < self.da_prob:
            patch = median_blur(patch, self.mb_kernel)
        if self.gamma_contrast and random.uniform(0, 1) < self.da_prob:
            patch = gamma_contrast(patch, self.gc_gamma)
        if self.brightness and random.uniform(0, 1) < self.da_prob:
            patch = brightness(patch, self.brightness_factor)
        if self.contrast and random.uniform(0, 1) < self.da_prob:
            patch = contrast(patch, self.contrast_factor)
        if self.random_rot and random.uniform(0, 1) < self.da_prob:
            patch, mask, _ = random_rot(patch, mask, None, self.random_rot_range, mode=self.affine_mode)
        if self.shear and random.uniform(0, 1) < self.da_prob:
            patch, mask, _ = shear(patch, self.shear_range, mask, None, mode=self.affine_mode)
        if self.zoom and random.uniform(0, 1) < self.da_prob:
            patch, mask, _ = zoom(patch, self.zoom_range, mask, None, zoom_in_z=self.zoom_in_z, mode=self.affine_mode)
        if self.shift and random.uniform(0, 1) < self.da_prob:
            patch, mask, _ = shift(patch, mask, None, self.shift_range, mode=self.affine_mode)
        if self.elastic and random.uniform(0, 1) < self.da_prob:
            patch, mask, _ = elastic(patch, mask, None, self.e_alpha, self.e_sigma, mode = self.e_mode)
        if self.hflip and random.random() < self.da_prob:
            patch, mask, _ = flip_horizontal(patch, mask, None)
        if self.vflip and random.random() < self.da_prob:
            patch, mask, _ = flip_vertical(patch, mask, None)
        if self.zflip and random.random() < self.da_prob:
            patch = np.flip(patch, axis=0)
            mask = np.flip(mask, axis=0)
        patch = np.ascontiguousarray(patch)
        mask = np.ascontiguousarray(mask)
        return patch, mask

    def __getitem__(self, id: int|slice) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(id, int):
            img, mask = self.load_one_pair_patch(id)
            img = torch.from_numpy(img.astype('float32'))
            mask = torch.from_numpy(mask.astype('float32'))
            return img, mask
        elif isinstance(id, slice):
            indices = range(*id.indices(self.num_patches))
            imgs = []
            masks = []
            for i in indices:
                img, mask = self.load_one_pair_patch(i)
                imgs.append(torch.from_numpy(img.astype("float32")))
                masks.append(torch.from_numpy(mask.astype("float32")))
            return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)
