from typing import Literal
import numpy as np


def preprocess_img(img: np.ndarray,
                   preprocess_dict:dict) -> np.ndarray:
    assert img.ndim in (4, 5), "img must be 4D in CZYX axes or 5D in CBZYX axes."

    for i in range(img.shape[0]):
        new_img = img[i, ...].copy()
        # start percentile_clip_img
        perc_clip = preprocess_dict.get("perc_clip", False)
        perc_clip_range = preprocess_dict.get("perc_clip_range", None)
        if perc_clip and (perc_clip_range is not None):
            new_img = percentile_clip_img(new_img, perc_clip_range[0], perc_clip_range[1])
        # start normalization
        normalization = preprocess_dict.get("normalization", False)
        norm_type = preprocess_dict.get("norm_type", "zero_mean_unit_variance")
        if normalization:
            assert norm_type in ["scale_range", "zero_mean_unit_variance"], "please provcide normalization type."
            new_img = normalize_img(new_img, norm_type)
        if i==0:
            returned_img = np.zeros_like(img, dtype=new_img.dtype)
        returned_img[i, ...] = new_img
    return returned_img


def percentile_clip_img(img: np.ndarray, lower: float | int, upper: float | int ) -> np.ndarray:
    assert 0 <= lower <= 100, "lower must be in [0, 100]."
    assert 0 <= upper <= 100, "upper must be in [0, 100]."
    assert lower < upper, "lower must be smaller than upper."
    lower_value = np.percentile(img, lower)
    upper_value = np.percentile(img, upper)
    clipped_img = np.clip(img, lower_value, upper_value)
    return clipped_img

def normalize_img(img: np.ndarray, type: Literal["scale_range", "zero_mean_unit_variance"]) -> np.ndarray:
    eps = 1e-8
    new_img = img.astype('float32')
    if type == "scale_range":
        img_min = np.min(new_img)
        img_max = np.max(new_img)
        new_img = (new_img - img_min) / (img_max - img_min + eps)
    elif type == "zero_mean_unit_variance":
        img_mean = np.mean(new_img)
        img_std = np.std(new_img)
        new_img = (new_img - img_mean) / (img_std + eps)
    return new_img


def reflect_padding_img(img: np.ndarray,
                        patch_size: tuple[int, int, int] | list[int, int, int],) -> np.ndarray:
    assert img.ndim == 3 or img.ndim == 4, "img must be 3D in ZYX or 4D in CZYX axes."
    if img.ndim == 4: shape_ = np.array(img.shape[1:], dtype='int32')
    elif img.ndim == 3: shape_ = np.array(img.shape, dtype='int32')
    patch_ = np.array(patch_size, dtype='int32')
    assert np.all(shape_ <= np.array(patch_size)), "patch size is samller than img shape, cannot perform reflect padding."
    if np.all(shape_ == patch_): return img
    if img.ndim == 4:
        pad_width = np.zeros((4,2), dtype='int32')
        pad_width[1:, 1] = patch_ - shape_
    elif img.ndim ==  3:
        pad_width = np.zeros((3,2), dtype='int32')
        pad_width[:, 1] = patch_ - shape_
    if np.any(pad_width[:, 1]==1): pad_mode = "edge"
    else: pad_mode = "reflect"
    new_img = np.pad(img, pad_width=pad_width, mode=pad_mode)
    return new_img