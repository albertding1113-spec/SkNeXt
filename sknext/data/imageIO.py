import os
from pathlib import Path
from typing import Literal
import numpy as np
import tifffile
import zarr
import re
from typing import Iterator
from numcodecs import Blosc


def get_tif_path_in_folder(file_path: Path|str) -> list[Path|str]:
    file_path = Path(file_path)
    assert file_path.exists(), f"Folder does not exist: {file_path}"
    assert file_path.is_dir(), f"Input path is not a folder: {file_path}"
    tif_paths = [p.resolve() for p in file_path.iterdir() if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]]
    tif_paths = sorted(tif_paths)
    return tif_paths

def get_tif_path_dict_in_folder(file_path: Path|str, channel_names: list[str]) -> dict[str, Path|str]:
    file_path = Path(file_path)
    assert file_path.exists(), f"Folder does not exist: {file_path}"
    assert file_path.is_dir(), f"Input path is not a folder: {file_path}"
    semantic_names = []
    path_dict = {}
    for name in channel_names:
        if name in ["F", "P", "C", "A"]:
            path_dict["instance"] = []
        elif "S." in name:
            semantic_names.append(name[2:])
            path_dict[name] = []
        else: raise ValueError(f"Unsupported channel name: {name}")
    tif_length = 0
    for one_p in file_path.iterdir():
        if one_p.is_file() and one_p.suffix.lower() in [".tif", ".tiff"]:
            tif_length += 1
            one_c = str(one_p).split(".")[-2]
            if one_c in semantic_names:
                path_dict["S."+one_c].append(one_p.resolve())
            else:
                path_dict["instance"].append(one_p.resolve())
    for key in path_dict.keys(): path_dict[key] = sorted(path_dict[key])
    tif_length /= len(path_dict.keys())
    for name in path_dict.keys(): assert len(path_dict[name]) == tif_length, "unmatching number of different channels"
    return path_dict

def read_one_3D_tif(tif_path: str|Path, output_axes:Literal["ZYX", "CZYX", "ZYXC"] = "CZYX") -> np.ndarray:
    tif_path = Path(tif_path)
    img = tifffile.imread(str(tif_path))
    assert img.ndim == 3 or img.ndim == 4, "tiff image must be 3D or 4D."
    if output_axes == "ZYX":
        if img.ndim == 3:
            pass
        elif img.ndim == 4:
            raise AttributeError("tiff image must be 3D when output_axes is ZYX")
    elif output_axes == "CZYX":
        if img.ndim == 3:
            img = img[None, ...]
        elif img.ndim == 4:
            pass
    elif output_axes == "ZYXC":
        if img.ndim == 3:
            img = img[..., None]
        elif img.ndim == 4:
            img = np.transpose(img, (1, 2, 3, 0))
    return img


def calculate_patch_coordinates(img_shape: tuple[int]|list[int], # CZYX / ZYX
                                patch_size: tuple[int, int, int]|list[int, int, int],
                                overlap: tuple[int, int, int]|list[int, int, int],
                                padding: tuple[int, int, int]|list[int, int, int],) -> np.ndarray:
    if len(img_shape) == 4:
        img_shape = img_shape[1:]
    assert len(img_shape) == 3, "Input image must be 3D or 4D."
    step = np.array(patch_size, dtype='int32') - 2*np.array(padding, dtype='int32') - np.array(overlap, dtype='int32')
    assert np.all(step > 0), "Step of each patch should be larger than zero."
    step_count = np.ceil(np.array(img_shape,dtype='float32') / step.astype("float32"))
    step_count = step_count.astype('uint32')
    patch_coordinates = np.zeros((np.prod(step_count), 3, 2), dtype="uint32")
    patch_id = 0
    for i in range(step_count[0]):
        z_min = i * step[0]
        z_max = min((i + 1) * step[0], img_shape[0])
        for j in range(step_count[1]):
            y_min = j * step[1]
            y_max = min((j + 1) * step[1], img_shape[1])
            for k in range(step_count[2]):
                x_min = k * step[2]
                x_max = min((k + 1) * step[2], img_shape[2])
                patch_coordinates[patch_id, :, 0] = [z_min, y_min, x_min]
                patch_coordinates[patch_id, :, 1] = [z_max, y_max, x_max]
                patch_id += 1
    return patch_coordinates


def patch_coordinates_generator(img:np.ndarray, # CZYX / ZYX
                                patch_coordinates:np.ndarray)->Iterator[np.ndarray, np.ndarray]:
    if img.ndim == 4:
        for i in range(patch_coordinates.shape[0]):
            one_patch = img[:, patch_coordinates[i, 0, 0]:patch_coordinates[i, 0, 1],
            patch_coordinates[i, 1, 0]:patch_coordinates[i, 1, 1],
            patch_coordinates[i, 2, 0]:patch_coordinates[i, 2, 1]]
            yield one_patch, patch_coordinates[i, :, :]
    elif img.ndim == 3:
        for i in range(patch_coordinates.shape[0]):
            one_patch = img[patch_coordinates[i, 0, 0]:patch_coordinates[i, 0, 1],
            patch_coordinates[i, 1, 0]:patch_coordinates[i, 1, 1],
            patch_coordinates[i, 2, 0]:patch_coordinates[i, 2, 1]]
            yield one_patch, patch_coordinates[i, :, :]
