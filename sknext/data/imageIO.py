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


def patch_coordinates_iter(
    img_shape: tuple[int, int, int] | list[int],
    patch_size: tuple[int, int, int] | list[int],
    overlap: tuple[int, int, int] | list[int] = (0,0,0),
    padding: tuple[int, int, int] | list[int] = (0,0,0),
) -> Iterator[np.ndarray]:
    """
    Generate 3D patch coordinates lazily.

    Parameters
    ----------
    img_shape : sequence of int
        Image shape in ZYX order.

    patch_size : sequence of int
        Patch size in ZYX order.

    overlap : sequence of int
        Overlap size in voxels, in ZYX order.

    padding : sequence of int
        Invalid border width on each side of a predicted patch,
        in ZYX order.

    Yields
    ------
    np.ndarray
        Coordinate array with shape (3, 2):
        [
            [z_min, z_max],
            [y_min, y_max],
            [x_min, x_max],
        ]
        Maximum coordinates are exclusive, so they can be used
        directly for NumPy slicing.
    """
    if len(img_shape) != 3:
        raise ValueError("img_shape must contain three values in ZYX order.")
    if len(patch_size) != 3:
        raise ValueError("patch_size must contain three values in ZYX order.")
    if len(overlap) != 3:
        raise ValueError("overlap must contain three values in ZYX order.")
    if len(padding) != 3:
        raise ValueError("padding must contain three values in ZYX order.")
    img_shape_array = np.asarray(img_shape, dtype=np.int64)
    patch_size_array = np.asarray(patch_size, dtype=np.int64)
    overlap_array = np.asarray(overlap, dtype=np.int64)
    padding_array = np.asarray(padding, dtype=np.int64)
    if np.any(img_shape_array <= 0):
        raise ValueError(f"All image dimensions must be positive, got {img_shape_array.tolist()}.")
    if np.any(patch_size_array <= 0):
        raise ValueError(f"All patch dimensions must be positive, got {patch_size_array.tolist()}.")
    if np.any(overlap_array < 0):
        raise ValueError(f"overlap must be non-negative, got {overlap_array.tolist()}.")
    if np.any(padding_array < 0):
        raise ValueError(f"padding must be non-negative, got {padding_array.tolist()}.")
    # Distance between the starting positions of adjacent patches.
    step = (patch_size_array - 2 * padding_array - overlap_array)
    if np.any(step <= 0):
        raise ValueError(f"patch_size - 2 * padding - overlap must be positive on every axis, but got step={step.tolist()}.")

    def axis_start_positions(
        axis_size: int,
        axis_patch_size: int,
        axis_step: int,
    ) -> list[int]:
        """
        Generate patch starting positions along one axis.

        The final patch is aligned with the end of the image so that it
        normally retains the requested patch size.
        """
        # The image is smaller than one patch. The returned coordinate
        # will be clipped to the image boundary, and the caller can pad it.
        if axis_size <= axis_patch_size:
            return [0]
        last_start = axis_size - axis_patch_size
        starts = list(range(0,last_start + 1,axis_step,))
        # Ensure that the final patch reaches the image boundary.
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    z_starts = axis_start_positions(
        int(img_shape_array[0]),
        int(patch_size_array[0]),
        int(step[0]),
    )
    y_starts = axis_start_positions(
        int(img_shape_array[1]),
        int(patch_size_array[1]),
        int(step[1]),
    )
    x_starts = axis_start_positions(
        int(img_shape_array[2]),
        int(patch_size_array[2]),
        int(step[2]),
    )

    for z_min in z_starts:
        z_max = min(
            z_min + int(patch_size_array[0]),
            int(img_shape_array[0]),
        )
        for y_min in y_starts:
            y_max = min(
                y_min + int(patch_size_array[1]),
                int(img_shape_array[1]),
            )
            for x_min in x_starts:
                x_max = min(
                    x_min + int(patch_size_array[2]),
                    int(img_shape_array[2]),
                )
                yield np.asarray(
                    [
                        [z_min, z_max],
                        [y_min, y_max],
                        [x_min, x_max],
                    ],
                    dtype=np.int64,
                )


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


def get_coord_after_padding(img_shape: tuple[int, int, int] | list[int, int, int],
                            coord: np.ndarray,
                            padding: tuple[int, int, int] | list[int, int, int] | np.ndarray)->tuple(np.ndarray):
    """
        Expand a 3D coordinate by padding while clipping it to image boundaries.

        Parameters
        ----------
        img_shape : tuple or list of int
            Image shape in ZYX order: (Z, Y, X).

        coord : np.ndarray
            Coordinate array with shape (3, 2):

            [
                [z_min, z_max],
                [y_min, y_max],
                [x_min, x_max],
            ]

            Maximum coordinates are exclusive.

        padding : tuple, list or np.ndarray
            Padding size in ZYX order: (z_padding, y_padding, x_padding).

        Returns
        -------
        np.ndarray
            Padded coordinates with shape (3, 2):

            [
                [z_min_padded, z_max_padded],
                [y_min_padded, y_max_padded],
                [x_min_padded, x_max_padded],
            ]

            Coordinates are clipped to the valid image range.
        """
    img_shape_array = np.asarray(img_shape, dtype=np.int64)
    coord_array = np.asarray(coord, dtype=np.int64)
    padding_array = np.asarray(padding, dtype=np.int64)
    if img_shape_array.shape != (3,):
        raise ValueError(f"`img_shape` must contain three values in ZYX order, but got shape {img_shape_array.shape}.")
    if coord_array.shape != (3, 2):
        raise ValueError(f"`coord` must have shape (3, 2), formatted as [[zmin, zmax], [ymin, ymax], [xmin, xmax]], but got shape {coord_array.shape}.")
    if padding_array.shape != (3,):
        raise ValueError("`padding` must contain three values in ZYX order, "f"but got shape {padding_array.shape}.")
    if np.any(img_shape_array <= 0):
        raise ValueError(f"All values in `img_shape` must be positive, got {img_shape_array.tolist()}.")
    if np.any(padding_array < 0):
        raise ValueError(f"All values in `padding` must be non-negative, got {padding_array.tolist()}.")
    coord_min = coord_array[:, 0]
    coord_max = coord_array[:, 1]
    if np.any(coord_min < 0):
        raise ValueError(f"Coordinate minimum values must be non-negative, got {coord_min.tolist()}.")
    if np.any(coord_min >= coord_max):
        raise ValueError(f"Every minimum coordinate must be smaller than its corresponding maximum coordinate, got {coord_array.tolist()}.")
    if np.any(coord_max > img_shape_array):
        raise ValueError(f"`coord` exceeds the image boundaries: coord={coord_array.tolist()}, img_shape={img_shape_array.tolist()}.")
    padded_coord = np.empty((3, 2), dtype=np.int64)
    padded_coord[:, 0] = np.maximum(coord_min - padding_array,0,)
    padded_coord[:, 1] = np.minimum(coord_max + padding_array, img_shape_array,)
    # Coordinates of the original region relative to the padded image.
    center_coord_in_padded = np.empty((3, 2), dtype=np.int64)
    center_coord_in_padded[:, 0] = (coord_min - padded_coord[:, 0])
    center_coord_in_padded[:, 1] = (coord_max - padded_coord[:, 0])
    return padded_coord, center_coord_in_padded
