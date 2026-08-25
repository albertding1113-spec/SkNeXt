from __future__ import annotations
import numpy as np
from pathlib import Path, PurePosixPath
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from numcodecs import Blosc
import tifffile
import h5py
import math
from itertools import product
from tqdm import tqdm
import shutil
from typing import Any, Callable, Iterable, Literal, Sequence
from sknext.data.imageIO import read_one_3D_tif, calculate_patch_coordinates, patch_coordinates_generator
from sknext.data.preprocessing import preprocess_img, reflect_padding_img
from sknext.data.label import generate_channels_from_labels

def create_patch_ome_zarr(
    output_path: str | Path,
    num_patches: int,
    c_raw: int,
    c_label: int,
    patch_size: tuple[int, int, int],
    raw_dtype: np.dtype | str = "float32",
    label_dtype: np.dtype | str = "uint8",
    overwrite: bool = True,
) -> tuple[zarr.Group, zarr.Array, zarr.Array]:
    output_path = Path(output_path)
    assert num_patches > 0, "num_patches must be > 0"
    assert c_raw > 0, "c_raw must be > 0"
    assert c_label > 0, "c_label must be > 0"
    assert len(patch_size) == 3, "patch_size must be (Z, Y, X)"
    z, y, x = patch_size
    mode = "w" if overwrite else "w-"
    if overwrite:
        remove_zarr_if_exists(output_path)
    store = LocalStore(str(output_path))
    root = zarr.open_group(
        store=store,
        mode=mode,
        zarr_format=3,
    )
    # compressor for fast IO
    fast_codec = BloscCodec(
        cname="blosclz",
        clevel=1,
        shuffle="bitshuffle",
    )
    data_raw = root.create_array(
        name="raw",
        shape=(num_patches, c_raw, z, y, x),
        chunks=(1, 1, z, y, x),
        dtype=np.dtype(raw_dtype),
        fill_value=0,
        compressors=[fast_codec],
        dimension_names=("n", "c", "z", "y", "x"),
        attributes={
            "layout": "NCZYX",
            "kind": "raw",
        },
    )
    data_label = root.create_array(
        name="label",
        shape=(num_patches, c_label, z, y, x),
        chunks=(1, 1, z, y, x),
        dtype=np.dtype(label_dtype),
        fill_value=0,
        compressors=[fast_codec],
        dimension_names=("n", "c", "z", "y", "x"),
        attributes={
            "layout": "NCZYX",
            "kind": "label",
        },
    )
    root.attrs["sknext"] = {
        "type": "patch_training_dataset",
        "zarr_format": 3,
        "layout": {
            "raw": "NCZYX",
            "label": "NCZYX",
        },
        "axes": [
            {"name": "n", "type": "sample"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "num_patches": num_patches,
        "raw_dtype": str(np.dtype(raw_dtype)),
        "label_dtype": str(np.dtype(label_dtype)),
        "patch_size_zyx": [z, y, x],
        "arrays": {
            "raw": {
                "shape": [num_patches, c_raw, z, y, x],
                "chunks": [1, 1, z, y, x],
            },
            "label": {
                "shape": [num_patches, c_label, z, y, x],
                "chunks": [1, 1, z, y, x],
            },
        },
    }
    return root, data_raw, data_label


def remove_zarr_if_exists(zarr_path: str | Path):
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        return
    # 防止误删普通数据目录
    if not zarr_path.name.endswith(".zarr"):
        raise ValueError(f"Refuse to delete non-zarr path: {zarr_path}")
    if zarr_path.is_dir():
        shutil.rmtree(zarr_path)
    elif zarr_path.is_file():
        zarr_path.unlink()
    else:
        raise RuntimeError(f"Unsupported path type: {zarr_path}")


def _normalize_patch_filter(filter_dict: dict | None) -> dict:
    """Validate and normalize patch-filter configuration."""
    if filter_dict is None:
        return {"enable": False, "props": [], "values": [], "signs": []}
    if not isinstance(filter_dict, dict):
        raise TypeError(
            f"filter_dict must be a dict or None, got {type(filter_dict).__name__}."
        )

    enable = bool(filter_dict.get("enable", False))
    props = list(filter_dict.get("props", []))
    values = list(filter_dict.get("values", []))
    signs = [str(sign).lower() for sign in filter_dict.get("signs", [])]

    if not (len(props) == len(values) == len(signs)):
        raise ValueError(
            "filter props, values and signs must have the same length."
        )

    supported_props = {"mean", "label_mean"}
    supported_signs = {"gt", "ge", "lt", "le"}
    unknown_props = [prop for prop in props if prop not in supported_props]
    unknown_signs = [sign for sign in signs if sign not in supported_signs]
    if unknown_props:
        raise ValueError(
            f"Unsupported filter properties: {unknown_props}. "
            f"Supported properties are {sorted(supported_props)}."
        )
    if unknown_signs:
        raise ValueError(
            f"Unsupported filter signs: {unknown_signs}. "
            f"Supported signs are {sorted(supported_signs)}."
        )

    normalized_values = []
    for value in values:
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"Filter values must be finite, got {value!r}.")
        normalized_values.append(numeric_value)

    if enable and not props:
        raise ValueError("Patch filtering is enabled, but no properties were provided.")

    return {
        "enable": enable,
        "props": props,
        "values": normalized_values,
        "signs": signs,
    }


def _crop_patch(img: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Crop one ZYX/CZYX patch using a (3, 2) coordinate array."""
    z0, z1 = map(int, coord[0])
    y0, y1 = map(int, coord[1])
    x0, x1 = map(int, coord[2])
    if img.ndim == 4:
        return img[:, z0:z1, y0:y1, x0:x1]
    if img.ndim == 3:
        return img[z0:z1, y0:y1, x0:x1]
    raise ValueError(f"img must be CZYX or ZYX, got shape {img.shape}.")


def patch_satisfies_filter(
    raw_img: np.ndarray,
    gt_img_dict: dict[str, np.ndarray],
    coord: np.ndarray,
    patch_size: Sequence[int],
    filter_dict: dict | None,
) -> bool:
    """Return True when a candidate training patch satisfies all filters.

    Supported properties
    --------------------
    mean
        Mean intensity of the original raw patch before preprocessing. For a
        multi-channel raw image the mean is taken across all channels/voxels.

    label_mean
        Fraction of spatial voxels occupied by at least one non-zero value in
        the original GT labels. When several GT label volumes are present they
        are combined by logical OR before the fraction is calculated.

    Notes
    -----
    Candidate patches can be smaller at an image boundary. The same reflect/
    edge padding used for the saved training patch is applied before filter
    statistics are calculated, so the statistic describes the actual patch
    shape written to Zarr.
    """
    normalized = _normalize_patch_filter(filter_dict)
    if not normalized["enable"]:
        return True

    requested_props = set(normalized["props"])
    measured: dict[str, float] = {}

    if "mean" in requested_props:
        raw_patch = _crop_patch(raw_img, coord)
        raw_patch = reflect_padding_img(raw_patch, tuple(map(int, patch_size)))
        measured["mean"] = float(np.mean(raw_patch, dtype=np.float64))

    if "label_mean" in requested_props:
        if not gt_img_dict:
            raise ValueError(
                "label_mean filter requires at least one ground-truth label volume."
            )
        label_union = None
        for gt_img in gt_img_dict.values():
            gt_patch = _crop_patch(gt_img, coord)
            current_mask = gt_patch != 0
            if label_union is None:
                label_union = current_mask.copy()
            else:
                label_union |= current_mask
        assert label_union is not None
        label_union = reflect_padding_img(
            label_union,
            tuple(map(int, patch_size)),
        )
        measured["label_mean"] = float(np.mean(label_union, dtype=np.float64))

    comparisons = {
        "gt": lambda lhs, rhs: lhs > rhs,
        "ge": lambda lhs, rhs: lhs >= rhs,
        "lt": lambda lhs, rhs: lhs < rhs,
        "le": lambda lhs, rhs: lhs <= rhs,
    }

    for prop, threshold, sign in zip(
        normalized["props"],
        normalized["values"],
        normalized["signs"],
    ):
        property_value = measured[prop]
        if not np.isfinite(property_value):
            return False
        if not comparisons[sign](property_value, threshold):
            return False
    return True


def _update_patch_zarr_metadata(
    root: zarr.Group,
    data_raw: zarr.Array,
    data_label: zarr.Array,
    filter_dict: dict,
) -> None:
    """Synchronize SkNeXt metadata after Zarr arrays have been resized."""
    metadata = dict(root.attrs.get("sknext", {}))
    metadata["num_patches"] = int(data_raw.shape[0])
    arrays = dict(metadata.get("arrays", {}))
    raw_meta = dict(arrays.get("raw", {}))
    label_meta = dict(arrays.get("label", {}))
    raw_meta["shape"] = [int(value) for value in data_raw.shape]
    label_meta["shape"] = [int(value) for value in data_label.shape]
    arrays["raw"] = raw_meta
    arrays["label"] = label_meta
    metadata["arrays"] = arrays
    metadata["filter"] = {
        "enabled": bool(filter_dict["enable"]),
        "props": list(filter_dict["props"]),
        "values": [float(value) for value in filter_dict["values"]],
        "signs": list(filter_dict["signs"]),
    }
    root.attrs["sknext"] = metadata


def tif_list_to_zarr(
        tif_list: list[Path | str],
        gt_tif_dict: dict[str, Path | str],
        zarr_path: Path | str,
        patch_size: tuple[int, int, int, int] | list[int],  # ZYXC
        overlap: tuple[int, int, int] | list[int],
        padding: tuple[int, int, int] | list[int],
        preprocess_dict: dict | None = None,
        channels: list[str] | tuple[str, ...] = (),
        channel_extra_opts: dict | None = None,
        filter_dict: dict | None = None,
):
    """Convert paired TIFF volumes into an NCZYX patch Zarr dataset.

    When ``filter_dict['enable']`` is True, candidate patches that do not
    satisfy every configured condition are skipped entirely. The Zarr N axis
    is grown only by the number of accepted patches, so rejected patches never
    appear as zero-filled samples.
    """
    preprocess_dict = {} if preprocess_dict is None else preprocess_dict
    channel_extra_opts = {} if channel_extra_opts is None else channel_extra_opts
    filter_dict = _normalize_patch_filter(filter_dict)

    if len(patch_size) != 4:
        raise ValueError("patch_size must contain four values in ZYXC order.")
    spatial_patch_size = tuple(map(int, patch_size[0:3]))
    expected_raw_channels = int(patch_size[3])

    for key in gt_tif_dict.keys():
        if len(tif_list) != len(gt_tif_dict[key]):
            raise ValueError("raw and label images must have the same number of files.")
        if not (key == "instance" or key in channels):
            raise ValueError(f"Unsupported key name: {key}")

    # Remove an old output immediately. This also prevents a stale dataset from
    # surviving when every candidate patch is rejected by the filter.
    remove_zarr_if_exists(zarr_path)

    root = None
    data_raw = None
    data_label = None
    saved_patch_num = 0
    candidate_patch_num = 0
    rejected_patch_num = 0

    for i in tqdm(range(len(tif_list))):
        tif_path = tif_list[i]
        one_gt_path_dict = {key: value[i] for key, value in gt_tif_dict.items()}
        raw_img = read_one_3D_tif(tif_path, "CZYX")
        gt_img_dict = {
            key: read_one_3D_tif(value, "ZYX")
            for key, value in one_gt_path_dict.items()
        }

        if raw_img.shape[0] != expected_raw_channels:
            raise ValueError(
                f"Raw channel count ({raw_img.shape[0]}) does not match "
                f"patch_size C ({expected_raw_channels}) for {tif_path}."
            )
        for gt_img in gt_img_dict.values():
            if raw_img.shape[1:] != gt_img.shape:
                raise ValueError("raw and label image must have the same spatial shape.")

        coords = calculate_patch_coordinates(
            raw_img.shape,
            spatial_patch_size,
            overlap,
            padding,
        )
        candidate_patch_num += int(coords.shape[0])

        if filter_dict["enable"]:
            keep_flags = np.fromiter(
                (
                    patch_satisfies_filter(
                        raw_img,
                        gt_img_dict,
                        coord,
                        spatial_patch_size,
                        filter_dict,
                    )
                    for coord in coords
                ),
                dtype=bool,
                count=coords.shape[0],
            )
            kept_coords = coords[keep_flags]
            rejected_patch_num += int(coords.shape[0] - kept_coords.shape[0])
        else:
            kept_coords = coords

        kept_num = int(kept_coords.shape[0])
        if kept_num == 0:
            continue

        # Expensive preprocessing/channel generation is deferred until we know
        # this source image contributes at least one patch.
        processed_img = preprocess_img(raw_img, preprocess_dict)  # float32, CZYX
        gt_ch_list = generate_channels_from_labels(
            gt_img_dict,
            channels,
            channel_extra_opts,
        )
        if len(gt_ch_list) == 0:
            raise ValueError("No output label channels were generated.")

        old_saved_patch_num = saved_patch_num
        saved_patch_num += kept_num

        if root is None:
            root, data_raw, data_label = create_patch_ome_zarr(
                zarr_path,
                saved_patch_num,
                expected_raw_channels,
                len(gt_ch_list),
                spatial_patch_size,
                raw_dtype=processed_img.dtype,
                label_dtype="uint8",
                overwrite=False,
            )
        else:
            assert data_raw is not None and data_label is not None
            if data_label.shape[1] != len(gt_ch_list):
                raise ValueError(
                    "Generated label channel count changed between input images: "
                    f"expected {data_label.shape[1]}, got {len(gt_ch_list)}."
                )
            data_raw.resize(
                (saved_patch_num, expected_raw_channels, *spatial_patch_size)
            )
            data_label.resize(
                (saved_patch_num, len(gt_ch_list), *spatial_patch_size)
            )

        assert data_raw is not None and data_label is not None
        for local_index, coord in enumerate(kept_coords):
            patch_id = old_saved_patch_num + local_index

            raw_patch = _crop_patch(processed_img, coord)
            raw_patch = reflect_padding_img(raw_patch, spatial_patch_size)
            data_raw[patch_id, :, :, :, :] = raw_patch

            for channel_index, gt_channel in enumerate(gt_ch_list):
                label_patch = _crop_patch(gt_channel, coord)
                label_patch = reflect_padding_img(label_patch, spatial_patch_size)
                data_label[patch_id, channel_index, :, :, :] = label_patch

    if root is None or data_raw is None or data_label is None:
        criteria = ", ".join(
            f"{prop} {sign} {value}"
            for prop, sign, value in zip(
                filter_dict["props"],
                filter_dict["signs"],
                filter_dict["values"],
            )
        )
        raise ValueError(
            "No training patches were saved. "
            f"candidate_patches={candidate_patch_num}, filter_enabled={filter_dict['enable']}, "
            f"criteria=[{criteria}]."
        )

    _update_patch_zarr_metadata(root, data_raw, data_label, filter_dict)

    if filter_dict["enable"]:
        print(
            "Patch filter summary: "
            f"candidates={candidate_patch_num}, "
            f"saved={saved_patch_num}, "
            f"rejected={rejected_patch_num}",
            flush=True,
        )

    return data_raw, data_label

def read_stack_from_ims(
    ims_path: str | Path,
    channel: int,
    coordinate: Sequence[Sequence[int]],
    resolution: int =0,
    timepoint: int = 0,
) -> np.ndarray:
    ims_path = Path(ims_path)
    assert ims_path.is_file(), f"IMS doesn't exist：{ims_path}"
    coord_array = np.asarray(coordinate)
    assert coord_array.shape == (2, 3), "coordinate must be [[zmin, ymin, xmin], [zmax, ymax, xmax]]"
    assert np.issubdtype(coord_array.dtype, np.integer), "coordinate must be integer"
    coord_array = coord_array.astype(np.int64)
    assert np.all(coord_array >= 0), "coordinate must be non-negative"
    coord_min = coord_array[0]
    coord_max = coord_array[1]
    assert np.all(coord_min < coord_max), "coordinate wrong"
    assert channel >= 0, "channel must be >= 0"
    assert timepoint >= 0, "timepoint must be >= 0"
    with (h5py.File(ims_path, mode="r") as ims_file):
        dataset_group_path = "/DataSet"
        assert dataset_group_path in ims_file, "not standard IMS file"
        dataset_group = ims_file[dataset_group_path]
        data_path = (f"/DataSet/ResolutionLevel {resolution}/TimePoint {timepoint}/Channel {channel}/Data")
        assert data_path in ims_file, f"Cannot find：{data_path}\n, please if check timepoint={timepoint} 和 channel={channel} exist"
        dataset = ims_file[data_path]
        assert dataset.ndim == 3, f"dataset shape is {dataset.shape}, not 3D dataset"
        dataset_shape = np.asarray(dataset.shape, dtype=np.int64)
        assert np.all(coord_max <= dataset_shape), (f"coordinate outside of dataset: "
                                                    f"dataset shape (Z,Y,X)：{dataset_shape.tolist()}\n, "
                                                    f"coordinate min：{coord_min.tolist()}\n, "
                                                    f"coordinate max：{coord_max.tolist()}")
        zmin, ymin, xmin = coord_min
        zmax, ymax, xmax = coord_max
        img = dataset[zmin:zmax, ymin:ymax, xmin:xmax]
        return np.asarray(img)

class IMSReader:
    """Read an Imaris ``.ims`` file as a virtual 4D CZYX array.

    Imaris stores every channel in a separate 3D ``Data`` dataset. This class
    presents those datasets as one array with shape ``(C, Z, Y, X)`` and
    supports NumPy-style basic indexing with integers, slices, and one
    ellipsis.

    Parameters
    ----------
    ims_path:
        Path to the Imaris ``.ims`` file.
    channel:
        Physical channel ID or a sequence of physical channel IDs to expose.
        ``None`` exposes every channel at the selected resolution and
        timepoint. Supplying one channel keeps the previous constructor form,
        while the returned view still follows CZYX indexing with ``C=1``.
    resolution:
        Imaris resolution-level index.
    timepoint:
        Imaris timepoint index.

    Notes
    -----
    The logical C axis follows ``channel_indices``. For example, when
    ``channel=(3, 1)``, ``reader[0]`` reads physical channel 3 and ``reader[1]``
    reads physical channel 1.
    """

    def __init__(
        self,
        ims_path: str | Path,
        channel: int | Sequence[int] | None = None,
        resolution: int = 0,
        timepoint: int = 0,
    ) -> None:
        self.ims_path = Path(ims_path)
        if not self.ims_path.is_file(): raise FileNotFoundError(f"IMS file does not exist: {self.ims_path}")
        self.resolution = self._validate_non_negative_int(resolution, "resolution")
        self.timepoint = self._validate_non_negative_int(timepoint, "timepoint")
        self.file = h5py.File(self.ims_path, mode="r")
        try:
            self.base_path = (
                f"/DataSet/ResolutionLevel {self.resolution}/"
                f"TimePoint {self.timepoint}"
            )
            if self.base_path not in self.file:
                raise KeyError(
                    f"Cannot find Imaris group {self.base_path!r}. Check "
                    f"resolution={self.resolution} and timepoint={self.timepoint}."
                )

            available = self._discover_channels(self.file[self.base_path])
            self.available_channels = tuple(available)
            self.channel_indices = self._normalize_channel_selection(
                channel, self.available_channels
            )
            self.channel = channel

            self.data_paths = tuple(
                f"{self.base_path}/Channel {channel_id}/Data"
                for channel_id in self.channel_indices
            )
            self.datasets = tuple(self.file[data_path] for data_path in self.data_paths)
            self._validate_datasets()

            self._spatial_shape = tuple(int(value) for value in self.datasets[0].shape)
            self._dtype = np.result_type(*(dataset.dtype for dataset in self.datasets))
        except Exception:
            self.file.close()
            raise

    @staticmethod
    def _validate_non_negative_int(value: int, name: str) -> int:
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be greater than or equal to zero.")
        return result

    @staticmethod
    def _discover_channels(timepoint_group: h5py.Group) -> list[int]:
        channel_ids: list[int] = []
        prefix = "Channel "

        for name, value in timepoint_group.items():
            if not name.startswith(prefix) or not isinstance(value, h5py.Group):
                continue
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                continue
            data_path = f"{value.name}/Data"
            if data_path in timepoint_group.file and isinstance(
                timepoint_group.file[data_path], h5py.Dataset
            ):
                channel_ids.append(int(suffix))

        channel_ids.sort()
        if not channel_ids:
            raise ValueError(
                f"No channel Data datasets were found under {timepoint_group.name!r}."
            )
        return channel_ids

    @staticmethod
    def _normalize_channel_selection(
        channel: int | Sequence[int] | None,
        available_channels: Sequence[int],
    ) -> tuple[int, ...]:
        available = tuple(int(value) for value in available_channels)
        available_set = set(available)

        if channel is None:
            selected = available
        elif isinstance(channel, (int, np.integer)):
            selected = (int(channel),)
        else:
            if isinstance(channel, (str, bytes)):
                raise TypeError("channel must be an integer, a sequence of integers, or None.")
            selected = tuple(int(value) for value in channel)
            if not selected:
                raise ValueError("channel selection must not be empty.")

        if len(set(selected)) != len(selected):
            raise ValueError(f"channel contains duplicate IDs: {selected}")

        missing = tuple(value for value in selected if value not in available_set)
        if missing:
            raise IndexError(
                f"Requested physical channel IDs do not exist: {missing}. "
                f"Available channel IDs: {available}."
            )
        return selected

    def _validate_datasets(self) -> None:
        reference_shape: tuple[int, ...] | None = None

        for channel_id, data_path, dataset in zip(
            self.channel_indices, self.data_paths, self.datasets
        ):
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{data_path!r} is not an HDF5 dataset.")
            if dataset.ndim != 3:
                raise ValueError(
                    f"Physical channel {channel_id} must be a 3D ZYX dataset; "
                    f"actual shape={dataset.shape}."
                )

            current_shape = tuple(int(value) for value in dataset.shape)
            if reference_shape is None:
                reference_shape = current_shape
            elif current_shape != reference_shape:
                raise ValueError(
                    "All selected channels must have the same ZYX shape; "
                    f"expected {reference_shape}, but physical channel {channel_id} "
                    f"has shape {current_shape}."
                )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Virtual array shape in CZYX order."""
        return (len(self.datasets), *self._spatial_shape)

    @property
    def ndim(self) -> int:
        return 4

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dtype)

    @property
    def chunks(self) -> tuple[int, int, int, int] | None:
        """Virtual CZYX chunk shape when all selected channels are chunked alike."""
        spatial_chunks = self.datasets[0].chunks
        if spatial_chunks is None:
            return None
        if any(dataset.chunks != spatial_chunks for dataset in self.datasets[1:]):
            return None
        return (1, *(int(value) for value in spatial_chunks))

    @property
    def closed(self) -> bool:
        return not hasattr(self, "file") or not self.file.id.valid

    def close(self) -> None:
        if hasattr(self, "file") and self.file.id.valid:
            self.file.close()

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("IMSReader has been closed.")

    def __enter__(self) -> "IMSReader":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, key: Any) -> np.ndarray:
        """Read data using NumPy-style CZYX indexing.

        Examples
        --------
        ``reader[0]`` returns logical channel 0 as a ZYX array.
        ``reader[:, 10:20, 100:300, 200:500]`` returns a CZYX block.
        ``reader[1, ..., 100:200]`` returns a ZYX block from logical channel 1.
        ``reader[-1, ::-1, :, :]`` supports negative integer and slice indices.
        """
        self._ensure_open()
        normalized_key = self._normalize_key(key, self.shape)
        channel_key = normalized_key[0]
        spatial_key = normalized_key[1:]

        if isinstance(channel_key, int):
            return self._read_spatial(self.datasets[channel_key], spatial_key)

        channel_positions = list(range(*channel_key.indices(self.shape[0])))
        if not channel_positions:
            output_shape = (0, *self._indexed_shape(spatial_key, self._spatial_shape))
            return np.empty(output_shape, dtype=self.dtype)

        channel_arrays = [
            self._read_spatial(self.datasets[position], spatial_key)
            for position in channel_positions
        ]
        return np.stack(channel_arrays, axis=0)

    @staticmethod
    def _normalize_key(
        key: Any,
        shape: Sequence[int],
    ) -> tuple[int | slice, int | slice, int | slice, int | slice]:
        ndim = len(shape)
        items = list(key if isinstance(key, tuple) else (key,))

        ellipsis_positions = [
            index for index, item in enumerate(items) if item is Ellipsis
        ]
        if len(ellipsis_positions) > 1:
            raise IndexError("An index may contain at most one ellipsis (...).")
        if any(item is None for item in items):
            raise IndexError(
                "np.newaxis/None is not supported directly. Read the data first "
                "and then call np.expand_dims()."
            )

        if ellipsis_positions:
            ellipsis_index = ellipsis_positions[0]
            missing = ndim - (len(items) - 1)
            if missing < 0:
                raise IndexError(f"Too many indices for a {ndim}D CZYX array: {key!r}")
            items[ellipsis_index:ellipsis_index + 1] = [slice(None)] * missing
        elif len(items) < ndim:
            items.extend([slice(None)] * (ndim - len(items)))

        if len(items) != ndim:
            raise IndexError(f"Expected at most {ndim} CZYX indices; received {key!r}.")

        normalized: list[int | slice] = []
        for axis, (item, axis_size) in enumerate(zip(items, shape)):
            if isinstance(item, (int, np.integer)):
                index = int(item)
                if index < 0:
                    index += int(axis_size)
                if index < 0 or index >= int(axis_size):
                    raise IndexError(
                        f"Index {item} is out of bounds for axis {axis} "
                        f"with size {axis_size}."
                    )
                normalized.append(index)
            elif isinstance(item, slice):
                if item.step == 0:
                    raise ValueError("A slice step cannot be zero.")
                normalized.append(item)
            else:
                raise TypeError(
                    "Only integers, slices, and ellipsis are supported for direct "
                    f"reading; received {type(item).__name__} on axis {axis}."
                )

        return tuple(normalized)  # type: ignore[return-value]

    @staticmethod
    def _indexed_shape(
        key: Sequence[int | slice],
        shape: Sequence[int],
    ) -> tuple[int, ...]:
        output: list[int] = []
        for item, axis_size in zip(key, shape):
            if isinstance(item, int):
                continue
            output.append(len(range(*item.indices(int(axis_size)))))
        return tuple(output)

    @staticmethod
    def _read_spatial(
        dataset: h5py.Dataset,
        key: Sequence[int | slice],
    ) -> np.ndarray:
        """Read one ZYX dataset, including slices with negative steps."""
        h5_key: list[int | slice] = []
        take_operations: list[tuple[int, np.ndarray]] = []
        output_axis = 0

        for item, axis_size in zip(key, dataset.shape):
            if isinstance(item, int):
                h5_key.append(item)
                continue

            start, stop, step = item.indices(int(axis_size))
            if step > 0:
                h5_key.append(slice(start, stop, step))
            else:
                selected = np.arange(start, stop, step, dtype=np.int64)
                if selected.size == 0:
                    h5_key.append(slice(0, 0, 1))
                    take_operations.append((output_axis, selected))
                else:
                    lower = int(selected.min())
                    upper = int(selected.max()) + 1
                    h5_key.append(slice(lower, upper, 1))
                    take_operations.append((output_axis, selected - lower))
            output_axis += 1

        result = np.asarray(dataset[tuple(h5_key)])
        for axis, indices in take_operations:
            result = np.take(result, indices, axis=axis)
        return result

Coordinate4D = tuple[int, int, int, int]
Slice4D = tuple[slice, slice, slice, slice]
PyramidPolicy = Literal["remove", "keep", "error"]

class ZarrIOManager:
    """Create or reopen a CZYX OME-Zarr image and write by coordinates.

    Parameters
    ----------
    path:
        OME-Zarr directory path.
    shape:
        Target level-0 shape in ``(C, Z, Y, X)`` order. Required only when a
        new file is created. When reopening an existing file, omitting it
        causes the shape to be inferred from the existing level-0 array.
    dtype:
        Target dtype. Required only for a new file. Inferred when reopening.
    chunks:
        Chunk shape in ``(C, Z, Y, X)`` order. For a new file, the default is
        ``(1, 16, 256, 256)``. Existing files always retain their own chunks;
        an explicitly supplied value is used only as a compatibility check.
    mode:
        ``"a"`` creates the file if missing and otherwise reopens it.
        ``"r+"`` requires the file to exist and opens it for reading/writing.
        ``"r"`` requires the file to exist and opens it as read-only.
        None of these modes deletes the store.
    array_path:
        Level-0 array path inside the OME-Zarr group. For a new image the
        default is ``"0"``. For an existing image it can be omitted and is
        inferred from the first ``multiscales.datasets`` entry.
    pyramid_policy:
        Action taken before level-0 data are modified when lower-resolution
        pyramid levels already exist:

        - ``"remove"``: remove only the listed lower-resolution arrays and
          update ``multiscales`` to contain level 0 only. This is the default.
        - ``"keep"``: leave the pyramid untouched; lower levels can become
          inconsistent with level 0.
        - ``"error"``: reject the write until the caller handles the pyramid.
    consolidate_on_close:
        Generate/update Zarr v2 ``.zmetadata`` when ``close()`` is called.

    Notes
    -----
    Coordinates are always ``(c, z, y, x)``. Stop coordinates are exclusive.
    The class supports ``with`` but does not require it; call ``close()`` when
    finished if not using a context manager.
    """

    AXIS_NAMES = ("c", "z", "y", "x")
    AXES_TEMPLATE = (
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    )
    DEFAULT_CHUNKS = (1, 16, 256, 256)
    DEFAULT_COLORS = (
        "FFFFFF",
        "00FF00",
        "FF00FF",
        "00FFFF",
        "FFFF00",
        "FF0000",
        "0000FF",
        "FFA500",
    )

    def __init__(
        self,
        path: str | Path,
        shape: Sequence[int] | None = None,
        dtype: Any | None = None,
        *,
        chunks: Sequence[int] | None = None,
        mode: Literal["a", "r+", "r"] = "a",
        array_path: str | None = None,
        voxel_size: Sequence[float] | None = None,
        unit: str | None = None,
        image_name: str | None = None,
        channel_names: Sequence[str] | None = None,
        channel_colors: Sequence[str] | None = None,
        channel_windows: Sequence[Sequence[float]] | None = None,
        compressor: Any | None = None,
        fill_value: int | float = 0,
        dimension_separator: Literal["/", "."] = "/",
        pyramid_policy: PyramidPolicy = "remove",
        consolidate_on_close: bool = True,
    ) -> None:
        self.path = Path(path)
        self._closed = False
        self._metadata_dirty = False
        self._write_prepared = False
        self._read_only = mode == "r"
        self.consolidate_on_close = bool(consolidate_on_close) and not self._read_only

        if mode not in {"a", "r+", "r"}:
            raise ValueError(
                "mode must be 'a', 'r+', or 'r'; this class never deletes an existing store."
            )
        if dimension_separator not in {"/", "."}:
            raise ValueError("dimension_separator must be either '/' or '.'.")
        if pyramid_policy not in {"remove", "keep", "error"}:
            raise ValueError("pyramid_policy must be 'remove', 'keep', or 'error'.")
        self.pyramid_policy: PyramidPolicy = pyramid_policy

        store_exists = self._is_existing_zarr_group(self.path)
        self._validate_store_path(mode=mode, store_exists=store_exists)

        if not store_exists:
            if shape is None or dtype is None:
                raise ValueError("shape and dtype are required when creating a new OME-Zarr store.")
            self._create_new(
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                array_path=array_path or "0",
                voxel_size=voxel_size or (1.0, 1.0, 1.0),
                unit=unit or "micrometer",
                image_name=image_name or "image",
                channel_names=channel_names,
                channel_colors=channel_colors,
                channel_windows=channel_windows,
                compressor=compressor,
                fill_value=fill_value,
                dimension_separator=dimension_separator,
            )
        else:
            self._open_existing(
                requested_shape=shape,
                requested_dtype=dtype,
                requested_chunks=chunks,
                requested_array_path=array_path,
                open_mode=mode,
            )

    # ------------------------------------------------------------------
    # Store creation/opening
    # ------------------------------------------------------------------
    @staticmethod
    def _is_existing_zarr_group(path: Path) -> bool:
        return path.is_dir() and (path / ".zgroup").is_file()

    def _validate_store_path(self, *, mode: str, store_exists: bool) -> None:
        if self.path.exists() and not self.path.is_dir():
            raise ValueError(f"The target path exists but is not a directory: {self.path}")

        if mode in {"r", "r+"} and not store_exists:
            raise FileNotFoundError(
                f"mode={mode!r} requires an existing Zarr v2 group: {self.path}"
            )

        if self.path.is_dir() and not store_exists:
            entries = list(self.path.iterdir())
            if entries:
                raise ValueError(
                    f"The target directory is not empty and is not a Zarr v2 group; refusing to write: {self.path}"
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _create_new(
        self,
        *,
        shape: Sequence[int],
        dtype: Any,
        chunks: Sequence[int] | None,
        array_path: str,
        voxel_size: Sequence[float],
        unit: str,
        image_name: str,
        channel_names: Sequence[str] | None,
        channel_colors: Sequence[str] | None,
        channel_windows: Sequence[Sequence[float]] | None,
        compressor: Any | None,
        fill_value: int | float,
        dimension_separator: str,
    ) -> None:
        self.shape = self._validate_shape(shape, "shape")
        self.dtype = np.dtype(dtype)
        raw_chunks = chunks if chunks is not None else self.DEFAULT_CHUNKS
        self.chunks = self._normalize_chunks(raw_chunks, self.shape)
        self.array_path = self._normalize_array_path(array_path)
        self.voxel_size = self._validate_voxel_size(voxel_size)
        self.unit = str(unit)
        self.image_name = str(image_name)
        self.fill_value = fill_value

        c_size = self.shape[0]
        self.channel_names = self._normalize_channel_names(channel_names, c_size)
        self.channel_colors = self._normalize_channel_colors(channel_colors, c_size)
        self.channel_windows = self._normalize_channel_windows(
            channel_windows, c_size, self.dtype
        )

        if compressor is None:
            compressor = Blosc(
                cname="zstd",
                clevel=3,
                shuffle=Blosc.BITSHUFFLE,
            )
        self.compressor = compressor
        self.chunk_key_encoding = {
            "name": "v2",
            "configuration": {"separator": dimension_separator},
        }

        self.root = zarr.open_group(
            store=str(self.path),
            mode="a",
            zarr_format=2,
            use_consolidated=False,
        )
        self._level0 = self._create_array(
            path=self.array_path,
            shape=self.shape,
            chunks=self.chunks,
        )
        self._multiscale_index = 0
        self._write_new_ome_metadata()
        self._metadata_dirty = True

    def _open_existing(
        self,
        *,
        requested_shape: Sequence[int] | None,
        requested_dtype: Any | None,
        requested_chunks: Sequence[int] | None,
        requested_array_path: str | None,
        open_mode: Literal["r", "r+", "a"],
    ) -> None:
        zarr_open_mode = "r" if open_mode == "r" else "r+"
        self.root = zarr.open_group(
            store=str(self.path),
            mode=zarr_open_mode,
            zarr_format=2,
            use_consolidated=False,
        )

        multiscales = self.root.attrs.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            raise ValueError(
                f"The existing Zarr group does not contain OME-NGFF multiscales metadata: {self.path}"
            )

        if requested_array_path is None:
            first_datasets = multiscales[0].get("datasets", [])
            if not first_datasets or "path" not in first_datasets[0]:
                raise ValueError("No level-0 path was found in multiscales.datasets.")
            self.array_path = self._normalize_array_path(first_datasets[0]["path"])
            self._multiscale_index = 0
        else:
            self.array_path = self._normalize_array_path(requested_array_path)
            self._multiscale_index = self._find_multiscale_index(
                multiscales, self.array_path
            )

        try:
            level0 = self.root[self.array_path]
        except KeyError as exc:
            raise ValueError(
                f"The level-0 array does not exist in the OME-Zarr store: {self.array_path!r}"
            ) from exc
        if not isinstance(level0, zarr.Array):
            raise ValueError(f"{self.array_path!r} is not a Zarr array.")
        if level0.ndim != 4:
            raise ValueError(
                f"The target array must be 4D CZYX; actual shape={tuple(level0.shape)}."
            )

        self._level0 = level0
        self.shape = tuple(int(v) for v in level0.shape)
        self.dtype = np.dtype(level0.dtype)
        self.chunks = tuple(int(v) for v in level0.chunks)
        self.fill_value = getattr(level0, "fill_value", 0)
        self.compressor = getattr(level0, "compressor", None)
        self.chunk_key_encoding = None

        self._validate_existing_axes(multiscales[self._multiscale_index])
        self._check_requested_layout(
            requested_shape=requested_shape,
            requested_dtype=requested_dtype,
            requested_chunks=requested_chunks,
        )
        self._load_existing_metadata(multiscales[self._multiscale_index])

    @staticmethod
    def _find_multiscale_index(multiscales: list[Any], array_path: str) -> int:
        for index, item in enumerate(multiscales):
            for dataset in item.get("datasets", []):
                if str(dataset.get("path", "")).strip("/") == array_path:
                    return index
        raise ValueError(
            f"The specified array_path={array_path!r} is not listed in multiscales.datasets."
        )

    def _check_requested_layout(
        self,
        *,
        requested_shape: Sequence[int] | None,
        requested_dtype: Any | None,
        requested_chunks: Sequence[int] | None,
    ) -> None:
        if requested_shape is not None:
            expected = self._validate_shape(requested_shape, "shape")
            if expected != self.shape:
                raise ValueError(
                    f"The existing array has shape={self.shape}, which does not match the requested shape={expected}."
                )

        if requested_dtype is not None:
            expected_dtype = np.dtype(requested_dtype)
            if expected_dtype != self.dtype:
                raise ValueError(
                    f"The existing array has dtype={self.dtype}, which does not match the requested dtype={expected_dtype}."
                )

        if requested_chunks is not None:
            expected_chunks = self._normalize_chunks(requested_chunks, self.shape)
            if expected_chunks != self.chunks:
                raise ValueError(
                    f"The existing array has chunks={self.chunks}, which does not match the requested chunks={expected_chunks}."
                )

    def _validate_existing_axes(self, multiscale: dict[str, Any]) -> None:
        axes = multiscale.get("axes")
        if not isinstance(axes, list):
            raise ValueError("multiscales.axes is missing or has an invalid format.")
        names = tuple(
            axis.get("name") if isinstance(axis, dict) else str(axis)
            for axis in axes
        )
        if names != self.AXIS_NAMES:
            raise ValueError(
                f"This class supports only CZYX; the existing OME-Zarr axes are {names}."
            )

    def _load_existing_metadata(self, multiscale: dict[str, Any]) -> None:
        self.image_name = str(multiscale.get("name", "image"))
        axes = multiscale.get("axes", [])
        spatial_units = [
            axis.get("unit")
            for axis in axes
            if isinstance(axis, dict) and axis.get("name") in {"z", "y", "x"}
        ]
        self.unit = str(next((u for u in spatial_units if u), "micrometer"))

        self.voxel_size = (1.0, 1.0, 1.0)
        for dataset in multiscale.get("datasets", []):
            if str(dataset.get("path", "")).strip("/") != self.array_path:
                continue
            for transform in dataset.get("coordinateTransformations", []):
                if transform.get("type") == "scale":
                    scale = transform.get("scale", [])
                    if len(scale) == 4:
                        self.voxel_size = tuple(float(v) for v in scale[1:4])
                    break

        omero = self.root.attrs.get("omero", {})
        channels = omero.get("channels", []) if isinstance(omero, dict) else []
        c_size = self.shape[0]

        names: list[str] = []
        colors: list[str] = []
        windows: list[tuple[float, float, float, float]] = []
        defaults = self._default_windows(c_size, self.dtype)

        for channel_index in range(c_size):
            item = channels[channel_index] if channel_index < len(channels) else {}
            names.append(str(item.get("label", f"Channel {channel_index}")))
            color = str(
                item.get(
                    "color",
                    self.DEFAULT_COLORS[channel_index % len(self.DEFAULT_COLORS)],
                )
            ).strip().lstrip("#").upper()
            colors.append(color if self._is_valid_color(color) else "FFFFFF")

            window = item.get("window", {}) if isinstance(item, dict) else {}
            default = defaults[channel_index]
            windows.append(
                (
                    float(window.get("min", default[0])),
                    float(window.get("max", default[1])),
                    float(window.get("start", default[2])),
                    float(window.get("end", default[3])),
                )
            )

        self.channel_names = tuple(names)
        self.channel_colors = tuple(colors)
        self.channel_windows = tuple(windows)

    # ------------------------------------------------------------------
    # Validation and metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_array_path(path: str) -> str:
        normalized = str(path).replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("array_path must not be empty.")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError(f"Invalid array_path: {path!r}")
        return normalized

    @staticmethod
    def _validate_shape(values: Sequence[int], name: str) -> Coordinate4D:
        result = tuple(int(v) for v in values)
        if len(result) != 4:
            raise ValueError(f"{name} must contain four integers in CZYX order.")
        if any(v <= 0 for v in result):
            raise ValueError(f"{name} must contain only values greater than zero.")
        return result  # type: ignore[return-value]

    @staticmethod
    def _validate_coordinate(values: Sequence[int], name: str) -> Coordinate4D:
        result = tuple(int(v) for v in values)
        if len(result) != 4:
            raise ValueError(f"{name} must contain four integers in CZYX order.")
        if any(v < 0 for v in result):
            raise ValueError(f"{name} must contain only values greater than or equal to zero.")
        return result  # type: ignore[return-value]

    @classmethod
    def _normalize_chunks(
        cls, chunks: Sequence[int], shape: Sequence[int]
    ) -> Coordinate4D:
        raw = cls._validate_shape(chunks, "chunks")
        return tuple(min(int(s), int(c)) for s, c in zip(shape, raw))  # type: ignore[return-value]

    @staticmethod
    def _validate_voxel_size(values: Sequence[float]) -> tuple[float, float, float]:
        result = tuple(float(v) for v in values)
        if len(result) != 3 or any(v <= 0 for v in result):
            raise ValueError("voxel_size must contain three positive numbers in ZYX order.")
        return result  # type: ignore[return-value]

    @classmethod
    def _normalize_channel_names(
        cls, names: Sequence[str] | None, c_size: int
    ) -> tuple[str, ...]:
        if names is None:
            return tuple(f"Channel {i}" for i in range(c_size))
        result = tuple(str(value) for value in names)
        if len(result) != c_size:
            raise ValueError("The length of channel_names must equal C.")
        return result

    @classmethod
    def _is_valid_color(cls, color: str) -> bool:
        return len(color) == 6 and all(ch in "0123456789ABCDEF" for ch in color)

    @classmethod
    def _normalize_channel_colors(
        cls, colors: Sequence[str] | None, c_size: int
    ) -> tuple[str, ...]:
        if colors is None:
            return tuple(
                cls.DEFAULT_COLORS[i % len(cls.DEFAULT_COLORS)]
                for i in range(c_size)
            )
        if len(colors) != c_size:
            raise ValueError("The length of channel_colors must equal C.")

        result: list[str] = []
        for color in colors:
            normalized = str(color).strip().lstrip("#").upper()
            if not cls._is_valid_color(normalized):
                raise ValueError(f"Invalid RGB color: {color!r}.")
            result.append(normalized)
        return tuple(result)

    @staticmethod
    def _default_windows(
        c_size: int, dtype: np.dtype[Any]
    ) -> tuple[tuple[float, float, float, float], ...]:
        if np.issubdtype(dtype, np.bool_):
            low, high = 0.0, 1.0
        elif np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            low, high = float(info.min), float(info.max)
        else:
            low, high = 0.0, 1.0
        return tuple((low, high, low, high) for _ in range(c_size))

    @classmethod
    def _normalize_channel_windows(
        cls,
        windows: Sequence[Sequence[float]] | None,
        c_size: int,
        dtype: np.dtype[Any],
    ) -> tuple[tuple[float, float, float, float], ...]:
        if windows is None:
            return cls._default_windows(c_size, dtype)
        if len(windows) != c_size:
            raise ValueError("The length of channel_windows must equal C.")

        result: list[tuple[float, float, float, float]] = []
        for window in windows:
            values = tuple(float(v) for v in window)
            if len(values) == 2:
                start, end = values
                min_value, max_value = start, end
            elif len(values) == 4:
                min_value, max_value, start, end = values
            else:
                raise ValueError(
                    "Each channel_windows entry must be either (start, end) or "
                    "(min, max, start, end)."
                )
            if not min_value <= start <= end <= max_value:
                raise ValueError("Each channel window must satisfy min <= start <= end <= max.")
            result.append((min_value, max_value, start, end))
        return tuple(result)

    def _create_array(
        self,
        *,
        path: str,
        shape: Sequence[int],
        chunks: Sequence[int],
    ) -> zarr.Array:
        array = self.root.create_array(
            path,
            shape=tuple(int(v) for v in shape),
            chunks=tuple(int(v) for v in chunks),
            dtype=self.dtype,
            compressor=self.compressor,
            fill_value=self.fill_value,
            chunk_key_encoding=self.chunk_key_encoding,
            overwrite=False,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = list(self.AXIS_NAMES)
        return array

    def _level0_dataset_metadata(self) -> dict[str, Any]:
        vz, vy, vx = self.voxel_size
        return {
            "path": self.array_path,
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [1.0, vz, vy, vx],
                }
            ],
        }

    def _write_new_ome_metadata(self) -> None:
        axes = [dict(axis) for axis in self.AXES_TEMPLATE]
        for axis in axes:
            if axis["name"] in {"z", "y", "x"}:
                axis["unit"] = self.unit

        self.root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": self.image_name,
                "axes": axes,
                "datasets": [self._level0_dataset_metadata()],
            }
        ]
        self._write_omero_metadata()

    def _write_omero_metadata(self) -> None:
        channels = []
        for name, color, window in zip(
            self.channel_names, self.channel_colors, self.channel_windows
        ):
            min_value, max_value, start, end = window
            channels.append(
                {
                    "active": True,
                    "coefficient": 1.0,
                    "color": color,
                    "family": "linear",
                    "inverted": False,
                    "label": name,
                    "window": {
                        "min": min_value,
                        "max": max_value,
                        "start": start,
                        "end": end,
                    },
                }
            )

        self.root.attrs["omero"] = {
            "version": "0.4",
            "name": self.image_name,
            "channels": channels,
            "rdefs": {
                "defaultZ": self.shape[1] // 2,
                "model": "color" if self.shape[0] > 1 else "greyscale",
            },
        }

    # ------------------------------------------------------------------
    # Public properties and coordinate writes
    # ------------------------------------------------------------------
    @property
    def array(self) -> zarr.Array:
        """The existing or newly created level-0 CZYX array."""
        return self._level0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ZarrWriter is closed.")

    def _ensure_writable(self) -> None:
        self._ensure_open()
        if self._read_only:
            raise PermissionError(
                "This ZarrWriter was opened with mode='r' and is read-only."
            )

    def _prepare_level0_write(self) -> None:
        self._ensure_writable()
        if self._write_prepared:
            return

        multiscales = list(self.root.attrs.get("multiscales", []))
        multiscale = dict(multiscales[self._multiscale_index])
        datasets = list(multiscale.get("datasets", []))
        lower_levels = [
            dict(dataset)
            for dataset in datasets
            if str(dataset.get("path", "")).strip("/") != self.array_path
        ]

        if lower_levels and self.pyramid_policy == "error":
            paths = [item.get("path") for item in lower_levels]
            raise RuntimeError(
                "Writing to level 0 would invalidate the existing pyramid. "
                f"Existing lower-resolution levels: {paths}. Use pyramid_policy='remove' "
                "or explicitly choose 'keep'."
            )

        if lower_levels and self.pyramid_policy == "remove":
            for dataset in lower_levels:
                path = str(dataset.get("path", "")).strip("/")
                if path and path in self.root:
                    del self.root[path]

            level0_item = next(
                (
                    dict(dataset)
                    for dataset in datasets
                    if str(dataset.get("path", "")).strip("/") == self.array_path
                ),
                self._level0_dataset_metadata(),
            )
            multiscale["datasets"] = [level0_item]
            multiscale.pop("type", None)
            multiscale.pop("metadata", None)
            multiscales[self._multiscale_index] = multiscale
            self.root.attrs["multiscales"] = multiscales
            self._metadata_dirty = True

        self._write_prepared = True

    def _target_region(
        self, start: Sequence[int], source_shape: Sequence[int]
    ) -> Slice4D:
        start4 = self._validate_coordinate(start, "start")
        source4 = self._validate_shape(source_shape, "source.shape")
        stop = tuple(s + length for s, length in zip(start4, source4))
        if any(end > limit for end, limit in zip(stop, self.shape)):
            raise IndexError(
                "The write region exceeds the target array bounds: "
                f"start={start4}, source_shape={source4}, target_shape={self.shape}"
            )
        return tuple(slice(begin, end) for begin, end in zip(start4, stop))  # type: ignore[return-value]

    @staticmethod
    def _source_chunk_shape(source: Any, fallback: Sequence[int]) -> Coordinate4D:
        source_shape = tuple(int(v) for v in source.shape)
        source_chunks = getattr(source, "chunks", None)
        if source_chunks is None or len(source_chunks) != 4:
            raw = tuple(int(v) for v in fallback)
        else:
            raw = tuple(int(v) for v in source_chunks)
        return tuple(min(size, chunk) for size, chunk in zip(source_shape, raw))  # type: ignore[return-value]

    def write(
        self,
        source: Any,
        start: Sequence[int] = (0, 0, 0, 0),
        *,
        source_chunks: Sequence[int] | None = None,
        casting: Literal[
            "no", "equiv", "safe", "same_kind", "unsafe"
        ] = "same_kind",
        progress_callback: Callable[[int, int, Slice4D], None] | None = None,
    ) -> Slice4D:
        """Stream an arbitrary-size CZYX source into level 0 at ``start``.

        ``source`` may be a NumPy array, memmap, h5py.Dataset, zarr.Array, or
        another object exposing ``shape``, optional ``dtype``, and slicing.
        The source is copied block by block and is not loaded all at once.

        Returns
        -------
        tuple[slice, slice, slice, slice]
            Target CZYX region written to the OME-Zarr array.
        """
        self._ensure_open()
        if not hasattr(source, "shape"):
            raise TypeError("source must provide a shape attribute and support slice-based reading.")
        source_shape = tuple(int(v) for v in source.shape)
        if len(source_shape) != 4:
            raise ValueError(
                f"source must be 4D CZYX; actual shape={source_shape}."
            )
        target_region = self._target_region(start, source_shape)

        source_dtype = getattr(source, "dtype", None)
        if source_dtype is not None and not np.can_cast(
            np.dtype(source_dtype), self.dtype, casting=casting
        ):
            raise TypeError(
                f"Cannot cast source dtype {np.dtype(source_dtype)} to target dtype "
                f"{self.dtype} using casting={casting!r}. Set casting='unsafe' "
                "explicitly if this conversion is intentional."
            )

        if source_chunks is None:
            copy_chunks = self._source_chunk_shape(source, self.chunks)
        else:
            copy_chunks = self._normalize_chunks(source_chunks, source_shape)

        self._prepare_level0_write()
        start4 = tuple(item.start for item in target_region)
        grid = tuple(
            math.ceil(size / chunk)
            for size, chunk in zip(source_shape, copy_chunks)
        )
        total = math.prod(grid)
        completed = 0

        for chunk_index in product(*(range(count) for count in grid)):
            source_slices = tuple(
                slice(index * chunk, min((index + 1) * chunk, size))
                for index, chunk, size in zip(
                    chunk_index, copy_chunks, source_shape
                )
            )
            target_slices = tuple(
                slice(
                    target_start + source_slice.start,
                    target_start + source_slice.stop,
                )
                for target_start, source_slice in zip(start4, source_slices)
            )

            block = np.asarray(source[source_slices])
            if not np.can_cast(block.dtype, self.dtype, casting=casting):
                raise TypeError(
                    f"Cannot cast block dtype {block.dtype} to target dtype {self.dtype} "
                    f"using casting={casting!r}."
                )
            self._level0[target_slices] = block.astype(self.dtype, copy=False)

            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, target_slices)  # type: ignore[arg-type]

        return target_region

    def write_block(
        self,
        data: Any,
        start: Sequence[int],
        *,
        casting: Literal[
            "no", "equiv", "safe", "same_kind", "unsafe"
        ] = "same_kind",
    ) -> Slice4D:
        """Write one in-memory CZYX block at a CZYX start coordinate."""
        block = np.asarray(data)
        if block.ndim != 4:
            raise ValueError(f"data must be 4D CZYX; actual ndim={block.ndim}.")
        return self.write(
            block,
            start=start,
            source_chunks=block.shape,
            casting=casting,
        )

    def write_region(
        self,
        source: Any,
        region: Sequence[Sequence[int]],
        *,
        source_chunks: Sequence[int] | None = None,
        casting: Literal[
            "no", "equiv", "safe", "same_kind", "unsafe"
        ] = "same_kind",
        progress_callback: Callable[[int, int, Slice4D], None] | None = None,
    ) -> Slice4D:
        """Write to ``region=[[c0,z0,y0,x0], [c1,z1,y1,x1]]``.

        The second coordinate is exclusive and the region extent must equal
        ``source.shape``.
        """
        if len(region) != 2:
            raise ValueError("region must contain start and stop CZYX coordinates.")
        start = self._validate_coordinate(region[0], "region start")
        stop = self._validate_coordinate(region[1], "region stop")
        if any(end <= begin for begin, end in zip(start, stop)):
            raise ValueError("region stop must be greater than start in every dimension.")
        expected_shape = tuple(end - begin for begin, end in zip(start, stop))
        actual_shape = tuple(int(v) for v in source.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"source.shape={actual_shape} does not match region extent={expected_shape}."
            )
        return self.write(
            source,
            start=start,
            source_chunks=source_chunks,
            casting=casting,
            progress_callback=progress_callback,
        )

    @property
    def level_paths(self) -> tuple[str, ...]:
        """Paths of the arrays listed for the selected OME-NGFF multiscale."""
        self._ensure_open()
        multiscales = self.root.attrs.get("multiscales", [])
        datasets = multiscales[self._multiscale_index].get("datasets", [])
        return tuple(
            self._normalize_array_path(dataset["path"])
            for dataset in datasets
            if isinstance(dataset, dict) and "path" in dataset
        )

    def get_array(self, level: int | str = 0) -> zarr.Array:
        """Return a Zarr array by pyramid index or internal array path.

        Parameters
        ----------
        level:
            Integer index into ``multiscales.datasets`` or an explicit Zarr
            array path such as ``"0"`` or ``"labels/neuron/0"``.
        """
        self._ensure_open()
        paths = self.level_paths

        if isinstance(level, str):
            path = self._normalize_array_path(level)
            if path not in paths:
                raise KeyError(
                    f"Array path {path!r} is not listed in the selected multiscale. "
                    f"Available paths: {paths}"
                )
        else:
            index = int(level)
            if index < 0:
                index += len(paths)
            if index < 0 or index >= len(paths):
                raise IndexError(
                    f"Pyramid level index {level} is out of range for {len(paths)} levels."
                )
            path = paths[index]

        try:
            array = self.root[path]
        except KeyError as exc:
            raise KeyError(f"The Zarr array does not exist: {path!r}") from exc
        if not isinstance(array, zarr.Array):
            raise TypeError(f"The selected path is not a Zarr array: {path!r}")
        return array

    @staticmethod
    def _normalize_read_key(key: Any, shape: Sequence[int]) -> tuple[Any, ...]:
        """Normalize NumPy-style basic indexing for a CZYX array.

        Supported index items are integers, slices, and one ellipsis. Missing
        dimensions are filled with ``slice(None)``. Negative integer indices
        are converted to their positive equivalents.
        """
        ndim = len(shape)
        items = list(key if isinstance(key, tuple) else (key,))

        ellipsis_positions = [
            index for index, item in enumerate(items) if item is Ellipsis
        ]
        if len(ellipsis_positions) > 1:
            raise IndexError("An index may contain at most one ellipsis (...).")
        if any(item is None for item in items):
            raise IndexError(
                "np.newaxis/None is not supported directly. Read the data first "
                "and then call np.expand_dims()."
            )

        if ellipsis_positions:
            ellipsis_index = ellipsis_positions[0]
            missing = ndim - (len(items) - 1)
            if missing < 0:
                raise IndexError(
                    f"Too many indices for a {ndim}D CZYX array: {key!r}"
                )
            items[ellipsis_index:ellipsis_index + 1] = [slice(None)] * missing
        elif len(items) < ndim:
            items.extend([slice(None)] * (ndim - len(items)))

        if len(items) != ndim:
            raise IndexError(f"Expected at most {ndim} CZYX indices; received {key!r}.")

        normalized: list[Any] = []
        for axis, (item, axis_size) in enumerate(zip(items, shape)):
            if isinstance(item, (int, np.integer)):
                index = int(item)
                if index < 0:
                    index += int(axis_size)
                if index < 0 or index >= int(axis_size):
                    raise IndexError(
                        f"Index {item} is out of bounds for axis {axis} with size {axis_size}."
                    )
                normalized.append(index)
            elif isinstance(item, slice):
                if item.step == 0:
                    raise ValueError("A slice step cannot be zero.")
                normalized.append(item)
            else:
                raise TypeError(
                    "Only integers, slices, and ellipsis are supported for direct reading; "
                    f"received {type(item).__name__} on axis {axis}."
                )

        return tuple(normalized)

    def read(self, key: Any = Ellipsis, *, level: int | str = 0) -> Any:
        """Read image data using NumPy-style indexing.

        Examples
        --------
        ``reader[0]`` returns channel 0 as ZYX.
        ``reader[:, 10:20, 100:200, 200:300]`` returns a CZYX block.
        ``reader.read((slice(None), 0), level=1)`` reads from pyramid level 1.
        """
        array = self.get_array(level)
        normalized_key = self._normalize_read_key(key, array.shape)
        return array[normalized_key]

    def read_region(
        self,
        start: Sequence[int],
        stop: Sequence[int],
        *,
        level: int | str = 0,
    ) -> np.ndarray:
        """Read a CZYX region using exclusive start/stop coordinates."""
        array = self.get_array(level)
        start4 = self._validate_coordinate(start, "read start")
        stop4 = self._validate_coordinate(stop, "read stop")

        if any(end <= begin for begin, end in zip(start4, stop4)):
            raise ValueError("read stop must be greater than start in every dimension.")
        if any(end > limit for end, limit in zip(stop4, array.shape)):
            raise IndexError(
                "The read region exceeds the selected array bounds: "
                f"start={start4}, stop={stop4}, shape={tuple(array.shape)}"
            )

        key = tuple(slice(begin, end) for begin, end in zip(start4, stop4))
        return np.asarray(array[key])

    def __getitem__(self, key: Any) -> Any:
        """Read from the level-0 CZYX array using NumPy-style indexing."""
        return self.read(key, level=0)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write to the level-0 CZYX array using NumPy-style indexing."""
        self._prepare_level0_write()
        normalized_key = self._normalize_read_key(key, self._level0.shape)
        self._level0[normalized_key] = value

    # ------------------------------------------------------------------
    # Optional pyramid generation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_pyramid_factors(
        factors: Iterable[Sequence[int]],
    ) -> list[tuple[int, int, int]]:
        result: list[tuple[int, int, int]] = []
        previous = (1, 1, 1)
        for item in factors:
            factor = tuple(int(v) for v in item)
            if len(factor) != 3 or any(v < 1 for v in factor):
                raise ValueError("Each factor must contain three integers greater than or equal to 1 in ZYX order.")
            if factor == (1, 1, 1):
                continue
            if any(current < old for current, old in zip(factor, previous)):
                raise ValueError("factors must be ordered monotonically non-decreasing.")
            if any(current % old != 0 for current, old in zip(factor, previous)):
                raise ValueError("Each cumulative factor must be evenly divisible by the preceding factor.")
            result.append(factor)  # type: ignore[arg-type]
            previous = factor  # type: ignore[assignment]
        return result

    @staticmethod
    def _normalize_downsample_modes(
        mode: str | Sequence[str], c_size: int
    ) -> tuple[str, ...]:
        allowed = {"mean", "nearest", "max"}
        if isinstance(mode, str):
            modes = (mode.lower(),) * c_size
        else:
            modes = tuple(str(value).lower() for value in mode)
            if len(modes) != c_size:
                raise ValueError(f"The number of per-channel modes must equal C={c_size}.")
        invalid = sorted(set(modes) - allowed)
        if invalid:
            raise ValueError(f"Invalid downsample mode: {invalid}")
        return modes

    def _generated_level_path(self, level: int) -> str:
        level0 = PurePosixPath(self.array_path)
        parent = level0.parent
        name = str(level)
        return name if str(parent) == "." else str(parent / name)

    def _remove_listed_lower_levels(self) -> None:
        multiscales = list(self.root.attrs.get("multiscales", []))
        multiscale = dict(multiscales[self._multiscale_index])
        datasets = list(multiscale.get("datasets", []))
        for dataset in datasets:
            path = str(dataset.get("path", "")).strip("/")
            if path and path != self.array_path and path in self.root:
                del self.root[path]
        multiscale["datasets"] = [self._level0_dataset_metadata()]
        multiscale.pop("type", None)
        multiscale.pop("metadata", None)
        multiscales[self._multiscale_index] = multiscale
        self.root.attrs["multiscales"] = multiscales
        self._metadata_dirty = True

    def build_pyramid(
        self,
        factors: Iterable[Sequence[int]] = (
            (1, 2, 2),
            (1, 4, 4),
            (2, 8, 8),
        ),
        *,
        mode: str | Sequence[str] = "mean",
    ) -> None:
        """Rebuild lower-resolution levels from the current level-0 array.

        ``factors`` are cumulative ZYX factors relative to level 0. ``mode``
        may be one string for all channels or one string per channel. Supported
        values are ``mean``, ``nearest``, and ``max``.
        """
        self._ensure_writable()
        try:
            import dask.array as da
        except ImportError as exc:
            raise ImportError(
                "build_pyramid() requires dask[array]."
            ) from exc

        factors_list = self._validate_pyramid_factors(factors)
        modes = self._normalize_downsample_modes(mode, self.shape[0])
        self._remove_listed_lower_levels()

        source = da.from_zarr(str(self.path), component=self.array_path)
        previous_factor = (1, 1, 1)
        datasets = [self._level0_dataset_metadata()]
        vz, vy, vx = self.voxel_size

        for level, factor in enumerate(factors_list, start=1):
            rz, ry, rx = tuple(
                current // previous
                for current, previous in zip(factor, previous_factor)
            )
            channel_results = []

            for channel_index, channel_mode in enumerate(modes):
                channel = source[channel_index : channel_index + 1]
                if channel_mode == "nearest":
                    down = channel[:, ::rz, ::ry, ::rx]
                else:
                    spatial_shape = channel.shape[1:]
                    pads = tuple(
                        (-int(size)) % ratio
                        for size, ratio in zip(spatial_shape, (rz, ry, rx))
                    )
                    padded = da.pad(
                        channel,
                        ((0, 0), (0, pads[0]), (0, pads[1]), (0, pads[2])),
                        mode="edge",
                    )
                    reduction = np.mean if channel_mode == "mean" else np.max
                    down = da.coarsen(
                        reduction,
                        padded,
                        axes={1: rz, 2: ry, 3: rx},
                        trim_excess=True,
                    )
                    if channel_mode == "mean" and np.issubdtype(
                        self.dtype, np.integer
                    ):
                        info = np.iinfo(self.dtype)
                        down = da.clip(da.rint(down), info.min, info.max)
                    down = down.astype(self.dtype)
                channel_results.append(down)

            level_data = da.concatenate(channel_results, axis=0)
            level_shape = tuple(int(v) for v in level_data.shape)
            level_chunks = tuple(
                min(chunk, size) for chunk, size in zip(self.chunks, level_shape)
            )
            level_path = self._generated_level_path(level)
            target = self._create_array(
                path=level_path,
                shape=level_shape,
                chunks=level_chunks,
            )
            da.store(
                level_data.rechunk(level_chunks),
                target,
                lock=False,
                compute=True,
            )
            datasets.append(
                {
                    "path": level_path,
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [
                                1.0,
                                vz * factor[0],
                                vy * factor[1],
                                vx * factor[2],
                            ],
                        }
                    ],
                }
            )
            source = da.from_zarr(str(self.path), component=level_path)
            previous_factor = factor

        multiscales = list(self.root.attrs.get("multiscales", []))
        multiscale = dict(multiscales[self._multiscale_index])
        multiscale["datasets"] = datasets
        if len(set(modes)) == 1:
            multiscale["type"] = modes[0]
            multiscale["metadata"] = {"method": f"channel-wise {modes[0]}"}
        else:
            multiscale.pop("type", None)
            multiscale["metadata"] = {
                "channelDownsampling": [
                    {
                        "channel": index,
                        "label": self.channel_names[index],
                        "mode": channel_mode,
                    }
                    for index, channel_mode in enumerate(modes)
                ]
            }
        multiscales[self._multiscale_index] = multiscale
        self.root.attrs["multiscales"] = multiscales
        self._metadata_dirty = True
        self._write_prepared = False

    # ------------------------------------------------------------------
    # Metadata maintenance and lifecycle
    # ------------------------------------------------------------------
    def set_channel_metadata(
        self,
        *,
        channel_names: Sequence[str] | None = None,
        channel_colors: Sequence[str] | None = None,
        channel_windows: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Update OMERO display metadata without altering image values."""
        self._ensure_writable()
        c_size = self.shape[0]
        if channel_names is not None:
            self.channel_names = self._normalize_channel_names(
                channel_names, c_size
            )
        if channel_colors is not None:
            self.channel_colors = self._normalize_channel_colors(
                channel_colors, c_size
            )
        if channel_windows is not None:
            self.channel_windows = self._normalize_channel_windows(
                channel_windows, c_size, self.dtype
            )
        self._write_omero_metadata()
        self._metadata_dirty = True

    def consolidate_metadata(self) -> None:
        """Generate/update Zarr v2 consolidated metadata (.zmetadata)."""
        self._ensure_writable()
        zarr.consolidate_metadata(str(self.path), zarr_format=2)
        self._metadata_dirty = False

    def close(self, *, consolidate: bool | None = None) -> None:
        """Finalize metadata. Safe to call more than once."""
        if self._closed:
            return
        do_consolidate = (
            self.consolidate_on_close if consolidate is None else bool(consolidate)
        )
        if do_consolidate and not self._read_only:
            self.consolidate_metadata()
        self._closed = True

    def __enter__(self) -> "ZarrIOManager":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close(consolidate=exc_type is None and self.consolidate_on_close)
        return False


PathType = Literal["ims", "zarr", "dir", "file"]
def detect_path_type(path: str | Path) -> PathType:
    """
    Determine whether a path is an IMS file, a Zarr store,
    an ordinary directory, or another ordinary file.

    Parameters
    ----------
    path:
        File or directory path.

    Returns
    -------
    Literal["ims", "zarr", "dir", "file"]
        "ims":
            An Imaris IMS file.
        "zarr":
            A valid Zarr v2 or Zarr v3 store.
        "directory":
            An ordinary directory that is not a Zarr store.
        "file":
            An ordinary file that is not a recognized IMS file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # IMS is a single HDF5 file.
    if path.is_file():
        if h5py.is_hdf5(path):
            try:
                with h5py.File(path, mode="r") as file:
                    if _has_ims_structure(file):
                        return "ims"
            except OSError:
                pass

        return "file"

    # Zarr is normally a directory-based store.
    if path.is_dir():
        if _is_zarr_store(path):
            return "zarr"

        return "dir"

    # Covers unusual path types such as special device files.
    return "file"


def _has_ims_structure(file: h5py.File) -> bool:
    """
    Check whether an opened HDF5 file has an Imaris IMS data structure.
    """
    dataset_group = file.get("DataSet")

    if not isinstance(dataset_group, h5py.Group):
        return False

    for resolution_name, resolution_group in dataset_group.items():
        if not resolution_name.startswith("ResolutionLevel "):
            continue
        if not isinstance(resolution_group, h5py.Group):
            continue

        for timepoint_name, timepoint_group in resolution_group.items():
            if not timepoint_name.startswith("TimePoint "):
                continue
            if not isinstance(timepoint_group, h5py.Group):
                continue

            for channel_name, channel_group in timepoint_group.items():
                if not channel_name.startswith("Channel "):
                    continue
                if not isinstance(channel_group, h5py.Group):
                    continue

                data = channel_group.get("Data")

                if isinstance(data, h5py.Dataset) and data.ndim == 3:
                    return True

    return False


def _is_zarr_store(path: Path) -> bool:
    """
    Check whether a directory is a valid Zarr v2 or Zarr v3 store.
    """
    # Zarr v2 and v3 root markers.
    has_zarr_marker = any(
        (path / marker).is_file()
        for marker in (
            ".zgroup",
            ".zarray",
            ".zmetadata",
            "zarr.json",
        )
    )

    if not has_zarr_marker:
        return False

    try:
        zarr.open(store=str(path), mode="r")
        return True
    except Exception:
        return False