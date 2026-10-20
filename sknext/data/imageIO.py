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
    """Return sorted absolute paths to immediate .tif/.tiff files in file_path.

    The path must be an existing directory; the search is not recursive and
    extension matching is case-insensitive.
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Folder does not exist: {file_path}"
    assert file_path.is_dir(), f"Input path is not a folder: {file_path}"
    tif_paths = [p.resolve() for p in file_path.iterdir() if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]]
    tif_paths = sorted(tif_paths)
    return tif_paths

def get_tif_path_dict_in_folder(file_path: Path | str, channel_names: list[str]) -> dict[str, list[Path]]:
    """Discover instance/semantic TIFF labels without requiring instance masks.

    Supported layouts::

        sample_0001.tif       # instance, only when F/P/C/A is requested
        sample_0001.ER.tif    # channel S.ER
        sample_0001.mito.tif  # channel S.mito

    Return ordered groups with corresponding sample basenames. Unrequested TIFF
    groups are ignored. Raises on incomplete or duplicate semantic samples.
    """
    file_path = Path(file_path)
    if not file_path.is_dir():
        raise NotADirectoryError(f"Label directory does not exist: {file_path}")
    requested = list(channel_names)
    if not requested:
        raise ValueError("At least one channel must be configured")
    semantic = [name for name in requested if name.startswith("S.") and len(name) > 2]
    if len(semantic) != len(set(semantic)):
        raise ValueError("Duplicate semantic channels are not allowed")
    invalid = [name for name in requested if name not in {"F", "P", "C", "A"} and name not in semantic]
    if invalid:
        raise ValueError(f"Unsupported channel names: {invalid}")
    instance_requested = any(name in {"F", "P", "C", "A"} for name in requested)
    groups: dict[str, dict[str, Path]] = {ch: {} for ch in semantic}
    if instance_requested:
        groups["instance"] = {}
    files = sorted((p.resolve() for p in file_path.iterdir()
                    if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}))
    semantic_sample_names = {
        path.stem[:-(len(ch) - 1)]
        for path in files for ch in semantic
        if path.stem.endswith('.' + ch[2:])
    }
    for path in files:
        stem = path.stem
        target = None
        sample = stem
        for ch in semantic:
            suffix = "." + ch[2:]
            if stem.endswith(suffix):
                target = ch
                sample = stem[:-len(suffix)]
                break
        if target is None:
            if not instance_requested:
                continue  # existing instance masks should not be required or used
            # For mixed targets, a valid instance file must match a semantic
            # sample name; ignore unrelated TIFFs. For instance-only tasks,
            # preserve compatibility with arbitrary (even dotted) basenames.
            if semantic and stem not in semantic_sample_names:
                continue
            target = "instance"
        if not sample:
            raise ValueError(f"Empty sample name in {path}")
        if sample in groups[target]:
            raise ValueError(f"Duplicate label for {target!r}, sample {sample!r}")
        groups[target][sample] = path
    for ch, entries in groups.items():
        if not entries:
            raise ValueError(f"No label TIFF files found for {ch!r} in {file_path}")
    names_by_group = {ch: set(entries) for ch, entries in groups.items()}
    reference_name = next(iter(names_by_group))
    reference = names_by_group[reference_name]
    for ch, names in names_by_group.items():
        if names != reference:
            missing = sorted(reference - names)
            extra = sorted(names - reference)
            raise ValueError(f"Mismatched label samples for {ch}: missing={missing}, extra={extra}")
    return {ch: [entries[name] for name in sorted(reference)] for ch, entries in groups.items()}



def pair_raw_and_label_tiffs(raw_paths, gt_groups):
    """Return raw and GT paths matched by sample identity, not list position.

    Accepted GT names: ``sample.tif``, ``sample_gt.tif``, or
    ``prefix_gt_0001.tif`` for raw ``prefix_0001.tif``. The semantic
    suffix ``.ER`` in ``prefix_gt_0001.ER.tif`` is handled independently.
    Ambiguous matches, duplicate samples, and missing labels are fatal.
    """
    raw_paths = [Path(p) for p in raw_paths]
    raw_map = {}
    for path in raw_paths:
        if path.stem in raw_map:
            raise ValueError(f"Duplicate raw sample {path.stem!r}")
        raw_map[path.stem] = path
    if not raw_map:
        raise ValueError("No raw TIFF files were supplied")
    if not gt_groups:
        raise ValueError("No label groups were supplied")
    result = {}
    for channel, paths in gt_groups.items():
        indexed = {}
        suffix = '.' + channel[2:] if channel.startswith('S.') else ''
        for item in paths:
            path = Path(item)
            stem = path.stem
            if suffix:
                if not stem.endswith(suffix):
                    raise ValueError(f"Label {path.name!r} is not a {channel} TIFF")
                stem = stem[:-len(suffix)]
            candidates = [stem]
            # Legacy label convention: xxx_gt_0000 / xxx_0000_gt.
            candidates.extend([re.sub(r'_gt(?=_|$)', '', stem, count=1),
                               re.sub(r'(?<=_)gt_', '', stem, count=1)])
            matches = set(candidates) & set(raw_map)
            if len(matches) != 1:
                raise ValueError(f"Could not uniquely pair label {path.name!r} "
                                 f"with a raw TIFF: possible matches={sorted(matches)}")
            sample = matches.pop()
            if sample in indexed:
                raise ValueError(f"Duplicate {channel} label for {sample!r}")
            indexed[sample] = path
        missing = set(raw_map) - set(indexed)
        extra = set(indexed) - set(raw_map)
        if missing or extra:
            raise ValueError(f"Unmatched {channel} labels: missing={sorted(missing)} "
                             f"extra={sorted(extra)}")
        result[channel] = [indexed[stem] for stem in sorted(raw_map)]
    return [raw_map[stem] for stem in sorted(raw_map)], result


def read_one_3D_tif(tif_path: str|Path, output_axes:Literal["ZYX", "CZYX", "ZYXC"] = "CZYX") -> np.ndarray:
    """Read a TIFF volume and arrange its axes as requested.

    Args:
        tif_path: Path to a three-dimensional ZYX or four-dimensional CZYX TIFF.
        output_axes: ZYX, CZYX, or ZYXC; ZYX requires a three-dimensional input.

    Returns:
        Image array with a singleton channel axis added when needed.
    """
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


def calculate_patch_coordinates(
        img_shape: tuple[int, int, int] | list[int],
        patch_size: tuple[int, int, int] | list[int],
        overlap: tuple[int, int, int] | list[int] = (0, 0, 0),
        padding: tuple[int, int, int] | list[int] = (0, 0, 0),
):
    """Collect spatial patch bounds for an image into an integer array.

    Args:
        img_shape: ZYX shape, or CZYX shape whose channel dimension is ignored.
        patch_size: Spatial patch dimensions in ZYX order.
        overlap: Overlap in voxels along Z, Y, and X.
        padding: Spatial halo widths passed to patch_coordinates_iter.

    Returns:
        An (N, 3, 2) array of half-open bounds, including an empty array for N=0.
    """
    if len(img_shape) == 4:
        img_shape = img_shape[1:]

    coordinates = list(
        patch_coordinates_iter(
            img_shape=img_shape,
            patch_size=patch_size,
            overlap=overlap,
            padding=padding,
        )
    )

    if not coordinates:
        return np.empty((0, 3, 2), dtype=np.int64)

    return np.stack(coordinates, axis=0)


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

def central_block_coordinates_iter(
    img_shape: tuple[int, int, int],
    central_block_size: tuple[int, int, int],
):
    """Yield nonoverlapping block bounds covering a ZYX image.

    img_shape and central_block_size contain three spatial sizes. Each yielded
    int64 array has shape (3, 2) with inclusive starts and exclusive stops;
    blocks touching the high image edges are clipped to the image shape.
    """
    z_size, y_size, x_size = map(int, img_shape)
    bz, by, bx = map(int, central_block_size)

    for z0 in range(0, z_size, bz):
        z1 = min(z0 + bz, z_size)

        for y0 in range(0, y_size, by):
            y1 = min(y0 + by, y_size)

            for x0 in range(0, x_size, bx):
                x1 = min(x0 + bx, x_size)

                yield np.asarray(
                    [
                        [z0, z1],
                        [y0, y1],
                        [x0, x1],
                    ],
                    dtype=np.int64,
                )

def patch_coordinates_generator(img:np.ndarray, # CZYX / ZYX
                                patch_coordinates:np.ndarray)->Iterator[np.ndarray, np.ndarray]:
    """Yield (patch_view, bounds) pairs from ZYX or CZYX img.

    patch_coordinates is an (N, 3, 2) array of half-open ZYX bounds. Channel
    axes are preserved and slices may share memory with the input image.
    """
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
