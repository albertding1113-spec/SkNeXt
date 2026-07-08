import numpy as np
from scipy import ndimage as ndi
from typing import Literal, Sequence


def post_processing_instance(label: np.ndarray, process_dict: dict):
    """
    Post-process instance segmentation labels.

    Supported input shapes:
    - 3D: ZYX
    - 4D: CZYX

    Returns
    -------
    np.ndarray
        Same shape and dtype as input.
    """
    label = np.asarray(label)

    if label.ndim not in (3, 4):
        raise ValueError(
            f"`label` must be 3D ZYX or 4D CZYX, but got shape {label.shape}."
        )

    output = label.copy()

    fill_holes = bool(process_dict.get("fill_holes", False))

    remove_small = process_dict.get("remove_small", -1)
    remove_small = -1 if remove_small is None else int(remove_small)

    remove_large = process_dict.get("remove_large", -1)
    remove_large = -1 if remove_large is None else int(remove_large)

    def _process_one_channel(one_label: np.ndarray) -> np.ndarray:
        """Process one ZYX instance-label volume."""
        processed = one_label.copy()

        if fill_holes:
            processed = fill_instance_holes(processed)

        if remove_small > 0:
            processed = remove_small_instances(processed, remove_small)

        if remove_large > 0:
            processed = remove_large_instances(processed, remove_large)

        return processed

    # 3D: ZYX
    if label.ndim == 3:
        output = _process_one_channel(output)

    # 4D: CZYX
    else:
        channel_num = output.shape[0]
        for c in range(channel_num):
            output[c, :, :, :] = _process_one_channel(output[c, :, :, :])

    return output

def post_processing_semantic(label: np.ndarray, process_dict: dict):
    """
    Post-process semantic segmentation labels.

    Supported input shapes:
    - 3D: ZYX
    - 4D: CZYX

    Notes
    -----
    For 3D input, the function treats it as one semantic label volume.
    For 4D input, each channel is processed independently.
    """
    label = np.asarray(label)

    if label.ndim not in (3, 4):
        raise ValueError(
            f"`label` must be 3D ZYX or 4D CZYX, but got shape {label.shape}."
        )

    output = label.copy()

    fill_holes = bool(process_dict.get("fill_holes", False))

    remove_small = process_dict.get("remove_small", -1)
    remove_small = -1 if remove_small is None else int(remove_small)

    remove_large = process_dict.get("remove_large", -1)
    remove_large = -1 if remove_large is None else int(remove_large)

    def _process_one_channel(one_label: np.ndarray) -> np.ndarray:
        """Process one ZYX semantic-label volume."""
        processed = one_label.copy()

        if fill_holes:
            processed = fill_semantic_holes(processed)

        if remove_small > 0:
            processed = remove_small_semantics(processed, remove_small)

        if remove_large > 0:
            processed = remove_large_semantics(processed, remove_large)

        return processed

    # 3D: ZYX
    if label.ndim == 3:
        output = _process_one_channel(output)

    # 4D: CZYX
    else:
        channel_num = output.shape[0]
        for c in range(channel_num):
            output[c, :, :, :] = _process_one_channel(output[c, :, :, :])

    return output

def fill_instance_holes(
    label: np.ndarray,
    background: int = 0,
    mode: str = "3d",
) -> np.ndarray:
    """
    Fill holes inside each instance of an instance-segmentation label image.

    Parameters
    ----------
    label : np.ndarray
        Instance label image.

        Supported shapes:
        - 2D: (Y, X)
        - 3D: (Z, Y, X)

        Different instances must have different positive integer IDs.
        Background is normally 0.

    background : int, default=0
        Background label value.

    mode : {"3d", "2d"}, default="3d"
        Filling mode for 3D data:

        - "3d":
          Fill enclosed cavities in the full 3D instance volume.
        - "2d":
          Fill holes independently on every Z slice.

        For 2D input, this parameter has no effect.

    Returns
    -------
    np.ndarray
        Label image after hole filling. The dtype is the same as the input.
    """
    label = np.asarray(label)
    if label.ndim not in (2, 3):
        raise ValueError(f"`label` must be a 2D or 3D array, but got shape {label.shape}.")
    if mode not in ("2d", "3d"):
        raise ValueError(f"`mode` must be '2d' or '3d', but got {mode!r}.")
    output = label.copy()
    instance_ids = np.unique(label)
    instance_ids = instance_ids[instance_ids != background]
    for instance_id in instance_ids:
        # Obtain the bounding box of the current instance.
        coordinates = np.where(label == instance_id)
        if coordinates[0].size == 0:
            continue
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1)for axis in coordinates)
        local_label = label[slices]
        instance_mask = local_label == instance_id
        if label.ndim == 2 or mode == "3d":
            filled_mask = ndi.binary_fill_holes(instance_mask)
        else:
            # Fill holes independently in each Z slice.
            filled_mask = np.empty_like(instance_mask, dtype=bool)
            for z_index in range(instance_mask.shape[0]):
                filled_mask[z_index] = ndi.binary_fill_holes(instance_mask[z_index])
        # Newly filled area.
        hole_mask = filled_mask & ~instance_mask
        # Only fill background pixels/voxels. Do not overwrite other instances.
        local_output = output[slices]
        writable_mask = hole_mask & (local_output == background)
        local_output[writable_mask] = instance_id
    return output


def remove_small_instances(
    label: np.ndarray,
    thresh: int,
    background: int = 0,
    relabel: bool = False,
) -> np.ndarray:
    """
    Remove instances whose size is smaller than a specified threshold.

    Parameters
    ----------
    label : np.ndarray
        2D or 3D integer instance label image.
        Each instance must have a unique integer ID.

    thresh : int
        Minimum number of pixels or voxels to retain an instance.
        Instances with size < thresh are removed.

    background : int, default=0
        Background label value.

    relabel : bool, default=False
        Whether to relabel the remaining instances consecutively as
        1, 2, ..., N.

    Returns
    -------
    np.ndarray
        Processed instance label image.
    """
    label = np.asarray(label)
    if not np.issubdtype(label.dtype, np.integer):
        raise TypeError(f"`label` must have an integer dtype, but got {label.dtype}.")
    if thresh < 0:
        raise ValueError(f"`thresh` must be non-negative, but got {thresh}.")
    output = label.copy()
    instance_ids, instance_sizes = np.unique(label, return_counts=True,)

    # Exclude background from instance statistics.
    foreground_mask = instance_ids != background
    instance_ids = instance_ids[foreground_mask]
    instance_sizes = instance_sizes[foreground_mask]

    # Find IDs whose sizes are smaller than the threshold.
    small_ids = instance_ids[instance_sizes < thresh]
    if small_ids.size > 0:
        output[np.isin(output, small_ids)] = background
    if relabel:
        remaining_ids = np.unique(output)
        remaining_ids = remaining_ids[remaining_ids != background]
        relabeled = np.full(
            output.shape,
            fill_value=background,
            dtype=output.dtype,
        )
        for new_id, old_id in enumerate(remaining_ids, start=1):
            relabeled[output == old_id] = new_id
        output = relabeled
    return output

import numpy as np


def remove_large_instances(
    label: np.ndarray,
    thresh: int,
    background: int = 0,
    relabel: bool = False,
) -> np.ndarray:
    """
    Remove instances whose number of pixels/voxels is larger than `thresh`.

    Parameters
    ----------
    label : np.ndarray
        Integer instance label image. Supports arrays of any dimension,
        such as (Y, X) or (Z, Y, X).

    thresh : int
        Maximum instance size to retain.
        Instances satisfying size > thresh are removed.

    background : int, default=0
        Background label value.

    relabel : bool, default=False
        If True, relabel the retained instances consecutively as
        1, 2, ..., N.

    Returns
    -------
    np.ndarray
        Filtered instance label image with the same dtype as the input.
    """
    label = np.asarray(label)
    if not np.issubdtype(label.dtype, np.integer):
        raise TypeError(f"`label` must have an integer dtype, but got {label.dtype}.")

    if not isinstance(thresh, (int, np.integer)):
        raise TypeError(f"`thresh` must be an integer, but got {type(thresh).__name__}.")

    if thresh < 0:
        raise ValueError(f"`thresh` must be non-negative, but got {thresh}.")

    output = label.copy()

    # Count the number of pixels/voxels belonging to every label ID.
    instance_ids, instance_sizes = np.unique(
        label,
        return_counts=True,
    )

    # Exclude the background label.
    foreground_mask = instance_ids != background
    instance_ids = instance_ids[foreground_mask]
    instance_sizes = instance_sizes[foreground_mask]

    # Find instances larger than the threshold.
    large_instance_ids = instance_ids[instance_sizes > thresh]

    # Set all large instances to the background value.
    if large_instance_ids.size > 0:
        output[np.isin(output, large_instance_ids)] = background

    if relabel:
        remaining_ids = np.unique(output)
        remaining_ids = remaining_ids[remaining_ids != background]

        relabeled = np.full(
            output.shape,
            fill_value=background,
            dtype=output.dtype,
        )

        for new_id, old_id in enumerate(remaining_ids, start=1):
            relabeled[output == old_id] = new_id

        output = relabeled

    return output

def fill_semantic_holes(
    label: np.ndarray,
    background: int = 0,
    mode: Literal["2d", "3d"] = "3d",
    class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Fill holes in a semantic-segmentation label image.

    The function processes every semantic class independently. Newly filled
    pixels or voxels are written only into background regions, so existing
    semantic classes will not be overwritten.

    Parameters
    ----------
    label : np.ndarray
        Semantic label image.

        Supported shapes:
        - 2D: (Y, X)
        - 3D: (Z, Y, X)

        Examples:
        - Binary mask: background=0, foreground=1
        - Multiclass mask: background=0, classes=1, 2, 3, ...

    background : int, default=0
        Background label value.

    mode : {"2d", "3d"}, default="3d"
        Hole-filling mode for a 3D image:

        - "3d":
          Fill cavities enclosed in the complete 3D volume.
        - "2d":
          Fill holes independently on each Z slice.

        For a 2D input image, this option has no effect.

    class_ids : sequence of int or None, default=None
        Semantic class IDs to process.

        When None, all label values except `background` are processed.

    Returns
    -------
    np.ndarray
        Hole-filled semantic label image with the same shape and dtype as
        the input.

    Notes
    -----
    This function expects a thresholded semantic label image, not a raw
    probability map or network logits.
    """
    label = np.asarray(label)
    if label.ndim not in (2, 3):
        raise ValueError( f"`label` must be a 2D or 3D array, but got shape {label.shape}.")
    if mode not in ("2d", "3d"):
        raise ValueError(f"`mode` must be either '2d' or '3d', but got {mode!r}.")
    if not (np.issubdtype(label.dtype, np.integer)or np.issubdtype(label.dtype, np.bool_)):
        raise TypeError(f"`label` must be an integer or boolean semantic label image, but got dtype {label.dtype}.")
    output = label.copy()
    if class_ids is None:
        semantic_ids = np.unique(label)
        semantic_ids = semantic_ids[semantic_ids != background]
    else:
        semantic_ids = np.asarray(class_ids)
        if semantic_ids.ndim != 1:
            raise ValueError("`class_ids` must be a one-dimensional sequence.")
        semantic_ids = semantic_ids[semantic_ids != background]
    for semantic_id in semantic_ids:
        coordinates = np.where(label == semantic_id)
        if coordinates[0].size == 0:
            continue
        # Crop to the bounding box of the current semantic class to reduce
        # temporary memory usage.
        local_slices = tuple(
            slice(
                int(axis_coordinates.min()),
                int(axis_coordinates.max()) + 1,
            )
            for axis_coordinates in coordinates
        )

        local_label = label[local_slices]
        semantic_mask = local_label == semantic_id

        if label.ndim == 2 or mode == "3d":
            filled_mask = ndi.binary_fill_holes(semantic_mask)

        else:
            # Fill every XY slice independently.
            filled_mask = np.empty_like(semantic_mask, dtype=bool)

            for z_index in range(semantic_mask.shape[0]):
                filled_mask[z_index] = ndi.binary_fill_holes(
                    semantic_mask[z_index]
                )
        # Regions newly generated by hole filling.
        hole_mask = filled_mask & ~semantic_mask
        # Do not overwrite other semantic classes.
        local_output = output[local_slices]
        writable_mask = hole_mask & (local_output == background)
        local_output[writable_mask] = semantic_id
    return output


def remove_small_semantics(
    label: np.ndarray,
    thresh: int,
    background: int = 0,
    connectivity: int = 1,
    mode: Literal["2d", "3d"] = "3d",
    class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Remove small disconnected components from a semantic-segmentation label.

    Each semantic class is processed independently. A connected component
    whose size is smaller than `thresh` is replaced with `background`.

    Parameters
    ----------
    label : np.ndarray
        Semantic label image.

        Supported shapes:
        - 2D: (Y, X)
        - 3D: (Z, Y, X)

        Examples:
        - Binary segmentation: 0=background, 1=foreground
        - Multiclass segmentation: 0=background, 1/2/3=semantic classes

    thresh : int
        Minimum component size to retain.

        Components satisfying:

            component_size < thresh

        are removed.

    background : int, default=0
        Background label value.

    connectivity : int, default=1
        Connectivity used for connected-component analysis.

        For 2D:
        - 1: 4-connectivity
        - 2: 8-connectivity

        For 3D:
        - 1: 6-connectivity
        - 2: 18-connectivity
        - 3: 26-connectivity

    mode : {"2d", "3d"}, default="3d"
        Processing mode for 3D input:

        - "3d":
          Detect connected components in the complete 3D volume.
        - "2d":
          Process every XY slice independently.

        For 2D input, this parameter has no effect.

    class_ids : sequence of int or None, default=None
        Semantic class IDs to process.

        If None, all label values except `background` are processed.

    Returns
    -------
    np.ndarray
        Processed semantic label image with the same shape and dtype as
        the input.
    """
    label = np.asarray(label)

    if label.ndim not in (2, 3):
        raise ValueError(
            f"`label` must be a 2D or 3D array, but got shape {label.shape}."
        )

    if not (
        np.issubdtype(label.dtype, np.integer)
        or np.issubdtype(label.dtype, np.bool_)
    ):
        raise TypeError(
            "`label` must have an integer or boolean dtype, "
            f"but got {label.dtype}."
        )

    if not isinstance(thresh, (int, np.integer)):
        raise TypeError(
            f"`thresh` must be an integer, but got {type(thresh).__name__}."
        )

    if thresh < 0:
        raise ValueError(
            f"`thresh` must be non-negative, but got {thresh}."
        )

    if mode not in ("2d", "3d"):
        raise ValueError(
            f"`mode` must be either '2d' or '3d', but got {mode!r}."
        )

    max_connectivity = label.ndim if mode == "3d" else 2

    if not isinstance(connectivity, (int, np.integer)):
        raise TypeError("`connectivity` must be an integer.")

    if not 1 <= connectivity <= max_connectivity:
        raise ValueError(
            f"`connectivity` must be in [1, {max_connectivity}], "
            f"but got {connectivity}."
        )

    output = label.copy()

    if thresh == 0:
        return output

    if class_ids is None:
        semantic_ids = np.unique(label)
        semantic_ids = semantic_ids[semantic_ids != background]
    else:
        semantic_ids = np.asarray(class_ids)

        if semantic_ids.ndim != 1:
            raise ValueError(
                "`class_ids` must be a one-dimensional sequence."
            )

        semantic_ids = np.unique(semantic_ids)
        semantic_ids = semantic_ids[semantic_ids != background]

    def remove_from_mask(
        semantic_mask: np.ndarray,
        structure: np.ndarray,
    ) -> np.ndarray:
        component_labels, component_num = ndi.label(
            semantic_mask,
            structure=structure,
        )

        if component_num == 0:
            return semantic_mask

        component_sizes = np.bincount(component_labels.ravel())

        # ID 0 represents background and must never be treated as a component.
        small_component_ids = np.flatnonzero(
            component_sizes < thresh
        )
        small_component_ids = small_component_ids[
            small_component_ids != 0
        ]

        if small_component_ids.size == 0:
            return semantic_mask

        small_mask = np.isin(
            component_labels,
            small_component_ids,
        )

        return semantic_mask & ~small_mask

    for semantic_id in semantic_ids:
        semantic_mask = label == semantic_id

        if not np.any(semantic_mask):
            continue

        if label.ndim == 2:
            structure = ndi.generate_binary_structure(
                rank=2,
                connectivity=connectivity,
            )

            retained_mask = remove_from_mask(
                semantic_mask,
                structure,
            )

            removed_mask = semantic_mask & ~retained_mask
            output[removed_mask] = background

        elif mode == "3d":
            structure = ndi.generate_binary_structure(
                rank=3,
                connectivity=connectivity,
            )

            retained_mask = remove_from_mask(
                semantic_mask,
                structure,
            )

            removed_mask = semantic_mask & ~retained_mask
            output[removed_mask] = background

        else:
            structure = ndi.generate_binary_structure(
                rank=2,
                connectivity=connectivity,
            )

            for z_index in range(label.shape[0]):
                slice_mask = semantic_mask[z_index]

                if not np.any(slice_mask):
                    continue

                retained_slice = remove_from_mask(
                    slice_mask,
                    structure,
                )

                removed_slice = slice_mask & ~retained_slice
                output[z_index][removed_slice] = background

    return output

def remove_large_semantics(
    label: np.ndarray,
    thresh: int,
    background: int = 0,
    connectivity: int = 1,
    mode: Literal["2d", "3d"] = "3d",
    class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Remove large disconnected components from a semantic-segmentation label.

    Each semantic class is processed independently. A connected component
    whose size is larger than `thresh` is replaced with `background`.

    Parameters
    ----------
    label : np.ndarray
        Semantic label image.

        Supported shapes:
        - 2D: (Y, X)
        - 3D: (Z, Y, X)

        Examples:
        - Binary segmentation: 0=background, 1=foreground
        - Multiclass segmentation: 0=background, 1/2/3=semantic classes

    thresh : int
        Maximum component size to retain.

        Components satisfying:

            component_size > thresh

        are removed. Components whose size is exactly equal to `thresh`
        are retained.

    background : int, default=0
        Background label value.

    connectivity : int, default=1
        Connectivity used for connected-component analysis.

        For 2D:
        - 1: 4-connectivity
        - 2: 8-connectivity

        For 3D:
        - 1: 6-connectivity
        - 2: 18-connectivity
        - 3: 26-connectivity

    mode : {"2d", "3d"}, default="3d"
        Processing mode for 3D input:

        - "3d":
          Detect connected components in the complete 3D volume.
        - "2d":
          Detect and remove components independently on every XY slice.

        For 2D input, this parameter has no effect.

    class_ids : sequence of int or None, default=None
        Semantic class IDs to process.

        If None, all values except `background` are processed.

    Returns
    -------
    np.ndarray
        Processed semantic label image with the same shape and dtype as
        the input.
    """
    label = np.asarray(label)

    if label.ndim not in (2, 3):
        raise ValueError(
            f"`label` must be a 2D or 3D array, but got shape {label.shape}."
        )

    if not (
        np.issubdtype(label.dtype, np.integer)
        or np.issubdtype(label.dtype, np.bool_)
    ):
        raise TypeError(
            "`label` must have an integer or boolean dtype, "
            f"but got {label.dtype}."
        )

    if not isinstance(thresh, (int, np.integer)):
        raise TypeError(
            f"`thresh` must be an integer, but got "
            f"{type(thresh).__name__}."
        )

    if thresh < 0:
        raise ValueError(
            f"`thresh` must be non-negative, but got {thresh}."
        )

    if mode not in ("2d", "3d"):
        raise ValueError(
            f"`mode` must be either '2d' or '3d', but got {mode!r}."
        )

    if not isinstance(connectivity, (int, np.integer)):
        raise TypeError(
            f"`connectivity` must be an integer, but got "
            f"{type(connectivity).__name__}."
        )

    processing_ndim = 2 if label.ndim == 2 or mode == "2d" else 3

    if not 1 <= int(connectivity) <= processing_ndim:
        raise ValueError(
            f"`connectivity` must be in [1, {processing_ndim}] "
            f"for the selected processing mode, but got {connectivity}."
        )

    output = label.copy()

    if class_ids is None:
        semantic_ids = np.unique(label)
        semantic_ids = semantic_ids[semantic_ids != background]
    else:
        semantic_ids = np.asarray(class_ids)

        if semantic_ids.ndim != 1:
            raise ValueError(
                "`class_ids` must be a one-dimensional sequence."
            )

        semantic_ids = np.unique(semantic_ids)
        semantic_ids = semantic_ids[semantic_ids != background]

    def find_large_component_mask(
        semantic_mask: np.ndarray,
        structure: np.ndarray,
    ) -> np.ndarray:
        """
        Return a boolean mask marking connected components larger than thresh.
        """
        component_labels, component_num = ndi.label(
            semantic_mask,
            structure=structure,
        )

        if component_num == 0:
            return np.zeros_like(semantic_mask, dtype=bool)

        component_sizes = np.bincount(component_labels.ravel())

        # Label 0 is connected-component background and must be excluded.
        large_component_ids = np.flatnonzero(
            component_sizes > thresh
        )
        large_component_ids = large_component_ids[
            large_component_ids != 0
        ]

        if large_component_ids.size == 0:
            return np.zeros_like(semantic_mask, dtype=bool)

        return np.isin(
            component_labels,
            large_component_ids,
        )

    for semantic_id in semantic_ids:
        semantic_mask = label == semantic_id

        if not np.any(semantic_mask):
            continue

        if label.ndim == 2:
            structure = ndi.generate_binary_structure(
                rank=2,
                connectivity=int(connectivity),
            )

            large_mask = find_large_component_mask(
                semantic_mask,
                structure,
            )

            output[large_mask] = background

        elif mode == "3d":
            structure = ndi.generate_binary_structure(
                rank=3,
                connectivity=int(connectivity),
            )

            large_mask = find_large_component_mask(
                semantic_mask,
                structure,
            )

            output[large_mask] = background

        else:
            structure = ndi.generate_binary_structure(
                rank=2,
                connectivity=int(connectivity),
            )

            for z_index in range(label.shape[0]):
                slice_mask = semantic_mask[z_index]

                if not np.any(slice_mask):
                    continue

                large_slice_mask = find_large_component_mask(
                    slice_mask,
                    structure,
                )

                output[z_index][large_slice_mask] = background

    return output