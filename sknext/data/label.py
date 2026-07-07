import numpy as np
import torch
import numpy as np
from typing import Literal
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries, watershed as skimage_watershed
from skimage.filters import threshold_otsu


def generate_channels_from_labels(labels_dict: dict[np.ndarray], # ZYX
                                  channels: list[str]|tuple[str],
                                  channel_extra_opts: dict = {}) -> list[np.ndarray]:
    for label in labels_dict.values(): assert label.ndim == 3, "label should be a 3d array"
    ch_img_list = []
    for channel in channels:
        if channel in ["F", "P", "C", "A"] or "S." in channel: pass
        else: raise ValueError("channel must be one of ['F', 'P', 'C', 'A'] or be a semantic channel.")
        if channel == "F":
            extra_opts = channel_extra_opts.get(channel, {})
            dilation = extra_opts.get("dilation", 0)
            erosion = extra_opts.get("erosion", 0)
            _ch_img = labels_dict["instance"].copy()
            _ch_img = (_ch_img > 0).astype(np.uint8)
            _ch_img = mask_dilation_erosion(_ch_img, dilation, erosion)
            ch_img_list.append(_ch_img)
        elif channel == "P":
            extra_opts = channel_extra_opts.get(channel, {"type": "skeleton"})
            type =  extra_opts.get("type", "skeleton")
            dilation = extra_opts.get("dilation", 0)
            erosion = extra_opts.get("erosion", 0)
            _ch_img = labels_dict["instance"].copy()
            if type == "skeleton":
                _ch_img = reconstruct_skeleton_from_mask(_ch_img)
            elif type == "centroid":
                _ch_img = reconstruct_centroid_from_mask(_ch_img)
            _ch_img = mask_dilation_erosion(_ch_img, dilation, erosion)
            ch_img_list.append(_ch_img)
        elif channel == "C":
            extra_opts = channel_extra_opts.get(channel, {"mode": "inner"})
            contour_mode = extra_opts.get("mode", "inner")
            _ch_img = labels_dict["instance"].copy()
            _ch_img = reconstruct_contour_from_mask(_ch_img, contour_mode)
            ch_img_list.append(_ch_img)
        elif channel == "A":
            extra_opts = channel_extra_opts.get(channel, {"z_affinities": [1], "y_affinities": [1], "x_affinities": [1]})
            z_affinities = extra_opts.get("z_affinities", [1])
            y_affinities = extra_opts.get("y_affinities", [1])
            x_affinities = extra_opts.get("x_affinities", [1])
            assert len(z_affinities) == len(y_affinities) == len(x_affinities), "ZYX affinities should have same length."
            _ch_img = labels_dict["instance"].copy()
            _ch_img = calc_affinities_from_mask(_ch_img, {"z_affinities": z_affinities,
                                                            "y_affinities": y_affinities,
                                                            "x_affinities": x_affinities})
            for i in range(_ch_img.shape[0]):
                ch_img_list.append(_ch_img[i, :, :, :])
        elif "S." in channel:
            extra_opts = channel_extra_opts.get(channel, {})
            dilation = extra_opts.get("dilation", 0)
            erosion = extra_opts.get("erosion", 0)
            _ch_img = labels_dict[channel].copy()
            _ch_img = (_ch_img > 0).astype(np.uint8)
            _ch_img = mask_dilation_erosion(_ch_img, dilation, erosion)
            ch_img_list.append(_ch_img)
        else: raise NotImplementedError(f"Unknown channel: f{channel}")
    return ch_img_list

def generate_channels_from_instance_labels(labels: np.ndarray, # ZYX
                                           channels: list[str]|tuple[str],
                                           channel_extra_opts: dict = {}) -> list[np.ndarray]:
    assert labels.ndim == 3, "labels should be a 3d array"
    ch_img_list = []
    for channel in channels:
        assert channel in ["F", "P", "C", "A"], "channel must be one of ['F', 'P', 'C', 'A']."
        if channel == "F":
            extra_opts = channel_extra_opts.get(channel, {})
            dilation = extra_opts.get("dilation", 0)
            erosion = extra_opts.get("erosion", 0)
            one_ch_img = (labels > 0).astype(np.uint8)
            one_ch_img = mask_dilation_erosion(one_ch_img, dilation, erosion)
            ch_img_list.append(one_ch_img)
        elif channel == "P":
            extra_opts = channel_extra_opts.get(channel, {"type": "skeleton"})
            type =  extra_opts.get("type", "skeleton")
            dilation = extra_opts.get("dilation", 0)
            erosion = extra_opts.get("erosion", 0)
            if type == "skeleton":
                one_ch_img = reconstruct_skeleton_from_mask(labels)
            elif type == "centroid":
                one_ch_img = reconstruct_centroid_from_mask(labels)
            one_ch_img = mask_dilation_erosion(one_ch_img, dilation, erosion)
            ch_img_list.append(one_ch_img)
        elif channel == "C":
            extra_opts = channel_extra_opts.get(channel, {"mode": "inner"})
            contour_mode = extra_opts.get("mode", "inner")
            one_ch_img = reconstruct_contour_from_mask(labels, contour_mode)
            ch_img_list.append(one_ch_img)
        elif channel == "A":
            assert len(channels) == 1, "Affinity channel should be used alone."
            extra_opts = channel_extra_opts.get(channel, {"z_affinities": [1], "y_affinities": [1], "x_affinities": [1]})
            z_affinities = extra_opts.get("z_affinities", [1])
            y_affinities = extra_opts.get("y_affinities", [1])
            x_affinities = extra_opts.get("x_affinities", [1])
            assert len(z_affinities) == len(y_affinities) == len(x_affinities), "ZYX affinities should have same length."
            one_ch_img = calc_affinities_from_mask(labels, {"z_affinities": z_affinities,
                                                            "y_affinities": y_affinities,
                                                            "x_affinities": x_affinities})
            for i in range(one_ch_img.shape[0]):
                ch_img_list.append(one_ch_img[i, :, :, :])
    return ch_img_list


def mask_dilation_erosion(img: np.ndarray,
                          dilation: int = 0,
                          erosion: int = 0) -> np.array:
    assert dilation >= 0, "dilation must be >= 0."
    assert erosion >= 0, "erosion must be >= 0."
    original_dtype = img.dtype
    mask = img > 0

    def _process_one_mask(one_mask: np.ndarray) -> np.ndarray:
        structure = generate_binary_structure(
            rank=one_mask.ndim,
            connectivity=1,
        )
        out = one_mask

        if dilation > 0:
            out = binary_dilation(
                out,
                structure=structure,
                iterations=dilation,
            )
        if erosion > 0:
            out = binary_erosion(
                out,
                structure=structure,
                iterations=erosion,
            )
        return out

    if mask.ndim in [2, 3]:
        result = _process_one_mask(mask)
    else:
        # CZYX
        result = np.zeros_like(mask, dtype=bool)
        for c in range(mask.shape[0]):
            result[c] = _process_one_mask(mask[c])
    return result.astype(original_dtype)


def reconstruct_skeleton_from_mask(labels: np.ndarray) -> np.ndarray:
    assert labels.ndim == 3, "labels should be a 3d array"
    skeleton_labels = np.zeros_like(labels)
    props = regionprops(labels)
    for prop in props:
        instance_id = prop.label
        z0, y0, x0, z1, y1, x1 = prop.bbox
        instance_crop = labels[z0:z1, y0:y1, x0:x1] == instance_id
        # add padding to avoid border effect
        padded_mask = np.pad(
            instance_crop,
            pad_width=1,
            mode="constant",
            constant_values=False,
        )
        skeleton_crop = skeletonize(padded_mask)
        # remove padding
        skeleton_crop = skeleton_crop[1:-1, 1:-1, 1:-1]
        target = skeleton_labels[z0:z1, y0:y1, x0:x1]
        target[skeleton_crop] = instance_id
    return (skeleton_labels > 0).astype('uint8')


def reconstruct_centroid_from_mask(labels:np.ndarray) -> np.ndarray:
    assert labels.ndim == 3, "labels should be a 3d array"
    centroid_labels = np.zeros_like(labels)
    props = regionprops(labels)
    for prop in props:
        instance_id = prop.label
        # prop.centroid, ZYX
        centroid = np.asarray(prop.centroid, dtype=np.float32)
        coords = prop.coords
        if coords.shape[0] == 0:
            continue
        distances = np.sum((coords - centroid) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        z, y, x = coords[nearest_idx]
        centroid_labels[z, y, x] = instance_id
    return (centroid_labels > 0).astype('uint8')


def reconstruct_contour_from_mask(labels: np.ndarray,
                                  mode: Literal["thick", "inner", "outer"]) -> np.ndarray:
    assert labels.ndim == 3, "labels should be a 3d array"
    assert mode in ["thick", "inner", "outer"], "unkonwn contour reconstruction mode"
    contour = find_boundaries(
        labels,
        mode=mode,
        connectivity=1,
        background=0,
    )
    return contour.astype(np.uint8)


def calc_affinities_from_mask(labels: np.ndarray, affinities: dict) -> np.ndarray:
    assert labels.ndim == 3, "labels should be a 3d array"
    z_affinities = affinities.get("z_affinities", [1])
    y_affinities = affinities.get("y_affinities", [1])
    x_affinities = affinities.get("x_affinities", [1])
    labels = np.asarray(labels)
    z_size, y_size, x_size = labels.shape
    n_aff = len(z_affinities)
    affinity_maps = np.zeros(
        shape=(3 * n_aff, z_size, y_size, x_size),
        dtype=np.float32,
    )
    def _calc_one_affinity(
            labels: np.ndarray,
            dz: int = 0,
            dy: int = 0,
            dx: int = 0,
    ) -> np.ndarray:
        if dz == 0 and dy == 0 and dx == 0:
            raise ValueError("Affinity offset cannot be (0, 0, 0).")
        z_size, y_size, x_size = labels.shape
        one_affinity = np.zeros_like(labels, dtype=np.float32)
        # source valid range
        z_src_start = max(0, -dz)
        z_src_end = min(z_size, z_size - dz)
        y_src_start = max(0, -dy)
        y_src_end = min(y_size, y_size - dy)
        x_src_start = max(0, -dx)
        x_src_end = min(x_size, x_size - dx)
        # destination valid range
        z_dst_start = z_src_start + dz
        z_dst_end = z_src_end + dz
        y_dst_start = y_src_start + dy
        y_dst_end = y_src_end + dy
        x_dst_start = x_src_start + dx
        x_dst_end = x_src_end + dx
        src = labels[
            z_src_start:z_src_end,
            y_src_start:y_src_end,
            x_src_start:x_src_end,
        ]
        dst = labels[
            z_dst_start:z_dst_end,
            y_dst_start:y_dst_end,
            x_dst_start:x_dst_end,
        ]
        same_instance = (src == dst) & (src != 0)
        one_affinity[
            z_src_start:z_src_end,
            y_src_start:y_src_end,
            x_src_start:x_src_end,
        ] = same_instance.astype(np.float32)
        return one_affinity
    for i, (dz,dy,dx) in enumerate(zip(z_affinities, y_affinities, x_affinities)):
        affinity_maps[i] = _calc_one_affinity(labels, dz=dz, dy=0, dx=0)
        affinity_maps[n_aff*1+i] = _calc_one_affinity(labels, dz=0, dy=dy, dx=0)
        affinity_maps[n_aff*2+i] = _calc_one_affinity(labels, dz=0, dy=0, dx=dx)
    return affinity_maps.astype('uint8')


def watershed_with_sk(
    raw_logits: np.ndarray,
    ch_names: list[str],
    seed_chs: list[str],
    seed_chs_thresh: list[float | str],
    topo_surface_ch: str,
    growth_mask_chs: list[str],
    growth_mask_chs_thresh: list[float | str],
    skeleton_label: np.ndarray | None = None,
) -> np.ndarray:
    """Convert F/C/P predictions into an instance-label image by watershed.

    When ``skeleton_label`` is provided, it must be a spatial label image in
    which 0 is background and every positive integer is a skeleton/neuronal ID.
    A connected seed component is retained only when it overlaps at least one
    positive skeleton ID. The retained seed inherits that skeleton ID, so the
    watershed result preserves the skeleton IDs instead of generating new,
    sequential instance IDs.

    If one connected seed component overlaps multiple skeleton IDs, the seed is
    split by nearest overlapping skeleton voxel. This prevents a connected seed
    from merging multiple skeleton-associated neurons into one marker.

    Parameters
    ----------
    raw_logits:
        Network output in ``CZYX``, ``ZYXC``, ``CYX`` or ``YXC`` order. Values
        may be probabilities in ``[0, 1]`` or raw logits. Values outside
        ``[0, 1]`` cause the complete input to be converted with sigmoid.
    ch_names:
        Channel names corresponding to the channel axis of ``raw_logits``.
    seed_chs:
        Channels used to construct the seed mask. Conditions are combined with
        logical AND. For channel ``C``, values at or below the threshold are
        retained; for other channels, values above the threshold are retained.
    seed_chs_thresh:
        One threshold per seed channel. Each value may be a number in ``[0, 1]``
        or ``"auto"`` for Otsu thresholding. An empty list means all ``"auto"``.
    topo_surface_ch:
        Channel used as watershed elevation. ``C`` is used directly; positive
        evidence channels such as ``F`` and ``P`` are inverted to ``1 - p``.
    growth_mask_chs:
        Channels defining the region in which markers may grow. Conditions are
        combined with logical AND.
    growth_mask_chs_thresh:
        One threshold per growth-mask channel. An empty list means all
        ``"auto"``.
    skeleton_label:
        Optional integer skeleton-label image with the same spatial shape as the
        prediction. ``0`` means background and positive values are skeleton IDs.
        Every skeleton voxel is used directly as a watershed marker. A predicted
        seed component is retained only if it overlaps a positive skeleton ID; the
        retained component inherits that ID. Predicted seed components without any
        skeleton overlap are discarded.

    Returns
    -------
    np.ndarray
        Watershed instance labels in ``ZYX`` or ``YX`` order, dtype ``uint16``.
        Background is 0. Without ``skeleton_label``, instances are numbered from
        1. With ``skeleton_label``, positive output IDs match skeleton IDs.
    """
    raw_logits = np.asarray(raw_logits)
    ch_names = list(ch_names)

    if raw_logits.ndim != 4:
        raise ValueError("raw_logits must be a channel-first/channel-last 3D array: (C,Z,Y,X), or (Z,Y,X,C).")
    if not ch_names:
        raise ValueError("ch_names must not be empty.")
    if len(set(ch_names)) != len(ch_names):
        raise ValueError(f"ch_names contains duplicate names: {ch_names!r}.")

    # Normalize to channel-first: C + spatial dimensions.
    if raw_logits.shape[0] == len(ch_names):
        probability = raw_logits
    elif raw_logits.shape[-1] == len(ch_names):
        probability = np.moveaxis(raw_logits, -1, 0)
    else:
        raise ValueError(f"Cannot locate the channel axis: raw_logits.shape={raw_logits.shape}, len(ch_names)={len(ch_names)}.")

    probability = probability.astype(np.float32, copy=False)
    probability = np.nan_to_num(probability,nan=0.0,posinf=1.0,neginf=0.0,)
    # SkNeXt uses linear output heads. Convert raw logits when needed.
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("raw logits must be in range [0, 1].")

    channel_index = {name: index for index, name in enumerate(ch_names)}

    def require_channels(names: list[str], argument_name: str) -> None:
        missing = [name for name in names if name not in channel_index]
        if missing:
            raise ValueError(f"{argument_name} contains unavailable channels {missing}; available channels are {ch_names}.")

    require_channels(seed_chs, "seed_chs")
    require_channels([topo_surface_ch], "topo_surface_ch")
    require_channels(growth_mask_chs, "growth_mask_chs")

    def normalize_thresholds(
        thresholds: list[float | str],
        channels: list[str],
        argument_name: str,
    ) -> list[float | str]:
        if not thresholds:
            return ["auto"] * len(channels)
        result = list(thresholds)
        if len(result) != len(channels):
            raise ValueError(f"{argument_name} must have one value per channel: got {len(result)} thresholds for {len(channels)} channels.")
        return result

    seed_chs_thresh = normalize_thresholds(seed_chs_thresh, seed_chs,"seed_chs_thresh",)
    growth_mask_chs_thresh = normalize_thresholds(growth_mask_chs_thresh, growth_mask_chs,"growth_mask_chs_thresh",)

    def resolve_threshold(channel: np.ndarray, value: float | str) -> float:
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "auto":
                finite = channel[np.isfinite(channel)]
                if finite.size == 0:
                    return 0.5
                if float(finite.min()) == float(finite.max()):
                    return 0.5
                return float(threshold_otsu(finite))
            try:
                threshold = float(text)
            except ValueError as error:
                raise ValueError(f"Threshold must be a number or 'auto', got {value!r}.") from error
        elif isinstance(value, (int, float, np.integer, np.floating)):
            threshold = float(value)
        else:
            raise TypeError("Threshold must be a number or 'auto', got {type(value).__name__}.")
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be finite and within [0, 1], got {threshold}.")
        return threshold

    def threshold_channel(
        channel_name: str,
        threshold_value: float | str,
    ) -> np.ndarray:
        channel = probability[channel_index[channel_name]]
        threshold = resolve_threshold(channel, threshold_value)
        # C is boundary probability: low C means object interior.
        if channel_name == "C":
            return channel <= threshold
        # F/P and semantic channels are positive evidence maps.
        return channel > threshold

    spatial_shape = probability.shape[1:]

    seed_mask = np.ones(spatial_shape, dtype=bool)
    for channel_name, threshold_value in zip(seed_chs, seed_chs_thresh):
        seed_mask &= threshold_channel(channel_name, threshold_value)
    growth_mask = np.ones(spatial_shape, dtype=bool)
    for channel_name, threshold_value in zip(growth_mask_chs, growth_mask_chs_thresh,):
        growth_mask &= threshold_channel(channel_name, threshold_value)
    # Face-connected components: 4-connectivity in 2D and 6-connectivity in 3D.
    connectivity = ndi.generate_binary_structure(seed_mask.ndim, 1)

    if skeleton_label is None:
        # Standard mode: markers must stay inside the allowed growth region.
        seed_mask &= growth_mask
        if not np.any(seed_mask) or not np.any(growth_mask):
            return np.zeros(spatial_shape, dtype=np.uint16)
        markers, marker_count = ndi.label(seed_mask, structure=connectivity)
        if marker_count == 0:
            return np.zeros(spatial_shape, dtype=np.uint16)
        markers = markers.astype(np.int64, copy=False)
    else:
        skeleton_label = np.asarray(skeleton_label)
        if skeleton_label.shape != spatial_shape:
            raise ValueError(f"skeleton_label must have the same spatial shape as the prediction: expected {spatial_shape}, got {skeleton_label.shape}.")
        if not (np.issubdtype(skeleton_label.dtype, np.integer) or np.issubdtype(skeleton_label.dtype, np.bool_)):
            if not np.issubdtype(skeleton_label.dtype, np.number):
                raise TypeError(f"skeleton_label must be an integer ID label image, got dtype={skeleton_label.dtype}.")
            if not np.all(np.isfinite(skeleton_label)):
                raise ValueError("skeleton_label contains NaN or infinite values.")
            rounded = np.rint(skeleton_label)
            if not np.array_equal(skeleton_label, rounded):
                raise ValueError("skeleton_label must contain integer-valued skeleton IDs.")
            skeleton_label = rounded
        if np.any(skeleton_label < 0): raise ValueError("skeleton_label cannot contain negative IDs.")

        max_skeleton_id = int(np.max(skeleton_label, initial=0))
        if max_skeleton_id > np.iinfo(np.uint16).max:
            raise OverflowError(f"skeleton_label contains an ID larger than uint16 can represent: {max_skeleton_id}.")
        skeleton_label = skeleton_label.astype(np.uint16, copy=False)
        skeleton_region = skeleton_label > 0
        if not np.any(skeleton_region):
            # Skeleton-constrained mode has no valid marker IDs. Predicted seeds
            # are intentionally not allowed to create independent instances.
            return np.zeros(spatial_shape, dtype=np.uint16)
        # Skeleton voxels are markers themselves. Include them in the watershed
        # mask even when the predicted growth mask misses part of a skeleton.
        growth_mask |= skeleton_region
        seed_mask &= growth_mask
        markers = np.zeros(spatial_shape, dtype=np.int64)
        # Retain only connected predicted seed components that overlap at least
        # one skeleton ID. The full retained seed inherits the overlapping ID.
        if np.any(seed_mask):
            seed_components, seed_component_count = ndi.label(seed_mask, structure=connectivity,)
            component_slices = ndi.find_objects(seed_components)
            for component_id, component_slice in enumerate(component_slices, start=1,):
                if component_slice is None:
                    continue
                component_local = (seed_components[component_slice] == component_id)
                skeleton_local = skeleton_label[component_slice]
                overlap_local = component_local & (skeleton_local > 0)
                skeleton_ids = np.unique(skeleton_local[overlap_local])
                if skeleton_ids.size == 0:
                    # Predicted seeds without skeleton overlap are discarded.
                    continue
                marker_local = markers[component_slice]
                if skeleton_ids.size == 1:
                    marker_local[component_local] = int(skeleton_ids[0])
                    continue
                # If one connected predicted seed touches multiple skeleton IDs,
                # split it according to the nearest overlapping skeleton voxel.
                overlap_ids = np.where(overlap_local, skeleton_local,0,)
                _, nearest_indices = ndi.distance_transform_edt(overlap_ids == 0, return_indices=True,)
                nearest_skeleton_ids = overlap_ids[tuple(nearest_indices)]
                marker_local[component_local] = (nearest_skeleton_ids[component_local])
        # Merge the complete skeleton extent into the marker image. Skeleton IDs
        # take precedence at skeleton voxels and also work when no predicted seed
        # overlaps a given skeleton. Disconnected regions carrying the same ID are
        # intentionally treated as parts of the same final instance.
        markers[skeleton_region] = skeleton_label[skeleton_region].astype(np.int64,copy=False,)

    topography = probability[channel_index[topo_surface_ch]]
    if topo_surface_ch == "C":
        elevation = topography
    else:
        # Watershed grows from low basins; F/P confidence is high inside objects.
        elevation = 1.0 - topography

    instances = skimage_watershed(
        image=elevation.astype(np.float32, copy=False),
        markers=markers,
        mask=growth_mask,
        connectivity=connectivity,
        watershed_line=False,)
    if np.max(instances, initial=0) > np.iinfo(np.uint16).max:
        raise OverflowError("Watershed output IDs exceed uint16 range.")
    return instances.astype(np.uint16, copy=False)
