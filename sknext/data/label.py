import numpy as np
import torch
import numpy as np
from typing import Literal
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


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


