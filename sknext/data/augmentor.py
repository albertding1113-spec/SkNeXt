import random
import cv2
import math
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.draw import line
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from scipy.ndimage import binary_dilation as binary_dilation_scipy
from scipy.ndimage import rotate
from typing import Tuple, Union, Optional, List
from numpy.typing import NDArray
from scipy.ndimage import median_filter, shift as shift_nd
from skimage.transform import AffineTransform, ProjectiveTransform, warp

# img:NDArray #ZYXC, float32
def cutout(
    img: NDArray,                    # ZYXC / YXC
    mask: Optional[NDArray] = None,  # ZYXC / YXC，也兼容 ZYX / YX
    values: tuple[float, float] = (0.01, 0.05),
    size: tuple[float, float] = (0.05, 0.30),
) -> Union[
    NDArray,
    Tuple[NDArray, Optional[NDArray]],
]:
    """
    Randomly replace one cuboid/rectangular region with low-percentile
    image signals and set the corresponding mask region to zero.

    Parameters
    ----------
    img : NDArray
        Image in ZYXC or YXC order.

    mask : NDArray, optional
        Corresponding label in ZYXC/YXC order.
        A label without a channel axis, such as ZYX/YX, is also supported.

    values : tuple of float, default=(0.01, 0.05)
        Percentile interval used to generate the replacement signal.

        Both formats are accepted:
        - (0.01, 0.05): 1st to 5th percentile
        - (1, 5): 1st to 5th percentile

        Percentiles are calculated independently for each image channel.

    size : tuple of float, default=(0.05, 0.30)
        Minimum and maximum side-length ratio of the cutout region.
        One random ratio is sampled independently for every spatial axis.

    Returns
    -------
    NDArray or tuple
        If mask is None:
            augmented_img

        Otherwise:
            augmented_img, augmented_mask
    """
    img = np.asarray(img)
    if img.ndim not in (3, 4):
        raise ValueError(f"`img` must be YXC or ZYXC, but got shape {img.shape}.")
    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(f"`img` must have a numeric dtype, but got {img.dtype}.")
    if len(values) != 2:
        raise ValueError("`values` must contain exactly two values.")
    value_low = float(values[0])
    value_high = float(values[1])
    if value_low > value_high:
        raise ValueError(f"`values[0]` must not exceed `values[1]`, got {values}.")
    # Accept either fractions [0, 1] or percentile values [0, 100].
    if 0.0 <= value_low <= value_high <= 1.0:
        percentile_low = value_low * 100.0
        percentile_high = value_high * 100.0
    elif 0.0 <= value_low <= value_high <= 100.0:
        percentile_low = value_low
        percentile_high = value_high
    else:
        raise ValueError("`values` must be inside [0, 1] or [0, 100].")
    if len(size) != 2:
        raise ValueError("`size` must contain exactly two values.")
    size_low = float(size[0])
    size_high = float(size[1])
    if not (0.0 < size_low <= size_high <= 1.0):
        raise ValueError(f"`size` must satisfy 0 < min <= max <= 1, got {size}.")
    spatial_shape = img.shape[:-1]
    spatial_ndim = len(spatial_shape)
    # Randomly determine the cutout size along every spatial axis.
    cutout_shape = []
    for axis_size in spatial_shape:
        ratio = float(np.random.uniform(size_low, size_high))
        region_size = max(1, int(round(axis_size * ratio)))
        region_size = min(region_size, axis_size)
        cutout_shape.append(region_size)
    # Randomly determine the starting coordinate.
    starts = [int(np.random.randint(0, axis_size - region_size + 1)) for axis_size, region_size in zip(spatial_shape, cutout_shape,)]
    spatial_slices = tuple(slice(start, start + region_size) for start, region_size in zip(starts, cutout_shape))
    img_out = img.copy()
    # Calculate the low-percentile interval independently for each channel.
    spatial_axes = tuple(range(spatial_ndim))
    percentile_bounds = np.percentile(img.astype(np.float32, copy=False), q=(percentile_low, percentile_high), axis=spatial_axes,)
    channel_low_values = np.asarray(percentile_bounds[0])
    channel_high_values = np.asarray(percentile_bounds[1])
    # Fill each channel with its own low-intensity random signal.
    for channel_index in range(img.shape[-1]):
        low_signal = float(channel_low_values[channel_index])
        high_signal = float(channel_high_values[channel_index])
        if high_signal > low_signal:
            fill_value = float(np.random.uniform(low_signal, high_signal))
        else:
            fill_value = low_signal
        img_out[spatial_slices + (channel_index,)] = fill_value
    if mask is None:
        return img_out

    mask = np.asarray(mask)
    # Mask with channel axis: ZYXC / YXC.
    if mask.ndim == img.ndim:
        if mask.shape[:-1] != spatial_shape:
            raise ValueError(f"The spatial shape of `mask` must match `img`: img spatial shape={spatial_shape}, mask spatial shape={mask.shape[:-1]}.")
        mask_slices = spatial_slices + (slice(None),)
    # Mask without channel axis: ZYX / YX.
    elif mask.ndim == img.ndim - 1:
        if mask.shape != spatial_shape:
            raise ValueError(f"The spatial shape of `mask` must match `img`: img spatial shape={spatial_shape}, mask shape={mask.shape}.")
        mask_slices = spatial_slices
    else:
        raise ValueError(f"`mask` must either have the same dimensions as `img` or omit only the final channel dimension. Got img shape={img.shape}, mask shape={mask.shape}.")
    mask_out = mask.copy()
    mask_out[mask_slices] = 0
    return img_out, mask_out

def brightness(
    img: NDArray,
    brightness_factor: Tuple[float, float] = (0, 0),
) -> NDArray:
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got {img.shape}"
    assert np.issubdtype(img.dtype, np.floating), "img must be floating type"
    lo, hi = float(brightness_factor[0]), float(brightness_factor[1])
    assert lo <= hi, "brightness factor is wrong"
    delta = float(np.random.uniform(lo, hi))
    out = img.copy()
    out += delta
    return out

def contrast(img: NDArray, contrast_factor: Tuple[float, float] = (0, 0)) -> NDArray:
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got {img.shape}"
    lo, hi = float(contrast_factor[0]), float(contrast_factor[1])
    assert lo <= hi, "contrast factor is wrong"
    if lo > hi:
        lo, hi = hi, lo
    scale = 1.0 + float(np.random.uniform(lo, hi))
    out = img.copy()
    out *= scale
    return out

def random_rot(
    img: NDArray,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    angles: Union[Tuple[int, int], List[int]] = [0, 360],
    mode: str = "reflect",
    mask_type: str = "as_mask",
) -> Union[
    NDArray,
    Tuple[NDArray, Optional[NDArray], Optional[NDArray]],
]:
    """
    Apply a rotation to input ``image`` and ``mask`` (if provided).

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    angles : List of ints, optional
        List of angles to choose the rotation to be made. E.g. [90,180,360].

    mode : str, optional
        How to fill up the new values created. Options: ``constant``, ``reflect``, ``wrap``, ``symmetric``.

    mask_type : str, optional
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    Returns
    -------
    img : 3D/4D Numpy array
        Rotated image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Rotated mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Rotated heatmap. Returned if ``mask`` is provided. E.g. ``(y, x, channels)`` for ``2D`` or
        ``(y, x, z, channels)`` for ``3D``.
    """
    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got shape {img.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got shape {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heat must be 3D or 4D, got shape {heat.shape}"
    assert len(angles) == 2, "If a tuple is provided it must have length 2"
    lo, hi = float(angles[0]), float(angles[1])
    assert lo <= hi, "Angels parameters wrong"
    angle = float(np.random.uniform(lo, hi))
    # Map "symmetric" to SciPy's "mirror"
    _mode = "mirror" if mode == "symmetric" else mode
    # axes for (y, x) rotation
    axes_img = (1, 0) if img.ndim == 3 else (2, 1)
    def _rotate(arr: NDArray, axes: Tuple[int, int], order: int) -> NDArray:
        orig_dtype = arr.dtype
        out = rotate(arr.astype(np.float32, copy=False), angle=angle, axes=axes, reshape=False, order=order, mode=_mode)
        return out.astype(orig_dtype, copy=False)
    img_out = _rotate(img, axes_img, order=1)
    mask_out = None
    if mask is not None:
        order_mask = 0 if mask_type == "as_mask" else 1
        mask_out = _rotate(mask, axes_img, order=order_mask)
    heat_out = None
    if heat is not None:
        heat_out = _rotate(heat, axes_img, order=1)
    return img_out if mask is None and heat is None else (img_out, mask_out, heat_out)


def zoom(
        img: NDArray,
        zoom_range: Tuple[float, ...] = [0.9, 1.1],
        mask: Optional[NDArray] = None,
        heat: Optional[NDArray] = None,
        zoom_in_z: bool = False,
        mode: str = "reflect",
        mask_type: str = "as_mask",
) -> Union[NDArray, Tuple[NDArray, Optional[NDArray], Optional[NDArray]],]:
    """
    Apply zoom to input ``image`` and ``mask`` (if provided).

    Parameters
    ----------
    img : 3D/4D Numpy array
        Image to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    zoom_range : tuple of floats
        Defines minimum and maximum factors to scale the images. E.g. (0.8, 1.2).

    mask : 3D/4D Numpy array, optional
        Mask to rotate. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to rotate. E.g. ``(y, x, channels)`` for ``2D`` or
        ``(y, x, z, channels)`` for ``3D``.

    zoom_in_z: bool, optional
        Whether to apply or not zoom in Z axis.

    mode : str, optional
        How to fill up the new values created. Options: ``constant``, ``reflect``, ``wrap``, ``symmetric``.

    mask_type : str, optional
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    Returns
    -------
    img : 3D/4D Numpy array
        Zoomed image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Zoomed mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Zoomed heatmap. Returned if ``mask`` is provided. E.g. ``(y, x, channels)`` for ``2D`` or
        ``(y, x, z, channels)`` for ``3D``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    if mask is not None:
        assert mask.ndim in [3, 4], f"Mask must be 3D or 4D, got shape {mask.shape}"
    if heat is not None:
        assert heat.ndim in [3, 4], f"Heatmap must be 3D or 4D, got shape {heat.shape}"
    assert len(zoom_range) == 2, f"Zoom range is supposed to have 2 elements but provided {zoom_range} instead"
    assert zoom_range[0] <= zoom_range[1], "First element of zoom range must be lower than the second one"
    assert zoom_range[0] > 0, "Zoom range values must be greater than 0"
    zoom_selected = random.uniform(zoom_range[0], zoom_range[1])
    mask_order = 0 if mask_type == "as_mask" else 1
    if img.ndim == 4:
        z_zoom = zoom_selected if zoom_in_z else 1
        img_shape = [int(img.shape[0] * z_zoom),
                     int(img.shape[1] * zoom_selected),
                     int(img.shape[2] * zoom_selected)]
        if mask is not None:
            mask_shape = [int(mask.shape[0] * z_zoom),
                          int(mask.shape[1] * zoom_selected),
                          int(mask.shape[2] * zoom_selected)]
    else:
        img_shape = [int(img.shape[0] * zoom_selected),
                     int(img.shape[1] * zoom_selected)]
        if mask is not None:
            mask_shape = [int(mask.shape[0] * zoom_selected),
                          int(mask.shape[1] * zoom_selected),]
    img_shape += [img.shape[-1], ]
    if mask is not None:
        mask_shape += [mask.shape[-1], ]  # type: ignore

    if img_shape != img.shape:
        img_orig_shape = img.shape
        img = resize(img, img_shape, order=1, mode=mode, clip=True, preserve_range=True,anti_aliasing=True)
        if mask is not None:
            mask_orig_shape = mask.shape
            mask = resize(mask, mask_shape, order=mask_order, mode=mode, clip=True, preserve_range=True, anti_aliasing=True)
        if heat is not None:
            heat = resize(heat, img_shape[:-1], order=1, mode=mode, clip=True, preserve_range=True, anti_aliasing=True)
        if zoom_selected >= 1:
            img = center_crop_single(img, img_orig_shape)
            if mask is not None:
                mask = center_crop_single(mask, mask_orig_shape)
            if heat is not None:
                heat = center_crop_single(heat, img_orig_shape[:-1])
        else:
            if img.ndim == 4:
                img_pad_tup = (
                    (int((img_orig_shape[0] - img_shape[0]) // 2), math.ceil((img_orig_shape[0] - img_shape[0]) / 2),),
                    (int((img_orig_shape[1] - img_shape[1]) // 2), math.ceil((img_orig_shape[1] - img_shape[1]) / 2),),
                    (int((img_orig_shape[2] - img_shape[2]) // 2), math.ceil((img_orig_shape[2] - img_shape[2]) / 2),),
                    (0, 0),
                )
                if mask is not None:
                    mask_pad_tup = (
                        (int((mask_orig_shape[0] - mask_shape[0]) // 2), math.ceil((mask_orig_shape[0] - mask_shape[0]) / 2),),
                        (int((mask_orig_shape[1] - mask_shape[1]) // 2), math.ceil((mask_orig_shape[1] - mask_shape[1]) / 2),),
                        (int((mask_orig_shape[2] - mask_shape[2]) // 2), math.ceil((mask_orig_shape[2] - mask_shape[2]) / 2),),
                        (0, 0),
                    )
            else:
                img_pad_tup = (
                    (int((img_orig_shape[0] - img_shape[0]) // 2), math.ceil((img_orig_shape[0] - img_shape[0]) / 2),),
                    (int((img_orig_shape[1] - img_shape[1]) // 2), math.ceil((img_orig_shape[1] - img_shape[1]) / 2),),
                    (0, 0),
                )
                if mask is not None:
                    mask_pad_tup = (
                        (int((mask_orig_shape[0] - mask_shape[0]) // 2), math.ceil((mask_orig_shape[0] - mask_shape[0]) / 2),),
                        (int((mask_orig_shape[1] - mask_shape[1]) // 2), math.ceil((mask_orig_shape[1] - mask_shape[1]) / 2),),
                        (0, 0),
                    )
            img = np.pad(img, img_pad_tup, mode)  # type: ignore
            if mask is not None:
                mask = np.pad(mask, mask_pad_tup, mode)  # type: ignore
            if heat is not None:
                heat = np.pad(heat, img_pad_tup, mode)  # type: ignore
    if mask is None:
        return img
    else:
        return img, mask, heat


def gamma_contrast(img: NDArray, gamma: Tuple[float, float] = (0, 1)) -> NDArray:
    """
    Apply gamma contrast to input ``image``.

    Parameters
    ----------
    img : Numpy array
        Image to transform. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    gamma : tuple of 2 floats, optional
        Range of gamma intensity. E.g. ``(0.8, 1.3)``.

    Returns
    -------
    img : Numpy array
        Transformed image. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    assert len(gamma) == 2, "Gamma is supposed to have 2 elements but provided {} instead".format(gamma)
    assert gamma[0] <= gamma[1], "First element of gamma must be lower than the second one"
    assert gamma[0] > 0, "Gamma values must be greater than 0"
    _gamma = random.uniform(gamma[0], gamma[1])

    return adjust_gamma(np.clip(img, 0, 1), gamma=_gamma)  # type: ignore

def shear(
    image: NDArray,
    shear: tuple,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    cval: float = 0,
    mask_type: str = "as_mask",
    mode: str = "constant",
):
    """
    Apply a shear transformation to an image (and optional mask/heatmap).

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to shear. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to shear. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to shear. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    shear : tuple
        Shear range (min, max) in degrees for both x and y directions.

    cval : float
        Value used for points outside the boundaries.

    mask_type : str
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    mode : str
        Points outside boundaries are filled according to this mode.

    Returns
    -------
    img : 3D/4D Numpy array
        Sheared image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Sheared mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Sheared heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    """
    # Random shear (deg)
    shear_x = random.randint(shear[0], shear[1])
    shear_y = random.randint(shear[0], shear[1])

    def _restore_channels(original, warped):
        # If a single-channel input comes back as (H,W), expand to (H,W,1)
        if original is not None and original.ndim >= 3 and original.shape[-1] == 1 and warped.ndim == 2:
            return warped[..., np.newaxis]
        return warped

    def _warp_hwc(arr_hwc: NDArray, tform, order: int, cval: float, mode: str) -> NDArray:
        H, W = arr_hwc.shape[:2]
        orig_dtype = arr_hwc.dtype
        out = warp(
            arr_hwc,
            inverse_map=tform,
            cval=cval,
            mode=mode,
            order=order,
            output_shape=(H, W),
            preserve_range=True,     # keep original value range
        )
        # For masks (nearest/bilinear), cast back to original dtype
        if orig_dtype.kind != "f":
            out = out.astype(orig_dtype, copy=False)
        # Ensure 3D with channel axis if a single channel was reduced
        out = _restore_channels(arr_hwc, out)
        return out

    # Get spatial size and build transform on (H,W)
    if image.ndim == 3:         # (y, x, c)
        H_img, W_img = image.shape[:2]
        tform = _build_shear_matrix_skimage((H_img, W_img), np.deg2rad(shear_x), np.deg2rad(shear_y))
        img = _warp_hwc(image, tform, order=3, cval=cval, mode=mode)

        m = None
        if mask is not None:
            H_mask, W_mask = mask.shape[:2]
            # same transform (center/size differs? -> rebuild with mask size)
            tform_m = _build_shear_matrix_skimage((H_mask, W_mask), np.deg2rad(shear_x), np.deg2rad(shear_y))
            mask_order = 0 if mask_type == "as_mask" else 1
            m = _warp_hwc(mask, tform_m, order=mask_order, cval=cval, mode=mode)

        h = None
        if heat is not None:
            H_heat, W_heat = heat.shape[:2]
            heat_mins, heat_maxes = np.min(heat, axis=tuple(range(heat.ndim - 1))), np.max(heat, axis=tuple(range(heat.ndim - 1)))
            tform_h = _build_shear_matrix_skimage((H_heat, W_heat), np.deg2rad(shear_x), np.deg2rad(shear_y))
            h = _warp_hwc(heat, tform_h, order=3, cval=cval, mode=mode)
            np.clip(h, heat_mins, heat_maxes, out=h)
        return img, m, h

    elif image.ndim == 4:       # (z, y, x, c)
        Z, H, W, C = image.shape
        tform = _build_shear_matrix_skimage((H, W), np.deg2rad(shear_x), np.deg2rad(shear_y))

        # Image
        img_out = np.empty_like(image)
        for z in range(Z):
            img_out[z] = _warp_hwc(image[z], tform, order=3, cval=cval, mode=mode)

        # Mask
        m_out = None
        if mask is not None:
            Zm, Hm, Wm, Cm = mask.shape
            assert (Zm, Hm, Wm) == (Z, H, W), "mask shape must match image (z,y,x)"
            tform_m = _build_shear_matrix_skimage((Hm, Wm), np.deg2rad(shear_x), np.deg2rad(shear_y))
            mask_order = 0 if mask_type == "as_mask" else 1
            m_out = np.empty_like(mask)
            for z in range(Z):
                m_out[z] = _warp_hwc(mask[z], tform_m, order=mask_order, cval=cval, mode=mode)

        # Heat
        h_out = None
        if heat is not None:
            Zh, Hh, Wh, Ch = heat.shape
            assert (Zh, Hh, Wh) == (Z, H, W), "heat shape must match image (z,y,x)"
            tform_h = _build_shear_matrix_skimage((Hh, Wh), np.deg2rad(shear_x), np.deg2rad(shear_y))
            h_out = np.empty_like(heat)
            for z in range(Z):
                h_out[z] = _warp_hwc(heat[z], tform_h, order=3, cval=cval, mode=mode)
        return img_out, m_out, h_out
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim} (expected 3 or 4)")


def _build_shear_matrix_skimage(image_shape: tuple,
                                shear_x_rad: float,
                                shear_y_rad: float,
                                shift_add: tuple = (0.5, 0.5)) -> ProjectiveTransform:
    """
    Build an affine transformation matrix for shear augmentation using skimage.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width, ...).

    shear_x_rad : float
        Shear angle in radians for the x direction.

    shear_y_rad : float
        Shear angle in radians for the y direction.

    shift_add : tuple, optional
        Additional shift to apply when centering the transformation.

    Returns
    -------
    matrix : AffineTransform
        Affine transformation matrix for shear.
    """
    h, w = image_shape[:2]
    if h == 0 or w == 0:
        return AffineTransform()

    shift_y = h / 2.0 - shift_add[0]
    shift_x = w / 2.0 - shift_add[1]

    matrix_to_topleft = AffineTransform(translation=[-shift_x, -shift_y])
    matrix_to_center = AffineTransform(translation=[shift_x, shift_y])

    matrix_shear_x = AffineTransform(shear=shear_x_rad)

    matrix_shear_y_rot = AffineTransform(rotation=-np.pi / 2)
    matrix_shear_y = AffineTransform(shear=shear_y_rad)
    matrix_shear_y_rot_inv = AffineTransform(rotation=np.pi / 2)

    # Correct order: shear_x then shear_y (via rotated frame)
    matrix = (
        matrix_to_topleft
        + matrix_shear_x
        + matrix_shear_y_rot
        + matrix_shear_y
        + matrix_shear_y_rot_inv
        + matrix_to_center
    )
    return matrix


def shift(
    image: NDArray,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    shift_range: Optional[tuple] = None,
    cval: float = 0,
    mask_type: str = "as_mask",
    mode: str = "reflect",
):
    """
    Shift an image (and optional mask/heatmap) by a random amount within a range.

    Parameters
    ----------
    image : 3D/4D Numpy array
        Image to shift. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Mask to shift. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Heatmap (float mask) to shift. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    shift_range : Optional[tuple]
        Range (min, max) for random shift in both x and y directions.

    cval : float
        Value used for points outside the boundaries.

    mask_type : str
        How to treat the mask during interpolation. Either as "as_mask" (order 0) or "as_image" (order 1).

    mode : str
        Points outside boundaries are filled according to this mode.

    Returns
    -------
    img : 3D/4D Numpy array
        Shifted image. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    mask : 3D/4D Numpy array, optional
        Shifted mask. E.g. ``(y, x, channels)`` for ``2D`` or  ``(y, x, z, channels)`` for ``3D``.

    heat : 3D/4D Numpy array, optional
        Shifted heatmap. E.g. ``(y, x, channels)`` for ``2D`` or ``(y, x, z, channels)`` for ``3D``.

    """
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heat must be 3D or 4D, got {heat.shape}"
    assert shift_range is not None and len(shift_range) == 2, \
        f"shift_range must be (min, max); got {shift_range}"
    # Get spatial size for (y, x)
    if image.ndim == 3:        # (y, x, c)
        h, w = image.shape[:2]
    else:                      # (z, y, x, c)
        h, w = image.shape[1:3]
    # Sample a percentage and convert to pixel shifts
    shift_perc = random.uniform(shift_range[0], shift_range[1])
    x_pix = int(round(shift_perc * w))
    y_pix = int(round(shift_perc * h))

    # Build per-array shift tuples (keep z and c fixed)
    def get_shift_tuple(arr, x, y):
        if arr.ndim == 3:           # (y, x, c)
            return (y, x, 0)
        elif arr.ndim == 4:         # (z, y, x, c)
            return (0, y, x, 0)
        else:
            raise ValueError(f"Unsupported ndim: {arr.ndim}")
    # Shift image
    img = shift_nd(image, get_shift_tuple(image, x_pix, y_pix), order=3, mode=mode, cval=cval)
    # Shift mask
    if mask is not None:
        order_mask = 0 if mask_type == "as_mask" else 1
        mask = shift_nd(mask, get_shift_tuple(mask, x_pix, y_pix), order=order_mask, mode=mode, cval=cval)
    # Shift heatmap
    if heat is not None:
        heat_mins, heat_maxes = np.min(heat, axis=tuple(range(heat.ndim - 1))), np.max(heat, axis=tuple(range(heat.ndim - 1)))
        heat = shift_nd(heat, get_shift_tuple(heat, x_pix, y_pix), order=3, mode=mode, cval=cval)
        np.clip(heat, heat_mins, heat_maxes, out=heat)
    return img, mask, heat


def flip_horizontal(image: NDArray, mask: Optional[NDArray] = None, heat: Optional[NDArray] = None):
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heatmap must be 3D or 4D, got {heat.shape}"
    if image.ndim == 3:
        img = image[::-1]
        mask = mask[::-1] if mask is not None else None
        heat = heat[::-1] if heat is not None else None
    else:
        img = image[:, ::-1]
        mask = mask[:, ::-1] if mask is not None else None
        heat = heat[:, ::-1] if heat is not None else None
    return img, mask, heat


def flip_vertical(image: NDArray, mask: Optional[NDArray] = None, heat: Optional[NDArray] = None):
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if mask is not None:
        assert mask.ndim in (3, 4), f"Mask must be 3D or 4D, got {mask.shape}"
    if heat is not None:
        assert heat.ndim in (3, 4), f"Heatmap must be 3D or 4D, got {heat.shape}"
    if image.ndim == 3:
        img = image[:, ::-1]
        mask = mask[:, ::-1] if mask is not None else None
        heat = heat[:, ::-1] if heat is not None else None
    else:
        img = image[:, :, ::-1]
        mask = mask[:, :, ::-1] if mask is not None else None
        heat = heat[:, :, ::-1] if heat is not None else None
    return img, mask, heat


def gaussian_blur(image: NDArray, sigma: float | tuple = (0.5, 1.5)):
    if isinstance(sigma, tuple):
        sigma = random.uniform(sigma[0], sigma[1])
    return gaussian(image, sigma=sigma)


def median_blur(image: NDArray, k_range: Optional[tuple] = None):
    assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"
    if k_range is None or len(k_range) != 2:
        raise ValueError("k_range must be provided and have length 2")
    k = int(random.randint(k_range[0], k_range[1]))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return image
    # Build filter window that does NOT mix z or channels
    if image.ndim == 3:  # (y, x, c)
        size = (k, k, 1)
    else:  # (z, y, x, c)
        size = (1, k, k, 1)
    return median_filter(image, size=size)


def elastic(
    image: NDArray,
    mask: Optional[NDArray] = None,
    heat: Optional[NDArray] = None,
    alpha: float | tuple = 14,
    sigma: float | tuple = 4,
    mask_type: str = "as_mask",
    cval: float = 0,
    mode: str = "constant",
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
    """Apply one shared 2D elastic displacement field to image, mask and heatmap.

    Supported layouts are ``(Y, X, C)`` and ``(Z, Y, X, C)``. For a 3D
    volume, the same ``(Y, X)`` displacement field is broadcast to every
    Z-slice. Image, mask and heatmap always use the same displacement field;
    only their interpolation orders differ.
    """
    assert image.ndim in (3, 4), (
        f"Image must be (Y,X,C) or (Z,Y,X,C), got {image.shape}"
    )
    assert mask_type in ("as_mask", "as_image"), (
        "mask_type must be 'as_mask' or 'as_image'."
    )
    assert mode in _MAPPING_MODE_SCIPY_CV2, (
        f"Unsupported elastic mode: {mode}. "
        f"Available modes: {tuple(_MAPPING_MODE_SCIPY_CV2.keys())}"
    )

    def _spatial_shape(arr: NDArray) -> tuple[int, int]:
        return arr.shape[:2] if arr.ndim == 3 else arr.shape[1:3]

    image_spatial_shape = _spatial_shape(image)

    if mask is not None:
        assert mask.ndim == image.ndim, (
            f"Mask ndim must match image ndim: image={image.shape}, mask={mask.shape}"
        )
        assert _spatial_shape(mask) == image_spatial_shape, (
            "Mask spatial shape must match image spatial shape: "
            f"image={image.shape}, mask={mask.shape}"
        )
        if image.ndim == 4:
            assert mask.shape[0] == image.shape[0], (
                f"Mask Z size must match image Z size: image={image.shape}, mask={mask.shape}"
            )

    if heat is not None:
        assert heat.ndim == image.ndim, (
            f"Heatmap ndim must match image ndim: image={image.shape}, heat={heat.shape}"
        )
        assert _spatial_shape(heat) == image_spatial_shape, (
            "Heatmap spatial shape must match image spatial shape: "
            f"image={image.shape}, heat={heat.shape}"
        )
        if image.ndim == 4:
            assert heat.shape[0] == image.shape[0], (
                f"Heatmap Z size must match image Z size: image={image.shape}, heat={heat.shape}"
            )

    # Use a local RNG when random_seed is provided, otherwise retain the
    # project-wide np.random seed configured by set_seed().
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

    def _sample_parameter(value: float | tuple, name: str) -> float:
        if isinstance(value, tuple):
            assert len(value) == 2, f"{name} range must contain two values."
            low, high = float(value[0]), float(value[1])
            assert low <= high, f"{name} range must satisfy low <= high."
            return float(rng.uniform(low, high))
        return float(value)

    alpha_value = _sample_parameter(alpha, "alpha")
    sigma_value = _sample_parameter(sigma, "sigma")
    assert alpha_value >= 0, "alpha must be >= 0."
    assert sigma_value > 0, "sigma must be > 0."

    height, width = image_spatial_shape

    # Pad the random field before Gaussian smoothing to reduce edge artifacts.
    if sigma_value < 3.0:
        kernel_size = 3.3 * sigma_value
    elif sigma_value < 5.0:
        kernel_size = 2.9 * sigma_value
    else:
        kernel_size = 2.6 * sigma_value
    padding = int(max(kernel_size, 5))
    if padding % 2 == 0:
        padding += 1

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding

    # Generate dx/dy exactly once. Reusing these fields is essential for
    # maintaining pixel-wise registration between image and supervision.
    dx_random = rng.uniform(-1.0, 1.0, size=(padded_height, padded_width)).astype(np.float32)
    dy_random = rng.uniform(-1.0, 1.0, size=(padded_height, padded_width)).astype(np.float32)

    dx = gaussian(
        dx_random,
        sigma=sigma_value,
        preserve_range=True,
    ).astype(np.float32) * alpha_value
    dy = gaussian(
        dy_random,
        sigma=sigma_value,
        preserve_range=True,
    ).astype(np.float32) * alpha_value

    dx = np.ascontiguousarray(dx[padding:-padding, padding:-padding])
    dy = np.ascontiguousarray(dy[padding:-padding, padding:-padding])
    assert dx.shape == (height, width) and dy.shape == (height, width)

    image_out = _map_coordinates(
        image,
        dx,
        dy,
        order=3,
        cval=cval,
        mode=mode,
    )

    mask_out = None
    if mask is not None:
        mask_order = 0 if mask_type == "as_mask" else 1
        mask_out = _map_coordinates(
            mask,
            dx,
            dy,
            order=mask_order,
            cval=cval,
            mode=mode,
        )

    heat_out = None
    if heat is not None:
        reduce_axes = tuple(range(heat.ndim - 1))
        heat_min = np.min(heat, axis=reduce_axes)
        heat_max = np.max(heat, axis=reduce_axes)
        heat_out = _map_coordinates(
            heat,
            dx,
            dy,
            order=3,
            cval=cval,
            mode=mode,
        )
        np.clip(heat_out, heat_min, heat_max, out=heat_out)

    return image_out, mask_out, heat_out


def center_crop_single(
    img: NDArray,
    crop_shape: Tuple[int, ...],
) -> NDArray:
    """
    Extract the central patch from a single image.

    Parameters
    ----------
    img : 3D/4D array
        Image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

    crop_shape : 2/3 int tuple
        Size of the crop. E.g. ``(y, x)`` or ``(z, y, x)``.

    Returns
    -------
    img : 3D/4D Numpy array
        Center crop of the given image. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.
    """
    assert img.ndim in [3, 4], f"Image must be 3D or 4D, got shape {img.shape}"
    if img.ndim == 4:
        z, y, x, c = img.shape
        startz = max(z // 2 - crop_shape[0] // 2, 0)
        starty = max(y // 2 - crop_shape[1] // 2, 0)
        startx = max(x // 2 - crop_shape[2] // 2, 0)
        return img[
            startz : startz + crop_shape[0],
            starty : starty + crop_shape[1],
            startx : startx + crop_shape[2],
        ]
    else:
        y, x, c = img.shape
        starty = max(y // 2 - crop_shape[0] // 2, 0)
        startx = max(x // 2 - crop_shape[1] // 2, 0)
        return img[starty : starty + crop_shape[0], startx : startx + crop_shape[1]]

def _normalize_cv2_input_arr_(arr: NDArray) -> NDArray:
    """
    Ensure array is contiguous and owns its data for cv2 functions.

    Parameters
    ----------
    arr : NDArray
        Input array.
    """
    flags = arr.flags
    if not flags["OWNDATA"]:
        arr = np.copy(arr)
        flags = arr.flags
    if not flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr

def _draw_samples(alpha: float | tuple, sigma: float | tuple, nb_images: int) -> tuple:
    """
    Draw samples for alpha and sigma parameters.

    Parameters
    ----------
    alpha : float or tuple
        Alpha parameter or range (min, max).

    sigma : float or tuple
        Sigma parameter or range (min, max).

    nb_images : int
        Number of samples to draw.

    Returns
    -------
    alphas : NDArray
        Array of drawn alpha values.

    sigmas : NDArray
        Array of drawn sigma values.
    """

    # Use np.random for all randomness
    def draw_param(param: float | tuple, size: tuple) -> NDArray:
        if isinstance(param, (int, float)):
            out = np.full(size, param)
        elif isinstance(param, str):
            out = np.array([param] * size[0], dtype=object)
        elif isinstance(param, tuple):
            out = np.random.uniform(param[0], param[1], size=size) if len(param) == 2 else np.full(size, param[0])
        else:
            out = np.full(size, param)
        return out

    alphas = draw_param(alpha, (nb_images,))
    sigmas = draw_param(sigma, (nb_images,))

    return alphas, sigmas

_MAPPING_MODE_SCIPY_CV2 = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "symmetric": cv2.BORDER_REFLECT,
    "reflect": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
    "nearest": cv2.BORDER_REPLICATE,
}

_MAPPING_ORDER_SCIPY_CV2 = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_CUBIC,
    5: cv2.INTER_CUBIC,
}

def _map_coordinates(image: NDArray, dx: NDArray, dy: NDArray, order: int = 1, cval: float = 0, mode: str = "constant") -> NDArray:
    """
    Map input image to new coordinates defined by displacement fields dx and dy.

    Parameters
    ----------
    image : NDArray
        Input image array.

    dx : NDArray
        Displacement field in x direction.

    dy : NDArray
        Displacement field in y direction.

    order : int
        Interpolation order.

    cval : float
        Value used for points outside the boundaries.

    mode : str
        Points outside boundaries are filled according to this mode.

    Returns
    -------
    result : NDArray
        Transformed image array.
    """

    if image.size == 0:
        return np.copy(image)

    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)

    if order == 0 and image.dtype.name in ["uint64", "int64"]:
        raise Exception(
            "dtypes uint64 and int64 are only supported in "
            "ElasticTransformation for order=0, got order=%d with "
            "dtype=%s." % (order, image.dtype.name)
        )
    assert image.ndim in (3, 4), f"Expected 3D or 4D image, got {image.ndim}D with shape {image.shape}"

    # cv2 params
    border_mode = _MAPPING_MODE_SCIPY_CV2[mode]
    interpolation = _MAPPING_ORDER_SCIPY_CV2[order]
    if image.dtype.kind == "f":
        cval_cast = float(cval)
    else:
        cval_cast = int(cval)

    def _make_maps(h: int, w: int, dx2: NDArray, dy2: NDArray):
        """Build OpenCV remap maps for a single 2D field."""
        y, x = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        x_shifted = x - dx2
        y_shifted = y - dy2

        if interpolation == cv2.INTER_NEAREST:
            return x_shifted, y_shifted
        else:
            # returns (map1, map2) as optimized fixed-point/float maps
            return cv2.convertMaps(x_shifted, y_shifted, cv2.CV_32FC1, nninterpolation=False)

    def _remap_hwcn(arr_hwc: NDArray, map1: NDArray, map2: NDArray) -> NDArray:
        """
        Apply cv2.remap to (H, W, C) with any C (remap supports up to 4 channels at once).
        """
        H, W, C = arr_hwc.shape

        if C <= 4:
            border_val = (cval_cast,) * min(max(C, 1), 4)
            res = cv2.remap(
                _normalize_cv2_input_arr_(arr_hwc),
                map1,
                map2,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_val,
            )
            if res.ndim == 2:
                res = res[..., np.newaxis]
            return res

        # chunk channels in groups of up to 4
        chunks = []
        for i in range(0, C, 4):
            sub = arr_hwc[:, :, i : i + 4]
            border_val = (cval_cast,) * (sub.shape[-1])
            res = cv2.remap(
                _normalize_cv2_input_arr_(sub),
                map1,
                map2,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_val,
            )
            if res.ndim == 2:
                res = res[..., np.newaxis]
            chunks.append(res)
        return np.concatenate(chunks, axis=2)

    if image.ndim == 3:
        # (H, W, C)
        H, W, C = image.shape
        # accept dx/dy as (H,W)
        assert dx.shape == (H, W) and dy.shape == (H, W), \
            f"For 3D image (H,W,C), dx/dy must be (H,W); got dx {dx.shape}, dy {dy.shape}"

        map1, map2 = _make_maps(H, W, dx, dy)
        return _remap_hwcn(np.copy(image), map1, map2)

    else:
        # (Z, H, W, C)
        Z, H, W, C = image.shape
        result = np.empty_like(image)

        # dx/dy: either (H,W) or (Z,H,W)
        per_slice = dx.ndim == 3 and dy.ndim == 3
        if per_slice:
            assert dx.shape == (Z, H, W) and dy.shape == (Z, H, W), \
                f"For per-slice fields, dx/dy must be (Z,H,W); got dx {dx.shape}, dy {dy.shape}"
        else:
            assert dx.shape == (H, W) and dy.shape == (H, W), \
                f"For broadcast fields, dx/dy must be (H,W); got dx {dx.shape}, dy {dy.shape}"
            # precompute shared maps once
            shared_map1, shared_map2 = _make_maps(H, W, dx, dy)

        for z in range(Z):
            if per_slice:
                map1, map2 = _make_maps(H, W, dx[z], dy[z])
            else:
                map1, map2 = shared_map1, shared_map2

            slice_res = _remap_hwcn(image[z], map1, map2)
            result[z] = slice_res

        return result

