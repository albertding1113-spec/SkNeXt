import numpy as np
from pathlib import Path
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
import tifffile
from tqdm import tqdm
import shutil
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


def tif_list_to_zarr(
        tif_list: list[Path | str],
        gt_tif_dict: dict[str, Path | str],
        zarr_path: Path | str,
        patch_size: tuple[int, int, int, int] | list[int, int, int, int],#ZYXC
        overlap: tuple[int, int, int] | list[int, int, int],
        padding: tuple[int, int, int] | list[int, int, int],
        preprocess_dict: dict = {},
        channels: list[str] = [],
        channel_extra_opts: dict={},
):
    for key in gt_tif_dict.keys():
        assert len(tif_list) == len(gt_tif_dict[key]), "raw and label images must have the same num"
        if not (key == "instance" or key in channels): raise ValueError(f"Unsupported key name: {key}")
    patch_num = 0
    raw_patch_id = 0
    label_patch_id = 0
    for i in tqdm(range(len(tif_list))):
        _tif_path = tif_list[i]
        _gt_tif_dict = {key:value[i] for key, value in gt_tif_dict.items()}
        _img = read_one_3D_tif(_tif_path, "CZYX")
        _gt_img_dict = {key:read_one_3D_tif(value, "ZYX") for key, value in _gt_tif_dict.items()}
        for _gt_img in _gt_img_dict.values():
            assert _img.shape[1:] == _gt_img.shape, "raw and label image must have the same shape"
        _coord = calculate_patch_coordinates(_img.shape, patch_size[0:3], overlap, padding)
        patch_num += _coord.shape[0]
        # preprocessing raw images and labels
        _img = preprocess_img(_img, preprocess_dict)  # float32, CZYX
        # generate channels from labels
        _gt_ch_list = generate_channels_from_labels(_gt_img_dict, channels, channel_extra_opts)
        if i == 0:
            root, data_raw, data_label = create_patch_ome_zarr(zarr_path, patch_num, patch_size[3], len(_gt_ch_list),
                                                               patch_size[0:3], raw_dtype=_img.dtype, label_dtype="uint8")
        elif i > 0:
            data_raw.resize((patch_num, patch_size[3], *patch_size[0:3]))
            data_label.resize((patch_num, len(_gt_ch_list), *patch_size[0:3]))
        for _patch,_ in patch_coordinates_generator(_img, _coord):
            # reflect_padding
            _patch = reflect_padding_img(_patch, patch_size[0:3])
            data_raw[raw_patch_id, :, :, :, :] = _patch
            raw_patch_id += 1
        for j,_gt_ch in enumerate(_gt_ch_list):
            k = label_patch_id
            for _patch,_ in patch_coordinates_generator(_gt_ch, _coord):
                # reflect_padding
                _patch = reflect_padding_img(_patch, patch_size[0:3])
                data_label[k, j, :, :, :] = _patch
                k += 1
        label_patch_id = k
    return data_raw, data_label
