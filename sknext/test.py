from pathlib import Path
import numpy as np
import zarr
import tifffile
import hashlib
import sys
import struct

def zarr_nczyx_to_tif(
    zarr_path: str | Path,
    output_dir: str | Path,
    array_name: str = "raw",          # "raw" 或 "label"
    prefix: str | None = None,
    output_axes: str = "CZYX",        # "CZYX", "ZCYX", "ZYXC"
    overwrite: bool = True,
) -> list[Path]:
    """
    将 zarr 中 NCZYX 格式的数组保存为 N 个 tif 文件。

    zarr array shape:
        (N, C, Z, Y, X)

    每个 tif 保存一个 N:
        raw_000000.tif, raw_000001.tif, ...

    Parameters
    ----------
    zarr_path:
        zarr 文件夹路径，例如 "train.ome.zarr"
    output_dir:
        输出 tif 文件夹
    array_name:
        zarr 内部数组名，通常是 "raw" 或 "label"
    prefix:
        输出文件名前缀。默认使用 array_name
    output_axes:
        - "CZYX": 保持你的训练数据格式
        - "ZCYX": Fiji/ImageJ 中有时更直观
        - "ZYXC": 多通道放到最后
    overwrite:
        是否覆盖已存在 tif

    Returns
    -------
    saved_paths:
        保存出的 tif 路径列表
    """
    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert zarr_path.exists(), f"zarr path does not exist: {zarr_path}"

    root = zarr.open_group(str(zarr_path), mode="r")
    assert array_name in root, f"array_name={array_name} not found in zarr. Available arrays: {list(root.keys())}"

    data = root[array_name]
    assert data.ndim == 5, f"Expected NCZYX 5D array, got shape={data.shape}"

    n, c, z, y, x = data.shape
    prefix = array_name if prefix is None else prefix

    assert output_axes in ["CZYX", "ZCYX", "ZYXC"], \
        "output_axes must be one of ['CZYX', 'ZCYX', 'ZYXC']"

    saved_paths = []

    for i in range(n):
        one_patch = np.asarray(data[i])  # CZYX

        if c == 1:
            # 单通道建议直接保存成 ZYX，更容易在 Fiji 中查看
            tif_img = one_patch[0]       # ZYX
            axes = "ZYX"
        else:
            if output_axes == "CZYX":
                tif_img = one_patch      # CZYX
                axes = "CZYX"
            elif output_axes == "ZCYX":
                tif_img = np.transpose(one_patch, (1, 0, 2, 3))  # CZYX -> ZCYX
                axes = "ZCYX"
            elif output_axes == "ZYXC":
                tif_img = np.transpose(one_patch, (1, 2, 3, 0))  # CZYX -> ZYXC
                axes = "ZYXC"

        save_path = output_dir / f"{prefix}_{i:06d}.tif"

        if save_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {save_path}")

        tifffile.imwrite(
            str(save_path),
            tif_img,
            bigtiff=True,
            metadata={"axes": axes},
        )

        saved_paths.append(save_path)

    print(f"Saved {len(saved_paths)} tif files to: {output_dir}")
    return saved_paths

def test_zarr_nczyx_to_tif():
    """
    手动测试函数：把保存好的 zarr 导出为 tif。
    根据你的路径修改 zarr_path 和 output_dir。
    """
    zarr_path = r"E:\Albert_BigFile\Data\260611_SkNeXt_test\tif_test\train\train.ome.zarr"

    # 导出 raw patches
    raw_output_dir = r"E:\Albert_BigFile\Data\260611_SkNeXt_test\tif_test\debug_raw_tif"
    zarr_nczyx_to_tif(
        zarr_path=zarr_path,
        output_dir=raw_output_dir,
        array_name="raw",
        prefix="raw_patch",
        output_axes="CZYX",
        overwrite=True,
    )

    # 导出 label patches
    label_output_dir = r"E:\Albert_BigFile\Data\260611_SkNeXt_test\tif_test\debug_label_tif"
    zarr_nczyx_to_tif(
        zarr_path=zarr_path,
        output_dir=label_output_dir,
        array_name="label",
        prefix="label_patch",
        output_axes="CZYX",
        overwrite=True,
    )
