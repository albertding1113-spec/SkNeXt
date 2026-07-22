from __future__ import annotations

import gc
import math
import shutil
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
import z5py


def get_default_chunks(
    shape: Sequence[int],
    axes: str | None = None,
) -> tuple[int, ...]:
    """
    根据数组维度和轴顺序自动生成 chunk。

    常见结果：
        YX     -> (256, 256)
        ZYX    -> (16, 256, 256)
        CZYX   -> (1, 16, 256, 256)
        TCZYX  -> (1, 1, 16, 256, 256)
    """
    shape = tuple(int(v) for v in shape)
    ndim = len(shape)

    if axes is not None and len(axes) == ndim:
        axes = axes.upper()
        chunks = []

        for axis, axis_size in zip(axes, shape):
            if axis in {"X", "Y"}:
                chunk_size = 256
            elif axis == "Z":
                chunk_size = 16
            elif axis in {"C", "T"}:
                chunk_size = 1
            elif axis == "S":
                chunk_size = min(axis_size, 4)
            else:
                chunk_size = 1

            chunks.append(min(axis_size, chunk_size))

        return tuple(chunks)

    if ndim == 2:
        target_chunks = (256, 256)

    elif ndim == 3:
        target_chunks = (16, 256, 256)

    elif ndim == 4:
        target_chunks = (1, 16, 256, 256)

    elif ndim == 5:
        target_chunks = (1, 1, 16, 256, 256)

    else:
        target_chunks = (1,) * max(0, ndim - 3) + (16, 256, 256)

    return tuple(
        min(axis_size, chunk_size)
        for axis_size, chunk_size in zip(shape, target_chunks)
    )


def iter_chunk_slices(
    shape: Sequence[int],
    chunks: Sequence[int],
):
    """
    生成覆盖整个数组的分块切片。
    """
    starts_per_axis = [
        range(0, axis_size, chunk_size)
        for axis_size, chunk_size in zip(shape, chunks)
    ]

    for starts in product(*starts_per_axis):
        yield tuple(
            slice(start, min(start + chunk_size, axis_size))
            for start, chunk_size, axis_size in zip(
                starts,
                chunks,
                shape,
            )
        )


def create_nested_dataset(
    n5_file: z5py.N5File,
    dataset_name: str,
    *,
    shape: Sequence[int],
    dtype: np.dtype,
    chunks: Sequence[int],
    compression: str,
    n_threads: int,
):
    """
    创建支持嵌套路径的 N5 dataset。

    例如：
        raw
        labels
        volumes/raw
        volumes/labels
    """
    path_parts = [
        part
        for part in dataset_name.strip("/").split("/")
        if part
    ]

    if not path_parts:
        raise ValueError("dataset_name 不能为空")

    parent = n5_file

    for group_name in path_parts[:-1]:
        parent = parent.require_group(group_name)

    dataset = parent.create_dataset(
        path_parts[-1],
        shape=tuple(int(v) for v in shape),
        dtype=np.dtype(dtype),
        chunks=tuple(int(v) for v in chunks),
        compression=compression,
        n_threads=n_threads,
    )

    return dataset


def inspect_tiff(
    tif_path: str | Path,
    series_index: int = 0,
    pyramid_level: int = 0,
) -> tuple[tuple[int, ...], np.dtype, str | None]:
    """
    读取 TIFF 的 shape、dtype 和 axes，但不加载完整图像。
    """
    tif_path = Path(tif_path)

    with tifffile.TiffFile(tif_path) as tif:
        if not 0 <= series_index < len(tif.series):
            raise IndexError(
                f"{tif_path.name} 的 series_index={series_index} 超出范围，"
                f"共有 {len(tif.series)} 个 series。"
            )

        base_series = tif.series[series_index]

        if not 0 <= pyramid_level < len(base_series.levels):
            raise IndexError(
                f"{tif_path.name} 的 pyramid_level={pyramid_level} 超出范围，"
                f"共有 {len(base_series.levels)} 个 level。"
            )

        series = base_series.levels[pyramid_level]

        shape = tuple(int(v) for v in series.shape)
        dtype = np.dtype(series.dtype)
        axes = getattr(series, "axes", None)

    return shape, dtype, axes


def get_spatial_shape(
    shape: Sequence[int],
    axes: str | None,
) -> tuple[int, ...]:
    """
    获取 TIFF 的空间维度。

    例如：
        CZYX -> ZYX
        ZYX  -> ZYX
        CYX  -> YX
    """
    shape = tuple(int(v) for v in shape)

    if axes is not None and len(axes) == len(shape):
        axes = axes.upper()

        spatial_axes = [
            axis
            for axis in ("Z", "Y", "X")
            if axis in axes
        ]

        if spatial_axes:
            return tuple(
                shape[axes.index(axis)]
                for axis in spatial_axes
            )

    if len(shape) >= 3:
        return shape[-3:]

    return shape[-2:]


def write_tiff_to_n5_dataset(
    tif_path: str | Path,
    n5_file: z5py.N5File,
    dataset_name: str,
    *,
    series_index: int = 0,
    pyramid_level: int = 0,
    chunks: Sequence[int] | None = None,
    compression: str = "gzip",
    n_threads: int = 4,
    output_dtype: np.dtype | str | None = None,
    voxel_size: Sequence[float] | None = None,
    voxel_unit: str = "um",
    is_label: bool = False,
    sanitize_special_label_values: bool = True,
) -> int | None:
    """
    将一个 TIFF 分块写入指定的 N5 dataset。

    当 ``is_label=True`` 时，会在分块写入过程中计算真实最大标签 ID，
    并将其保存为 N5 dataset 的 ``maxId`` 属性，供 Paintera 初始化
    ID service 使用。

    Parameters
    ----------
    sanitize_special_label_values:
        仅对 label 生效。True 时，将负值以及无符号整数顶部的四个
        Paintera 保留值转换为背景 0，避免 ``-3`` 转为 uint32 后变成
        ``4294967293``。False 时，一旦检测到这些值就抛出异常。

    Returns
    -------
    int | None
        label dataset 返回写入的 ``maxId``；raw dataset 返回 None。
    """
    tif_path = Path(tif_path)

    with tifffile.TiffFile(tif_path) as tif:
        base_series = tif.series[series_index]
        series = base_series.levels[pyramid_level]

        axes = getattr(series, "axes", None)

        # 优先以内存映射形式读取，避免加载整个 TIFF
        data = series.asarray(out="memmap")

        shape = tuple(int(v) for v in data.shape)

        if output_dtype is None:
            # 去除大端/小端标志，使用本地字节序
            n5_dtype = np.dtype(data.dtype.name)
        else:
            n5_dtype = np.dtype(output_dtype)

        if is_label:
            if not (
                np.issubdtype(data.dtype, np.integer)
                or np.issubdtype(data.dtype, np.bool_)
            ):
                raise TypeError(
                    "Label TIFF 必须使用整数类型，"
                    f"当前输入类型为 {data.dtype}。"
                )

            if not (
                np.issubdtype(n5_dtype, np.integer)
                or np.issubdtype(n5_dtype, np.bool_)
            ):
                raise TypeError(
                    "Label dataset 必须使用整数类型，"
                    f"当前输出类型为 {n5_dtype}。"
                )

        if chunks is None:
            chunk_shape = get_default_chunks(shape, axes)
        else:
            chunk_shape = tuple(int(v) for v in chunks)

            if len(chunk_shape) != data.ndim:
                raise ValueError(
                    f"Dataset '{dataset_name}' 的 chunks={chunk_shape} "
                    f"有 {len(chunk_shape)} 维，但 TIFF shape={shape} "
                    f"有 {data.ndim} 维。"
                )

            if any(v <= 0 for v in chunk_shape):
                raise ValueError("chunk 中的所有数值必须大于 0")

            chunk_shape = tuple(
                min(axis_size, chunk_size)
                for axis_size, chunk_size in zip(shape, chunk_shape)
            )

        dataset = create_nested_dataset(
            n5_file=n5_file,
            dataset_name=dataset_name,
            shape=shape,
            dtype=n5_dtype,
            chunks=chunk_shape,
            compression=compression,
            n_threads=n_threads,
        )

        # 保存元数据
        dataset.attrs["axes"] = axes or ""
        dataset.attrs["sourceTiff"] = tif_path.name
        dataset.attrs["tiffSeries"] = int(series_index)
        dataset.attrs["tiffPyramidLevel"] = int(pyramid_level)
        dataset.attrs["isLabel"] = bool(is_label)

        if voxel_size is not None:
            voxel_size = tuple(float(v) for v in voxel_size)

            if len(voxel_size) != data.ndim:
                raise ValueError(
                    f"voxel_size={voxel_size} 有 {len(voxel_size)} 维，"
                    f"但数据 shape={shape} 有 {data.ndim} 维。"
                )

            dataset.attrs["voxelSize"] = list(voxel_size)
            dataset.attrs["voxelUnit"] = voxel_unit

        total_chunks = math.prod(
            math.ceil(axis_size / chunk_size)
            for axis_size, chunk_size in zip(shape, chunk_shape)
        )

        print()
        print(f"开始写入 dataset：{dataset_name}")
        print(f"  TIFF       ：{tif_path}")
        print(f"  shape      ：{shape}")
        print(f"  axes       ：{axes}")
        print(f"  input dtype：{data.dtype}")
        print(f"  N5 dtype   ：{n5_dtype}")
        print(f"  chunks     ：{chunk_shape}")
        print(f"  compression：{compression}")
        print(f"  总块数     ：{total_chunks}")

        previous_percent = -1
        max_label_id = 0
        sanitized_label_voxels = 0

        for chunk_index, block_slices in enumerate(
            iter_chunk_slices(shape, chunk_shape),
            start=1,
        ):
            source_block = np.asarray(data[block_slices])

            if is_label:
                # 先在源 dtype 上处理特殊值，避免负数转换为 uint32 后回绕。
                if np.issubdtype(source_block.dtype, np.signedinteger):
                    special_mask = source_block < 0
                elif (
                    np.issubdtype(source_block.dtype, np.unsignedinteger)
                    and np.issubdtype(n5_dtype, np.unsignedinteger)
                ):
                    # 只识别“目标 dtype”顶部的四个 Paintera 保留值。
                    # 例如输出 uint32 时识别 4294967292~4294967295；
                    # 不会误删 uint16 中合法的 65532~65535。
                    output_info = np.iinfo(n5_dtype)
                    reserved_start = int(output_info.max) - 3
                    reserved_end = int(output_info.max)
                    special_mask = (
                        (source_block >= reserved_start)
                        & (source_block <= reserved_end)
                    )
                else:
                    special_mask = np.zeros(source_block.shape, dtype=bool)

                special_count = int(np.count_nonzero(special_mask))

                if special_count > 0:
                    if not sanitize_special_label_values:
                        special_values = np.unique(source_block[special_mask])
                        raise ValueError(
                            f"Label TIFF 中检测到 {special_count} 个特殊/负标签值："
                            f"{special_values[:16].tolist()}。"
                            "请清理这些值，或设置 "
                            "sanitize_special_label_values=True。"
                        )

                    source_block = source_block.copy()
                    source_block[special_mask] = 0
                    sanitized_label_voxels += special_count

                if source_block.size > 0:
                    local_max_id = int(source_block.max())
                    max_label_id = max(max_label_id, local_max_id)

                # 在转换前检查目标 dtype，避免静默溢出或截断。
                if np.issubdtype(n5_dtype, np.bool_):
                    if local_max_id > 1:
                        raise OverflowError(
                            f"Label 最大值 {local_max_id} 不能写入 bool。"
                        )
                else:
                    output_info = np.iinfo(n5_dtype)
                    if local_max_id > int(output_info.max):
                        raise OverflowError(
                            f"Label 最大值 {local_max_id} 超出输出类型 "
                            f"{n5_dtype} 的范围。"
                        )

                    # 避免真实 label ID 占用 Paintera 的顶部四个保留值。
                    if np.issubdtype(n5_dtype, np.unsignedinteger):
                        paintera_reserved_start = int(output_info.max) - 3
                        if local_max_id >= paintera_reserved_start:
                            raise ValueError(
                                f"Label ID {local_max_id} 落入 {n5_dtype} 的 "
                                "Paintera 保留值范围。请改用更大的整数类型。"
                            )

            block = np.ascontiguousarray(
                source_block,
                dtype=n5_dtype,
            )

            dataset[block_slices] = block

            percent = int(chunk_index * 100 / total_chunks)

            if (
                percent >= previous_percent + 5
                or chunk_index == total_chunks
            ):
                print(
                    f"  进度：{chunk_index}/{total_chunks} "
                    f"chunks，{percent}%"
                )
                previous_percent = percent

        if is_label:
            # Paintera 会从 label dataset 本身读取该属性。
            dataset.attrs["maxId"] = int(max_label_id)
            dataset.attrs["backgroundId"] = 0

            print(f"  maxId      ：{max_label_id}")
            if sanitized_label_voxels > 0:
                print(
                    "  已转背景值 ："
                    f"{sanitized_label_voxels} 个特殊/负标签体素"
                )

        del dataset
        del data
        gc.collect()

        print(f"Dataset '{dataset_name}' 写入完成。")

        return int(max_label_id) if is_label else None


def raw_and_label_tif_to_n5(
    raw_tif_path: str | Path,
    label_tif_path: str | Path,
    n5_path: str | Path,
    *,
    raw_dataset_name: str = "raw",
    label_dataset_name: str = "labels",
    raw_chunks: Sequence[int] | None = None,
    label_chunks: Sequence[int] | None = None,
    raw_compression: str = "gzip",
    label_compression: str = "gzip",
    raw_dtype: np.dtype | str | None = None,
    label_dtype: np.dtype | str | None = None,
    raw_series_index: int = 0,
    label_series_index: int = 0,
    raw_pyramid_level: int = 0,
    label_pyramid_level: int = 0,
    raw_voxel_size: Sequence[float] | None = None,
    label_voxel_size: Sequence[float] | None = None,
    voxel_unit: str = "um",
    n_threads: int = 8,
    check_spatial_shape: bool = True,
    overwrite: bool = False,
    sanitize_special_label_values: bool = True,
) -> None:
    """
    将原始 TIFF 和 Label TIFF 写入同一个 N5 文件。

    输出结构：
        output.n5/
        ├── raw/
        └── labels/

    Parameters
    ----------
    raw_tif_path:
        原始荧光图像 TIFF。

    label_tif_path:
        Label 标签 TIFF。

    n5_path:
        输出 N5 文件夹。

    raw_dataset_name:
        原始图像在 N5 中的路径。

    label_dataset_name:
        标签图像在 N5 中的路径。

    raw_chunks:
        原始图像 chunk。

        ZYX:
            (16, 256, 256)

        CZYX:
            (1, 16, 256, 256)

    label_chunks:
        标签图像 chunk，通常为：
            (16, 256, 256)

    raw_dtype:
        原始图像输出类型。None 表示保持原类型。

    label_dtype:
        标签输出类型。None 表示保持原类型。
        实例标签通常推荐 np.uint32。

    check_spatial_shape:
        是否检查原始图像和标签的空间尺寸一致。

        支持：
            raw: CZYX，label: ZYX
            raw: ZYX，label: ZYX

    sanitize_special_label_values:
        是否把 label 中的负值和 Paintera 顶部保留值转换为背景 0。
        推荐保持 True，避免 ``-3`` 被写成 ``4294967293``。
    """
    raw_tif_path = Path(raw_tif_path)
    label_tif_path = Path(label_tif_path)
    n5_path = Path(n5_path)

    if not raw_tif_path.is_file():
        raise FileNotFoundError(f"找不到原始 TIFF：{raw_tif_path}")

    if not label_tif_path.is_file():
        raise FileNotFoundError(f"找不到 Label TIFF：{label_tif_path}")

    if raw_dataset_name.strip("/") == label_dataset_name.strip("/"):
        raise ValueError(
            "raw_dataset_name 和 label_dataset_name 不能相同"
        )

    raw_shape, raw_input_dtype, raw_axes = inspect_tiff(
        raw_tif_path,
        series_index=raw_series_index,
        pyramid_level=raw_pyramid_level,
    )

    label_shape, label_input_dtype, label_axes = inspect_tiff(
        label_tif_path,
        series_index=label_series_index,
        pyramid_level=label_pyramid_level,
    )

    print("输入数据检查：")
    print(
        f"  Raw   ：shape={raw_shape}, "
        f"axes={raw_axes}, dtype={raw_input_dtype}"
    )
    print(
        f"  Label ：shape={label_shape}, "
        f"axes={label_axes}, dtype={label_input_dtype}"
    )

    if check_spatial_shape:
        raw_spatial_shape = get_spatial_shape(
            raw_shape,
            raw_axes,
        )
        label_spatial_shape = get_spatial_shape(
            label_shape,
            label_axes,
        )

        # if raw_spatial_shape != label_spatial_shape:
        #     raise ValueError(
        #         "原始图像和 Label 的空间尺寸不一致：\n"
        #         f"  raw spatial shape   = {raw_spatial_shape}\n"
        #         f"  label spatial shape = {label_spatial_shape}\n"
        #         f"  raw shape/axes      = {raw_shape}/{raw_axes}\n"
        #         f"  label shape/axes    = {label_shape}/{label_axes}"
        #     )

    final_label_dtype = (
        label_input_dtype
        if label_dtype is None
        else np.dtype(label_dtype)
    )

    if not (
        np.issubdtype(final_label_dtype, np.integer)
        or np.issubdtype(final_label_dtype, np.bool_)
    ):
        raise TypeError(
            "Label TIFF 或 label_dtype 必须是整数类型。"
            f"当前类型为：{final_label_dtype}"
        )

    if n5_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"N5 输出路径已经存在：{n5_path}\n"
                "如需覆盖，请设置 overwrite=True。"
            )

        if n5_path.is_dir():
            shutil.rmtree(n5_path)
        else:
            n5_path.unlink()

    n5_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n5_file = z5py.N5File(
        str(n5_path),
        mode="w",
    )

    try:
        # N5 根目录元数据
        n5_file.attrs["rawDataset"] = raw_dataset_name
        n5_file.attrs["labelDataset"] = label_dataset_name
        n5_file.attrs["voxelUnit"] = voxel_unit

        write_tiff_to_n5_dataset(
            tif_path=raw_tif_path,
            n5_file=n5_file,
            dataset_name=raw_dataset_name,
            series_index=raw_series_index,
            pyramid_level=raw_pyramid_level,
            chunks=raw_chunks,
            compression=raw_compression,
            n_threads=n_threads,
            output_dtype=raw_dtype,
            voxel_size=raw_voxel_size,
            voxel_unit=voxel_unit,
            is_label=False,
        )

        label_max_id = write_tiff_to_n5_dataset(
            tif_path=label_tif_path,
            n5_file=n5_file,
            dataset_name=label_dataset_name,
            series_index=label_series_index,
            pyramid_level=label_pyramid_level,
            chunks=label_chunks,
            compression=label_compression,
            n_threads=n_threads,
            output_dtype=label_dtype,
            voxel_size=label_voxel_size,
            voxel_unit=voxel_unit,
            is_label=True,
            sanitize_special_label_values=sanitize_special_label_values,
        )

    finally:
        del n5_file
        gc.collect()

    print()
    print("全部转换完成。")
    print(f"N5 输出路径：{n5_path}")
    print(f"原始图像路径：/{raw_dataset_name}")
    print(f"标签图像路径：/{label_dataset_name}")
    print(f"标签 maxId ：{label_max_id}")


if __name__ == "__main__":
    raw_and_label_tif_to_n5(
        raw_tif_path=r"G:\Albert\data\260424_iterative_proofreading_training\raw\0012\0012.tif",
        label_tif_path=r"G:\Albert\data\260424_iterative_proofreading_training\raw\0012\0012_infer.tif",
        n5_path=r"G:\Albert\data\260424_iterative_proofreading_training\raw\0012\0012_paintera_data.n5",

        # N5 内部的 dataset 名称
        raw_dataset_name="raw",
        label_dataset_name="labels",

        # 假设原始图像和标签都是 ZYX
        raw_chunks=(16, 256, 256),
        label_chunks=(16, 256, 256),

        # 如果原始图像为 CZYX，可以使用：
        # raw_chunks=(1, 16, 256, 256),

        raw_compression="gzip",
        label_compression="gzip",

        # 原始图像保持原来的 dtype
        raw_dtype=None,

        # 实例标签通常使用 uint32
        label_dtype=np.uint32,

        # ZYX 数据的体素大小
        raw_voxel_size=(0.5, 0.15, 0.15),
        label_voxel_size=(0.5, 0.15, 0.15),
        voxel_unit="um",

        n_threads=8,

        # 将负值和 Paintera 的保留值转为背景 0，避免出现 4294967293。
        sanitize_special_label_values=True,
        overwrite=True,
    )