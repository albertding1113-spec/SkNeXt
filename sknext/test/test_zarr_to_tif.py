from pathlib import Path
import numpy as np
import zarr
import tifffile


def zarr_nczyx_to_tif(
    zarr_path: str | Path,
    output_dir: str | Path,
    array_name: str = "raw",
    prefix: str | None = None,
    output_axes: str = "CZYX",
    overwrite: bool = True,
) -> list[Path]:

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
            tif_img = one_patch[0]       # ZYX
            axes = "ZYX"
        else:
            if output_axes == "CZYX":
                tif_img = one_patch
                axes = "CZYX"
            elif output_axes == "ZCYX":
                tif_img = np.transpose(one_patch, (1, 0, 2, 3))
                axes = "ZCYX"
            else:  # "ZYXC"
                tif_img = np.transpose(one_patch, (1, 2, 3, 0))
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
    zarr_path = r"G:\Albert\data\260618_SkNeXt_dataset\train.ome.zarr"

    zarr_nczyx_to_tif(
        zarr_path=zarr_path,
        output_dir=r"G:\Albert\data\260618_SkNeXt_dataset\tif_test\debug_raw_tif",
        array_name="raw",
        prefix="raw_patch",
        output_axes="CZYX",
        overwrite=True,
    )

    zarr_nczyx_to_tif(
        zarr_path=zarr_path,
        output_dir=r"G:\Albert\data\260618_SkNeXt_dataset\tif_test\debug_label_tif",
        array_name="label",
        prefix="label_patch",
        output_axes="CZYX",
        overwrite=True,
    )

if __name__ == "__main__":
    test_zarr_nczyx_to_tif()