"""Copy a local CZYX OME-Zarr v2 store with new storage chunks.

Usage:
    python scripts/rechunk_ome_zarr.py input.ome.zarr output.ome.zarr
    python scripts/rechunk_ome_zarr.py input.ome.zarr output.ome.zarr --chunks 1 64 1024 1024
    python scripts/rechunk_ome_zarr.py input.ome.zarr output.ome.zarr --build-pyramid

Requires zarr>=3,<4 and the codecs used by the source. All arrays, including
pyramid levels and labels, must have four dimensions in CZYX order. The source
is never modified. An existing destination is rejected, including partial runs.
With --build-pyramid, rebuild five lower-resolution levels in the converted
volume after copying. Defaults to nearest-neighbor for segmentation labels;
use --pyramid-mode mean for probabilities or intensity images. This option
requires the SkNeXt environment and targets the root group's first multiscale.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shutil

import zarr


def rechunk(source: Path, destination: Path, chunks: tuple[int, ...], *,
            build_pyramid: bool = False, pyramid_mode: str = "nearest") -> None:
    source = source.resolve()
    destination = destination.resolve()
    if len(chunks) != 4 or any(value <= 0 for value in chunks):
        raise ValueError("Chunks must be four positive integers in CZYX order.")
    if pyramid_mode not in {"nearest", "mean", "max"}:
        raise ValueError("Pyramid mode must be nearest, mean, or max.")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested paths.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if not (source / ".zgroup").is_file():
        raise ValueError("Expected a local OME-Zarr v2 group (containing .zgroup).")

    # Validate all arrays before creating any output.
    array_metadata = sorted(source.rglob(".zarray"))
    if not array_metadata:
        raise ValueError("No Zarr arrays found in the source.")
    for path in array_metadata:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("zarr_format") != 2 or len(metadata["shape"]) != 4:
            raise ValueError(f"Expected a four-dimensional Zarr v2 array: {path.parent}")

    destination.mkdir(parents=True)
    # Copy group metadata and ancillary files, skipping array data. Consolidated
    # metadata must be regenerated because it includes the old chunk sizes.
    for directory, subdirs, files in os.walk(source):
        directory = Path(directory)
        target = destination / directory.relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        if ".zarray" in files:
            subdirs[:] = []
            files = [name for name in files if name.startswith(".")]
        for name in files:
            if name != ".zmetadata":
                shutil.copy2(directory / name, target / name)

    for path in array_metadata:
        relative = path.parent.relative_to(source)
        target = destination / relative
        metadata = json.loads(path.read_text(encoding="utf-8"))
        # Keep dtype, compressor, filters, order, fill value, dimension separator,
        # and OME attributes unchanged. Only the chunk grid changes.
        metadata["chunks"] = list(chunks)
        (target / ".zarray").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        src = zarr.open_array(str(path.parent), mode="r")
        dst = zarr.open_array(str(target), mode="r+")
        counts = tuple((size + chunk - 1) // chunk for size, chunk in zip(src.shape, chunks))
        total = math.prod(counts)
        print(f"{relative}: {src.chunks} -> {dst.chunks}; {total} chunks", flush=True)
        for number, index in enumerate(itertools.product(*(range(n) for n in counts)), 1):
            region = tuple(
                slice(i * chunk, min((i + 1) * chunk, size))
                for i, chunk, size in zip(index, chunks, src.shape)
            )
            dst[region] = src[region]
            if number % 100 == 0 or number == total:
                print(f"  {number}/{total}", flush=True)

    zarr.consolidate_metadata(str(destination), zarr_format=2)
    if build_pyramid:
        if __package__:
            from .build_ome_zarr_pyramid import build_pyramid as build_output_pyramid
        else:
            from build_ome_zarr_pyramid import build_pyramid as build_output_pyramid
        build_output_pyramid(destination, mode=pyramid_mode)
    print(f"Completed: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunks", type=int, nargs=4, default=(1, 64, 1024, 1024),
                        metavar=("C", "Z", "Y", "X"))
    parser.add_argument("--build-pyramid", action="store_true",
                        help="Build/rebuild five pyramid levels in the converted output")
    parser.add_argument("--pyramid-mode", choices=("nearest", "mean", "max"),
                        default="nearest",
                        help="Downsampling with --build-pyramid: nearest for labels (default), "
                             "mean for probabilities/images")
    args = parser.parse_args()
    rechunk(args.source, args.destination, tuple(args.chunks),
            build_pyramid=args.build_pyramid, pyramid_mode=args.pyramid_mode)
