"""Build an existing SkNeXt CZYX OME-Zarr v2 volume's pyramid in place.

Examples (run in the SkNeXt environment):
    python scripts/build_ome_zarr_pyramid.py results.ome.zarr
    python scripts/build_ome_zarr_pyramid.py probabilities.ome.zarr --mode mean
    python scripts/build_ome_zarr_pyramid.py results.ome.zarr --factor 1 2 2 --factor 2 4 4

Level zero is preserved. Existing lower-resolution levels are rebuilt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from a repository checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_FACTORS = ((1, 2, 2), (2, 4, 4), (4, 8, 8), (8, 16, 16), (16, 32, 32))


def build_pyramid(path: Path, *, mode: str = "nearest",
                  factors=DEFAULT_FACTORS) -> None:
    """Add pyramid arrays and NGFF metadata without changing level-zero voxels."""
    from sknext.data.datasetIO import ZarrIOManager

    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"OME-Zarr directory does not exist: {path}")
    with ZarrIOManager(path, mode="r+") as writer:
        print(f"Building pyramid: {path}\nShape: {writer.shape}; mode: {mode}", flush=True)
        writer.build_pyramid(factors=factors, mode=mode)
        print(f"Completed: {len(writer.level_paths) - 1} lower-resolution levels "
              f"({', '.join(writer.level_paths)}). Level zero preserved.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Existing local CZYX OME-Zarr v2 directory")
    parser.add_argument("--mode", choices=("nearest", "mean", "max"), default="nearest",
                        help="nearest for segmentation labels (default); mean for probabilities/images")
    parser.add_argument("--factor", dest="factors", nargs=3, type=int, action="append",
                        metavar=("Z", "Y", "X"),
                        help="Cumulative factor relative to level zero; repeat for each level. "
                             "Defaults to 1,2,2 / 2,4,4 / 4,8,8 / 8,16,16 / 16,32,32.")
    args = parser.parse_args()
    build_pyramid(args.path, mode=args.mode,
                  factors=args.factors if args.factors is not None else DEFAULT_FACTORS)


if __name__ == "__main__":
    main()
