import argparse
import os
import sys
from sknext._sknext import SkNeXt
__version__ = "0.1.0"


def main():
    """
    sknext
    --config $file
    --run_id $run_id
    --gpu $gpu_id
    """
    parser = argparse.ArgumentParser(
        prog="sknext",
        description=(
            "SkNeXt: ConvNeXt-V2 based semantic and instance segmentation "
            "for multi-type biological microscopy images."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        help="run identifier, e.g. 0.",
        default=0,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help=(
            "GPU id, e.g. 0 or 0,1. "
        ),
    )
    args = parser.parse_args()
    print("SkNeXt version:", __version__, flush=True)
    print("All input args: \n", vars(args), flush=True)
    _sknext = SkNeXt(**vars(args))
    _sknext.run()
    sys.exit(0)