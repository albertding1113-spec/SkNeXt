import argparse
import os
import sys
__version__ = "0.1.0"


def __getattr__(name):
    """Lazily expose SkNeXt without importing the training runtime at package import.

    Args:
        name: Requested package attribute.

    Returns:
        The SkNeXt class when requested.

    Raises:
        AttributeError: The requested attribute is not provided.
    """
    if name == "SkNeXt":
        from sknext._sknext import SkNeXt
        return SkNeXt
    raise AttributeError(name)


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
        required=False,
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
    parser.add_argument("--gui", action="store_true", help="Open the desktop interface.")
    args = parser.parse_args()
    if args.gui:
        from sknext.gui import main as gui_main
        gui_main(args.config)
        return
    if not args.config:
        parser.error("--config is required unless --gui is used")
    del args.gui
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from sknext._sknext import SkNeXt
    print("SkNeXt version:", __version__, flush=True)
    print("All input args: \n", vars(args), flush=True)
    _sknext = SkNeXt(**vars(args))
    _sknext.run()
    sys.exit(0)
