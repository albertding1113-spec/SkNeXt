"""Preview or delete directory-backed OME-Zarr stores on a mounted filesystem.

Examples (run on a cluster node permitted to perform filesystem maintenance):
    python scripts/delete_ome_zarr.py /scratch/user/sample.ome.zarr --root /scratch/user
    python scripts/delete_ome_zarr.py /scratch/user/sample.ome.zarr --root /scratch/user --delete

No Zarr package is needed. Deletion is sequential and permanent; stop any jobs
writing these stores first. Object storage URLs and zip stores are unsupported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time


def validate_target(path: Path, root: Path) -> Path:
    # Check before resolving so a link cannot silently select another store.
    if path.is_symlink():
        raise ValueError(f"Refusing a symbolic-link target: {path}")
    target = path.resolve(strict=True)
    if target == root or root not in target.parents:
        raise ValueError(f"Target must be strictly inside {root}: {target}")
    if not target.is_dir() or not target.name.lower().endswith(".zarr"):
        raise ValueError(f"Expected a directory ending in .zarr: {target}")
    # Do not require metadata: interrupted/partially deleted stores are valid
    # cleanup targets too. The explicit paths and root define the scope.
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="Exact store paths")
    parser.add_argument("--root", type=Path, required=True,
                        help="Allowed parent directory; the root itself is never deleted")
    parser.add_argument("--delete", action="store_true",
                        help="Permanently delete; omission only previews paths")
    args = parser.parse_args(argv)

    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir() or root == Path(root.anchor):
            raise ValueError("--root must be an existing directory below the filesystem root")
        # Validate the complete request before deleting any store.
        targets = list(dict.fromkeys(validate_target(path, root) for path in args.paths))
        for target in targets:
            if any(parent in targets for parent in target.parents):
                raise ValueError(f"Nested targets are not allowed: {target}")
        for target in targets:
            print(f"{'DELETE' if args.delete else 'PREVIEW'}: {target}", flush=True)
        if not args.delete:
            print("Nothing deleted. Add --delete to permanently remove these stores.")
            return 0

        for target in targets:
            validate_target(target, root)
            start = time.monotonic()
            print(f"Deleting {target} (large stores may take a while)...", flush=True)
            # rmtree removes internal symlinks rather than following them.
            shutil.rmtree(target)
            print(f"Deleted {target} in {time.monotonic() - start:.1f}s", flush=True)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Deletion stopped; a failed deletion may leave a partial store.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; a partially deleted store may remain.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
