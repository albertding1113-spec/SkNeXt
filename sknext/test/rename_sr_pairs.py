"""Incrementally rename paired TIFF files for SkNeXt super-resolution.

Existing pairs retain their sample IDs. Only newly added images are assigned
new sample IDs, after natural sorting inside each directory.

Raw: sample_0031.tif     GT: sample_0031.SR.tif
Legacy GT sample_0001.tif is upgraded to sample_0001.SR.tif.

Edit RAW_DIR, GT_DIR, DRY_RUN, then run:
    python rename_sr_pairs_incremental.py
"""

from pathlib import Path
import re
import uuid

# ---- EDIT THESE SETTINGS ----
RAW_DIR = Path(r"E:\Albert_BigFile\Data\261020_SkNeXt_dataset\dataset\super_resolution_train\raw")
GT_DIR = Path(r"E:\Albert_BigFile\Data\261020_SkNeXt_dataset\dataset\super_resolution_train\gt")
DRY_RUN = False # Preview first. Set False to actually rename.
# -----------------------------

RAW_PATTERN = re.compile(r"^sample_(\d+)\.tif$", re.IGNORECASE)
GT_PATTERN = re.compile(r"^sample_(\d+)(\.SR)?\.tif$", re.IGNORECASE)


def natural_key(path: Path):
    """Sort img2 before img10 (deterministic even with equal numeric parts)."""
    chunks = re.split(r"(\d+)", path.name)
    return (tuple((1, int(x)) if x.isdigit() else (0, x.casefold())
                  for x in chunks), path.name)
def tiff_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    return sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in (".tif", ".tiff")),
        key=natural_key,
    )


def classify(files: list[Path], is_gt: bool):
    """Return {existing_numeric_id: path} and a list of unnumbered new images.

    A GT filename sample_XXXX.tif is treated as a legacy existing ID and
    upgraded to the required sample_XXXX.SR.tif suffix later.
    """
    pattern = GT_PATTERN if is_gt else RAW_PATTERN
    existing: dict[int, Path] = {}
    new: list[Path] = []

    for path in files:
        match = pattern.fullmatch(path.name)
        if match:
            sample_id = int(match.group(1))
            if sample_id in existing:
                raise ValueError(
                    f"Duplicate sample ID {sample_id} in {path.parent}: "
                    f"{existing[sample_id].name} and {path.name}"
                )
            existing[sample_id] = path
        else:
            new.append(path)
    return existing, new


def rename_sr_pairs_incremental(
    raw_dir: str | Path, gt_dir: str | Path, dry_run: bool = True
) -> None:
    raw_dir, gt_dir = Path(raw_dir).resolve(), Path(gt_dir).resolve()
    if raw_dir == gt_dir:
        raise ValueError("Raw and GT folders must be different")

    raw_files = tiff_files(raw_dir)
    gt_files = tiff_files(gt_dir)
    raw_existing, raw_new = classify(raw_files, is_gt=False)
    gt_existing, gt_new = classify(gt_files, is_gt=True)

    # Existing IDs must pair exactly. Never silently realign old training data.
    raw_ids, gt_ids = set(raw_existing), set(gt_existing)
    if raw_ids != gt_ids:
        raise ValueError(
            "Existing numbered pairs do not match. "
            f"Raw IDs without GT: {sorted(raw_ids - gt_ids)}; "
            f"GT IDs without raw: {sorted(gt_ids - raw_ids)}."
        )
    if len(raw_new) != len(gt_new):
        raise ValueError(
            f"Number of new images differs: raw={len(raw_new)}, "
            f"GT={len(gt_new)}. Nothing renamed."
        )

    # Preserve numbered samples, including 0001..0030. Upgrade legacy GT only.
    planned: list[tuple[Path, Path]] = []
    for sample_id in sorted(gt_existing):
        source = gt_existing[sample_id]
        target = gt_dir / f"sample_{sample_id:04d}.SR.tif"
        if source != target:
            planned.append((source, target))
            print(f"[Existing GT] {source.name} -> {target.name}")

    # Append new pairs after the largest existing ID; never reuse old IDs.
    next_id = max(raw_ids, default=-1) + 1
    for sample_id, (raw, gt) in enumerate(zip(raw_new, gt_new), start=next_id):
        base = f"sample_{sample_id:04d}"
        raw_target = raw_dir / f"{base}.tif"
        gt_target = gt_dir / f"{base}.SR.tif"
        planned.extend(((raw, raw_target), (gt, gt_target)))
        print(f"[New {sample_id:04d}] RAW: {raw.name} -> {raw_target.name}")
        print(f"           GT:  {gt.name} -> {gt_target.name}")

    # Detect target collisions before doing anything, including case-only clashes.
    source_keys = {str(src).casefold() for src, _ in planned}
    destination_keys = [str(dst).casefold() for _, dst in planned]
    if len(destination_keys) != len(set(destination_keys)):
        raise ValueError("Two renames would produce the same destination filename")
    for src, dest in planned:
        if str(dest).casefold() != str(src).casefold() and dest.exists():
            if str(dest).casefold() not in source_keys:
                raise FileExistsError(f"Destination already exists: {dest}")

    print(f"\nExisting matched pairs: {len(raw_existing)}")
    print(f"New matched pairs:      {len(raw_new)}")
    print(f"Files to rename:        {len(planned)}")

    if dry_run:
        print("DRY RUN: no changes made. Verify that the sorted new pairs match.")
        return
    if not planned:
        print("Nothing to rename.")
        return

    # Two-phase transaction to prevent filename collisions (with rollback).
    staged: list[list] = []
    try:
        for src, dst in planned:
            temp = src.with_name(f".__sknext_{uuid.uuid4().hex}.tmp")
            src.rename(temp)
            staged.append([src, temp, dst, False])
        for item in staged:
            item[1].rename(item[2])
            item[3] = True
    except Exception:
        for src, temp, dst, finished in reversed(staged):
            if finished and dst.exists():
                dst.rename(temp)
        for src, temp, dst, finished in reversed(staged):
            if temp.exists():
                temp.rename(src)
        raise

    print(f"Completed: added {len(raw_new)} numbered pairs; preserved all existing IDs.")


if __name__ == "__main__":
    rename_sr_pairs_incremental(RAW_DIR, GT_DIR, dry_run=DRY_RUN)