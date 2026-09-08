"""
add_raw_images.py

Copies the untouched raw source image for each processed entry in a
comparison_manifest.json (output of visualize_iterative_refinement.py)
into that same output directory, and records the path in the manifest
so downstream tools (eval_viewer) can find it without guessing a
filename or extension.

Background: visualize_iterative_refinement.py only ever writes overlaid
images (ground truth, baseline, per-round predictions/diffs) — never a
plain copy of the original photo. This script backfills that using the
dataset the evaluation was originally run against, whose dataset.json
already records each stem's exact image_path (and therefore its real
file extension) — no guessing required, and no need to re-run the
(expensive, GPU-bound) inference script.

Output structure
-----------------
<output>/
    <stem>_raw.<ext>              copy of <dataset>/<image_path>, extension preserved
    comparison_manifest.json      updated in place: each processed entry's
                                   paths.raw_image = "<stem>_raw.<ext>"
    comparison_manifest.json.bak  backup of the manifest as it was before this script ran
                                   (only written once, never overwritten by later runs)

Usage
-----
    python add_raw_images.py --dataset <dir> --output <dir> [options]

Options
-------
    --dataset    PATH   Dataset folder containing dataset.json (required)
    --output     PATH   Evaluation output folder containing comparison_manifest.json
                         (required)
    --overwrite          Overwrite already-copied raw images. Default: skip existing files.
    --dry-run             Print what would be copied/updated without writing anything.

Examples
--------
    # Dry run — see what would happen
    python add_raw_images.py --dataset ./dataset --output ./refinement_visuals --dry-run

    # Actually copy the images and update the manifest
    python add_raw_images.py --dataset ./dataset --output ./refinement_visuals
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset_entries(dataset_dir: Path) -> dict[str, dict]:
    p = dataset_dir / "dataset.json"
    if not p.exists():
        sys.exit(f"Error: dataset.json not found in '{dataset_dir}'.")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    if not entries:
        sys.exit("Error: dataset.json contains no entries.")
    return {e["stem"]: e for e in entries}


def load_manifest(output_dir: Path) -> tuple[dict, Path]:
    p = output_dir / "comparison_manifest.json"
    if not p.exists():
        sys.exit(f"Error: comparison_manifest.json not found in '{output_dir}'.")
    with open(p, encoding="utf-8") as f:
        manifest = json.load(f)
    if "predictions" not in manifest:
        sys.exit(f"Error: '{p}' does not look like a comparison_manifest.json (no 'predictions' key).")
    return manifest, p


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def add_raw_images(
    manifest:        dict,
    dataset_entries: dict[str, dict],
    dataset_dir:     Path,
    output_dir:      Path,
    overwrite:       bool,
    dry_run:         bool,
) -> dict[str, int]:
    counts = {
        "copied": 0, "skipped_existing": 0,
        "missing_in_dataset": 0, "missing_source_file": 0,
        "skipped_not_processed": 0,
    }

    for entry in manifest["predictions"]:
        if entry.get("status") != "processed":
            counts["skipped_not_processed"] += 1
            continue

        stem = entry["stem"]
        ds_entry = dataset_entries.get(stem)
        if ds_entry is None:
            print(f"  [MISSING] {stem} — not found in dataset.json")
            counts["missing_in_dataset"] += 1
            continue

        image_path  = ds_entry["image_path"]
        source_path = dataset_dir / image_path
        if not source_path.exists():
            print(f"  [MISSING] {stem} — source file does not exist: {source_path}")
            counts["missing_source_file"] += 1
            continue

        ext             = Path(image_path).suffix
        dest_filename   = f"{stem}_raw{ext}"
        dest_path       = output_dir / dest_filename

        if dest_path.exists() and not overwrite:
            print(f"  [SKIP]        {stem}  (already exists: {dest_filename})")
            counts["skipped_existing"] += 1
        else:
            action = "WOULD-COPY" if dry_run else "COPY"
            print(f"  [{action}]{' ' * max(0, 10 - len(action))}{stem}  →  {dest_filename}")
            if not dry_run:
                shutil.copy2(source_path, dest_path)
            counts["copied"] += 1

        if not dry_run:
            entry.setdefault("paths", {})["raw_image"] = dest_filename

    return counts


def write_manifest_backup(manifest_path: Path) -> Path | None:
    backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
    if backup_path.exists():
        return None  # never overwrite an existing backup
    shutil.copy2(manifest_path, backup_path)
    return backup_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy raw source images into an evaluation output dir and record them in the manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output",  required=True, type=Path, metavar="PATH")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' does not exist.")
    if not args.output.is_dir():
        sys.exit(f"Error: --output '{args.output}' does not exist.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.output}")
    print(f"Mode    : {'DRY RUN (no changes written)' if args.dry_run else 'APPLY'}")
    print()

    dataset_entries       = load_dataset_entries(args.dataset)
    manifest, manifest_path = load_manifest(args.output)

    counts = add_raw_images(
        manifest=manifest,
        dataset_entries=dataset_entries,
        dataset_dir=args.dataset,
        output_dir=args.output,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        backup_path = write_manifest_backup(manifest_path)
        if backup_path is not None:
            print(f"\nManifest backed up to: {backup_path}")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest updated: {manifest_path}")

    print(
        f"\nDone.  Copied: {counts['copied']}  "
        f"Already present: {counts['skipped_existing']}  "
        f"Missing from dataset: {counts['missing_in_dataset']}  "
        f"Missing source file: {counts['missing_source_file']}"
    )

    if counts["missing_in_dataset"] > 0 or counts["missing_source_file"] > 0:
        print("[WARN] Some entries could not be matched to a raw image — see [MISSING] lines above.")


if __name__ == "__main__":
    main()
