"""
convert_to_sa1b.py

Converts a dataset produced by build_dataset.py into SA-1B format, as
required by the SAM2 official training code (SA1BRawDataset).

SA-1B format places a JSON annotation file next to each image:

    <output_dir>/
        images/
            <stem>.<ext>        # copied from source dataset
            <stem>.json         # generated annotation in SA-1B schema
        train.txt               # copied from source dataset (if found)
        val.txt                 # copied from source dataset (if found)

SA-1B JSON schema (fields required by SA1BRawDataset)
------------------------------------------------------
{
    "image": {
        "image_id": <int>,
        "width": <int>,
        "height": <int>
    },
    "annotations": [
        {
            "id": 1,
            "segmentation": {
                "size": [height, width],
                "counts": "<RLE-encoded binary mask string>"
            },
            "area": <int>,
            "bbox": [x, y, w, h],
            "predicted_iou": 1.0,
            "stability_score": 1.0,
            "crop_box": [0, 0, width, height]
        }
    ]
}

The mask is encoded using COCO RLE (pycocotools). Only pixels with value > 0
in the source mask PNG are treated as foreground.

Usage
-----
    python convert_to_sa1b.py --dataset <dir> --output <dir> [options]

Options
-------
    --dataset   PATH    Dataset folder containing dataset.json, images/, masks/
                        (required)
    --output    PATH    Destination folder for SA-1B formatted data (required)
    --split     PATH    Optional path to a .txt file listing stems to process
                        (one stem per line). If omitted, all entries in
                        dataset.json are converted.
    --fg-threshold INT  Pixel value threshold above which a pixel is foreground.
                        Default: 0 (any non-zero pixel = foreground).
    --overwrite         Overwrite existing output files. Default: skip with warning.
    --dry-run           Print what would be converted without writing anything.
    --copy-splits       Copy train.txt / val.txt from dataset dir to output dir
                        if found. Default: true (pass --no-copy-splits to disable).

Examples
--------
    # Convert full dataset
    python convert_to_sa1b.py --dataset ./dataset --output ./dataset_sa1b

    # Convert only the training split
    python convert_to_sa1b.py --dataset ./dataset --output ./dataset_sa1b \\
        --split ./dataset/train.txt

    # Convert validation split into a separate output folder
    python convert_to_sa1b.py --dataset ./dataset --output ./dataset_sa1b_val \\
        --split ./dataset/val.txt

    # Dry run to verify matching before writing
    python convert_to_sa1b.py --dataset ./dataset --output ./dataset_sa1b --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pycocotools.mask as mask_util
except ImportError:
    sys.exit(
        "Error: pycocotools is required.\n"
        "Install with: pip install pycocotools"
    )


# ---------------------------------------------------------------------------
# RLE encoding
# ---------------------------------------------------------------------------

def encode_mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary uint8 mask (H×W) into COCO RLE format.

    pycocotools requires Fortran-order (column-major) arrays.
    The returned dict has 'size' as [H, W] and 'counts' as a UTF-8 string.
    """
    binary = (mask > 0).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def compute_bbox(binary_mask: np.ndarray) -> list[int]:
    """
    Return [x, y, w, h] bounding box for the foreground region.
    Returns [0, 0, 0, 0] if the mask is empty.
    """
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min()),
        int(ys.max() - ys.min()),
    ]


# ---------------------------------------------------------------------------
# Annotation building
# ---------------------------------------------------------------------------

def build_annotation(
    image_id: int,
    image_path: Path,
    mask_path: Path,
    fg_threshold: int,
) -> dict:
    """
    Load image dimensions and mask, return a complete SA-1B annotation dict.
    """
    img = Image.open(image_path)
    w, h = img.size

    mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary   = (mask_raw > fg_threshold).astype(np.uint8)

    rle  = encode_mask_to_rle(binary)
    area = int(binary.sum())
    bbox = compute_bbox(binary)

    return {
        "image": {
            "image_id": image_id,
            "width":    w,
            "height":   h,
        },
        "annotations": [
            {
                "id":               1,
                "segmentation":     rle,
                "area":             area,
                "bbox":             bbox,
                "predicted_iou":    1.0,
                "stability_score":  1.0,
                "crop_box":         [0, 0, w, h],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_manifest(dataset_dir: Path) -> list[dict]:
    manifest_path = dataset_dir / "dataset.json"
    if not manifest_path.exists():
        sys.exit(
            f"Error: dataset.json not found in '{dataset_dir}'.\n"
            "Make sure --dataset points to a folder produced by build_dataset.py."
        )
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    if not entries:
        sys.exit(f"Error: No entries found in {manifest_path}.")
    return entries


def load_split_filter(split_path: Path) -> set[str] | None:
    """
    Load a stem filter from a .txt file (one stem per line).
    Returns None if no split file is provided.
    """
    if split_path is None:
        return None
    if not split_path.exists():
        sys.exit(f"Error: --split file '{split_path}' does not exist.")
    stems = {
        line.strip()
        for line in split_path.read_text().splitlines()
        if line.strip()
    }
    if not stems:
        sys.exit(f"Error: --split file '{split_path}' is empty.")
    return stems


def filter_entries(entries: list[dict], stem_filter: set[str] | None) -> list[dict]:
    """Apply stem filter if provided; otherwise return all entries."""
    if stem_filter is None:
        return entries
    filtered = [e for e in entries if e["stem"] in stem_filter]
    missing  = stem_filter - {e["stem"] for e in filtered}
    if missing:
        print(f"[WARN] {len(missing)} stem(s) from split file not found in dataset.json:")
        for s in sorted(missing):
            print(f"  {s}")
    return filtered


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_entry(
    entry:         dict,
    image_id:      int,
    dataset_dir:   Path,
    output_images: Path,
    fg_threshold:  int,
    overwrite:     bool,
) -> str:
    """
    Convert one dataset entry to SA-1B format.

    Returns "copied", "skipped", or "failed".
    """
    stem       = entry["stem"]
    image_src  = dataset_dir / entry["image_path"]
    mask_src   = dataset_dir / entry["mask_path"]

    # Determine destination paths
    image_dest = output_images / image_src.name
    json_dest  = image_dest.with_suffix(".json")

    # Skip check — only skip if both image and JSON already exist
    if image_dest.exists() and json_dest.exists() and not overwrite:
        return "skipped"

    try:
        # Build annotation (reads image dimensions + mask)
        annotation = build_annotation(image_id, image_src, mask_src, fg_threshold)

        # Copy image
        shutil.copy2(image_src, image_dest)

        # Write JSON annotation next to image
        json_dest.write_text(json.dumps(annotation))

        return "copied"

    except Exception as exc:
        print(f"  [FAIL]  '{stem}': {exc}")
        return "failed"


def run_conversion(
    entries:       list[dict],
    dataset_dir:   Path,
    output_dir:    Path,
    fg_threshold:  int,
    overwrite:     bool,
    dry_run:       bool,
) -> dict[str, int]:
    """
    Convert all entries and return counts.
    """
    output_images = output_dir / "images"

    if not dry_run:
        output_images.mkdir(parents=True, exist_ok=True)

    counts = {"copied": 0, "skipped": 0, "failed": 0}

    for image_id, entry in enumerate(entries):
        stem = entry["stem"]

        if dry_run:
            image_src = dataset_dir / entry["image_path"]
            mask_src  = dataset_dir / entry["mask_path"]
            img_ok    = "✓" if image_src.exists() else "✗ MISSING"
            mask_ok   = "✓" if mask_src.exists()  else "✗ MISSING"
            print(f"  {stem}")
            print(f"    image : {image_src}  [{img_ok}]")
            print(f"    mask  : {mask_src}  [{mask_ok}]")
            print(f"    json  : {(output_dir / 'images' / image_src.stem)}.json  [will be created]")
            counts["copied"] += 1
            continue

        result = convert_entry(
            entry, image_id, dataset_dir, output_images, fg_threshold, overwrite
        )
        counts[result] += 1

        if result == "copied":
            print(f"  [OK]    {stem}")
        elif result == "skipped":
            print(f"  [SKIP]  {stem} — already exists")

    return counts


# ---------------------------------------------------------------------------
# Split file handling
# ---------------------------------------------------------------------------

def copy_split_files(dataset_dir: Path, output_dir: Path) -> None:
    """Copy train.txt / val.txt from dataset dir to output dir if present."""
    for name in ("train.txt", "val.txt"):
        src = dataset_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            print(f"  Copied {name} → {output_dir / name}")
        else:
            print(f"  [INFO] {name} not found in dataset dir, skipping.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a build_dataset.py dataset to SA-1B format for SAM2 training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, metavar="PATH",
        help="Dataset folder containing dataset.json, images/, and masks/.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, metavar="PATH",
        help="Destination folder for SA-1B formatted output.",
    )
    parser.add_argument(
        "--split", type=Path, metavar="PATH", default=None,
        help=(
            "Optional .txt file listing stems to process (one per line). "
            "If omitted, all entries in dataset.json are converted."
        ),
    )
    parser.add_argument(
        "--fg-threshold", type=int, default=0, metavar="INT",
        help=(
            "Pixel value threshold: pixels > this value are foreground. "
            "Default: 0 (any non-zero pixel is foreground)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files. Default: skip existing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be converted without writing anything.",
    )
    parser.add_argument(
        "--no-copy-splits", action="store_true",
        help="Do not copy train.txt / val.txt to the output folder.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' is not a directory or does not exist.")
    if args.output.exists() and not args.output.is_dir():
        sys.exit(f"Error: --output '{args.output}' exists but is not a directory.")
    if not 0 <= args.fg_threshold <= 254:
        sys.exit(f"Error: --fg-threshold must be between 0 and 254, got {args.fg_threshold}.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Dataset       : {args.dataset}")
    print(f"Output        : {args.output}")
    print(f"Split filter  : {args.split or 'none (all entries)'}")
    print(f"FG threshold  : > {args.fg_threshold}")
    print(f"Overwrite     : {args.overwrite}")
    print(f"Dry run       : {args.dry_run}")
    print()

    entries     = load_manifest(args.dataset)
    stem_filter = load_split_filter(args.split)
    entries     = filter_entries(entries, stem_filter)

    if not entries:
        sys.exit("No entries to convert after applying split filter.")

    print(f"Entries to convert: {len(entries)}\n")

    if args.dry_run:
        print("Dry run — no files will be written:\n")

    counts = run_conversion(
        entries=entries,
        dataset_dir=args.dataset,
        output_dir=args.output,
        fg_threshold=args.fg_threshold,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if not args.dry_run and not args.no_copy_splits:
        print("\nCopying split files...")
        copy_split_files(args.dataset, args.output)

    print(
        f"\nDone.  Copied: {counts['copied']}  "
        f"Skipped: {counts['skipped']}  "
        f"Failed: {counts['failed']}"
    )

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()