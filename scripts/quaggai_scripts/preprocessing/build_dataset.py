"""
build_dataset.py

Scans an image folder and a mask folder, matches pairs by base stem,
and consolidates them into a structured dataset directory with a manifest.

Matching rules
--------------
Images : extensions .jpg .jpeg .png .tiff .tif are considered.
         Known image suffixes stripped to recover base stem: _cropped
         If none match, the full filename stem is used as-is.

Masks  : only .png files are considered.
         Known mask suffixes stripped to recover base stem (longest first):
         _area_mask, _mask

A pair is matched when both sides resolve to the same base stem.

Output structure
----------------
<output_dir>/
    images/
        <stem>.<ext>        # copied from source
        ...
    masks/
        <stem>_mask.png     # copied from source, renamed to canonical form
        ...
    dataset.json            # manifest with one entry per matched pair

dataset.json schema
-------------------
{
    "total_pairs": <int>,
    "entries": [
        {
            "stem":         "<base stem>",
            "image_path":   "images/<stem>.<ext>",    # relative to output_dir
            "mask_path":    "masks/<stem>_mask.png",  # relative to output_dir
            "image_source": "<absolute source path>",
            "mask_source":  "<absolute source path>"
        },
        ...
    ]
}

unmatched_report.json schema (written when --unmatched-report is set)
----------------------------------------------------------------------
{
    "images_without_mask": [
        {"stem": "...", "source_path": "..."},
        ...
    ],
    "masks_without_image": [
        {"stem": "...", "source_path": "..."},
        ...
    ],
    "stem_conflicts": {
        "images": ["Stem conflict 'foo': keeping foo.jpg, ignoring foo_cropped.jpg", ...],
        "masks":  ["Stem conflict 'bar': keeping bar_mask.png, ignoring bar_area_mask.png", ...]
    },
    "copy_failures": [
        {"stem": "...", "reason": "..."},
        ...
    ],
    "summary": {
        "images_without_mask":   <int>,
        "masks_without_image":   <int>,
        "stem_conflicts_images": <int>,
        "stem_conflicts_masks":  <int>,
        "copy_failures":         <int>
    }
}

Usage
-----
    python build_dataset.py --images <dir> --masks <dir> --output <dir> [options]

Options
-------
    --images            PATH   Folder containing source images (required)
    --masks             PATH   Folder containing source masks  (required)
    --output            PATH   Destination dataset folder      (required)
    --unmatched-report  PATH   Write a JSON report of all unmatched/failed
                               files to this path. Optional; issues are only
                               printed to console if omitted.
    --overwrite                Overwrite existing output files. Default: skip.
    --dry-run                  Print matched pairs and exit without writing.

Examples
--------
    python build_dataset.py --images ./raw_images --masks ./raw_masks --output ./dataset

    python build_dataset.py --images ./raw_images --masks ./raw_masks --output ./dataset \\
        --unmatched-report ./unmatched_report.json

    python build_dataset.py --images ./raw_images --masks ./raw_masks --output ./dataset \\
        --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants — extend these lists if new naming conventions are introduced
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif"})

# Suffixes stripped from image stems to recover the base stem.
IMAGE_SUFFIXES: tuple[str, ...] = ("_cropped",)

# Suffixes stripped from mask stems — longest first to avoid partial matches.
# "_area_mask" must be tried before "_mask" so that
# "img001_area_mask" → "img001", not "img001_area".
MASK_SUFFIXES: tuple[str, ...] = ("_area_mask", "_mask")


# ---------------------------------------------------------------------------
# Stem extraction
# ---------------------------------------------------------------------------

def extract_image_stem(path: Path) -> str:
    """
    Return the base stem for an image file.

    img001_cropped.jpg  -> "img001"
    img001.jpg          -> "img001"
    """
    stem = path.stem
    for suffix in IMAGE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def extract_mask_stem(path: Path) -> str:
    """
    Return the base stem for a mask file.

    img001_area_mask.png -> "img001"
    img001_mask.png      -> "img001"
    """
    stem = path.stem
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_images(image_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """
    Build a base-stem -> Path mapping for all recognised image files.

    If two image files resolve to the same base stem (e.g. img001.jpg and
    img001_cropped.jpg both map to "img001"), the conflict is recorded and
    only the first encountered alphabetically is kept.

    Returns:
        index     : stem -> source Path
        conflicts : human-readable conflict description strings
    """
    index: dict[str, Path] = {}
    conflicts: list[str] = []

    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = extract_image_stem(path)

        if stem in index:
            conflicts.append(
                f"Stem conflict '{stem}': keeping {index[stem].name}, "
                f"ignoring {path.name}"
            )
        else:
            index[stem] = path

    if conflicts:
        print(f"[WARN] {len(conflicts)} image stem conflict(s) detected:")
        for msg in conflicts:
            print(f"  {msg}")

    return index, conflicts


def index_masks(mask_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """
    Build a base-stem -> Path mapping for all .png mask files.

    Returns:
        index     : stem -> source Path
        conflicts : human-readable conflict description strings
    """
    index: dict[str, Path] = {}
    conflicts: list[str] = []

    for path in sorted(mask_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue

        stem = extract_mask_stem(path)

        if stem in index:
            conflicts.append(
                f"Stem conflict '{stem}': keeping {index[stem].name}, "
                f"ignoring {path.name}"
            )
        else:
            index[stem] = path

    if conflicts:
        print(f"[WARN] {len(conflicts)} mask stem conflict(s) detected:")
        for msg in conflicts:
            print(f"  {msg}")

    return index, conflicts


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------

def match_pairs(
    image_index: dict[str, Path],
    mask_index: dict[str, Path],
) -> tuple[list[tuple[str, Path, Path]], list[str], list[str]]:
    """
    Return:
        matched          : list of (base_stem, image_path, mask_path)
        unmatched_images : stems present in images but not masks
        unmatched_masks  : stems present in masks but not images
    """
    image_stems = set(image_index)
    mask_stems  = set(mask_index)

    matched_stems    = sorted(image_stems & mask_stems)
    unmatched_images = sorted(image_stems - mask_stems)
    unmatched_masks  = sorted(mask_stems  - image_stems)

    matched = [
        (stem, image_index[stem], mask_index[stem])
        for stem in matched_stems
    ]
    return matched, unmatched_images, unmatched_masks


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def canonical_image_name(stem: str, source: Path) -> str:
    """Preserve original extension; stem is already the base stem."""
    return f"{stem}{source.suffix.lower()}"


def canonical_mask_name(stem: str) -> str:
    return f"{stem}_mask.png"


def build_dataset(
    matched: list[tuple[str, Path, Path]],
    output_dir: Path,
    overwrite: bool,
) -> tuple[dict, list[dict], dict[str, int]]:
    """
    Copy matched pairs into output_dir/images/ and output_dir/masks/.

    Returns:
        manifest      : dict ready to serialise as dataset.json
        copy_failures : list of {"stem": ..., "reason": ...} dicts
        counts        : {"copied": n, "skipped": n, "failed": n}
    """
    images_dir = output_dir / "images"
    masks_dir  = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    entries:       list[dict] = []
    copy_failures: list[dict] = []
    counts: dict[str, int] = {"copied": 0, "skipped": 0, "failed": 0}

    for stem, image_src, mask_src in matched:
        image_dest_name = canonical_image_name(stem, image_src)
        mask_dest_name  = canonical_mask_name(stem)

        image_dest = images_dir / image_dest_name
        mask_dest  = masks_dir  / mask_dest_name

        image_rel = Path("images") / image_dest_name
        mask_rel  = Path("masks")  / mask_dest_name

        # Skip without overwriting unless --overwrite is set
        already_exists = image_dest.exists() or mask_dest.exists()
        if already_exists and not overwrite:
            print(f"  [SKIP]  '{stem}' — output already exists")
            counts["skipped"] += 1
            # Still include in manifest so the JSON reflects the full dataset
            entries.append(_make_entry(stem, image_rel, mask_rel, image_src, mask_src))
            continue

        try:
            shutil.copy2(image_src, image_dest)
            shutil.copy2(mask_src, mask_dest)
            print(f"  [OK]    {stem}  ({image_src.name} + {mask_src.name})")
            counts["copied"] += 1
            entries.append(_make_entry(stem, image_rel, mask_rel, image_src, mask_src))
        except Exception as exc:
            reason = str(exc)
            print(f"  [FAIL]  '{stem}': {reason}")
            counts["failed"] += 1
            copy_failures.append({"stem": stem, "reason": reason})

    manifest = {"total_pairs": len(entries), "entries": entries}
    return manifest, copy_failures, counts


def _make_entry(
    stem: str,
    image_rel: Path,
    mask_rel: Path,
    image_src: Path,
    mask_src: Path,
) -> dict:
    return {
        "stem":         stem,
        "image_path":   str(image_rel),
        "mask_path":    str(mask_rel),
        "image_source": str(image_src.resolve()),
        "mask_source":  str(mask_src.resolve()),
    }


def write_manifest(manifest: dict, output_dir: Path) -> Path:
    dest = output_dir / "dataset.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return dest


# ---------------------------------------------------------------------------
# Unmatched report
# ---------------------------------------------------------------------------

def build_unmatched_report(
    unmatched_images: list[str],
    unmatched_masks:  list[str],
    image_index:      dict[str, Path],
    mask_index:       dict[str, Path],
    image_conflicts:  list[str],
    mask_conflicts:   list[str],
    copy_failures:    list[dict],
) -> dict:
    """
    Assemble the full problem report as a serialisable dict.
    All source paths are stored as absolute strings for easy follow-up.
    """
    return {
        "images_without_mask": [
            {"stem": s, "source_path": str(image_index[s].resolve())}
            for s in unmatched_images
        ],
        "masks_without_image": [
            {"stem": s, "source_path": str(mask_index[s].resolve())}
            for s in unmatched_masks
        ],
        "stem_conflicts": {
            "images": image_conflicts,
            "masks":  mask_conflicts,
        },
        "copy_failures": copy_failures,
        "summary": {
            "images_without_mask":   len(unmatched_images),
            "masks_without_image":   len(unmatched_masks),
            "stem_conflicts_images": len(image_conflicts),
            "stem_conflicts_masks":  len(mask_conflicts),
            "copy_failures":         len(copy_failures),
        },
    }


def write_unmatched_report(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Console reporting helpers
# ---------------------------------------------------------------------------

def print_unmatched(unmatched_images: list[str], unmatched_masks: list[str]) -> None:
    if unmatched_images:
        print(f"\n[WARN] {len(unmatched_images)} image(s) with no matching mask:")
        for s in unmatched_images:
            print(f"  {s}")
    if unmatched_masks:
        print(f"\n[WARN] {len(unmatched_masks)} mask(s) with no matching image:")
        for s in unmatched_masks:
            print(f"  {s}")


def print_dry_run(matched: list[tuple[str, Path, Path]]) -> None:
    print(f"\nDry run -- {len(matched)} matched pair(s):\n")
    for stem, img, mask in matched:
        print(f"  {stem}")
        print(f"    image : {img}")
        print(f"    mask  : {mask}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match image/mask pairs by stem and build a structured dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--images", required=True, type=Path, metavar="PATH",
        help="Folder containing source images.",
    )
    parser.add_argument(
        "--masks", required=True, type=Path, metavar="PATH",
        help="Folder containing source masks (.png).",
    )
    parser.add_argument(
        "--output", required=True, type=Path, metavar="PATH",
        help="Destination dataset folder.",
    )
    parser.add_argument(
        "--unmatched-report", type=Path, metavar="PATH", default=None,
        help=(
            "Write a JSON report of unmatched files, stem conflicts, and copy "
            "failures to this path. If omitted, issues are only printed to console."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files in output. Default: skip.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print matched pairs and exit without writing anything.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for attr, label in [("images", "--images"), ("masks", "--masks")]:
        path = getattr(args, attr)
        if not path.is_dir():
            sys.exit(f"Error: {label} path '{path}' is not a directory or does not exist.")

    if args.unmatched_report is not None and args.unmatched_report.is_dir():
        sys.exit(
            f"Error: --unmatched-report '{args.unmatched_report}' is a directory. "
            "Provide a file path (e.g. ./unmatched_report.json)."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Images folder    : {args.images}")
    print(f"Masks folder     : {args.masks}")
    print(f"Output folder    : {args.output}")
    if args.unmatched_report:
        print(f"Unmatched report : {args.unmatched_report}")
    print()

    # Index both sides — retain conflict lists for the report
    image_index, image_conflicts = index_images(args.images)
    mask_index,  mask_conflicts  = index_masks(args.masks)

    print(f"\nFound {len(image_index)} image(s), {len(mask_index)} mask(s).")

    # Match
    matched, unmatched_images, unmatched_masks = match_pairs(image_index, mask_index)
    print(f"Matched pairs    : {len(matched)}")

    print_unmatched(unmatched_images, unmatched_masks)

    if not matched:
        sys.exit("\nNo matched pairs found. Check folder contents and naming conventions.")

    # Dry run — report and exit; no files or report written
    if args.dry_run:
        print_dry_run(matched)
        return

    # Build dataset
    print(f"\nCopying files to '{args.output}'...\n")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, copy_failures, counts = build_dataset(matched, args.output, args.overwrite)

    manifest_path = write_manifest(manifest, args.output)
    print(f"\nManifest written to : {manifest_path}")

    # Write unmatched report if requested
    if args.unmatched_report is not None:
        report = build_unmatched_report(
            unmatched_images=unmatched_images,
            unmatched_masks=unmatched_masks,
            image_index=image_index,
            mask_index=mask_index,
            image_conflicts=image_conflicts,
            mask_conflicts=mask_conflicts,
            copy_failures=copy_failures,
        )
        write_unmatched_report(report, args.unmatched_report)
        print(f"Unmatched report    : {args.unmatched_report}")

        s = report["summary"]
        if any(s.values()):
            print(
                f"\nReport summary  --  "
                f"images without mask: {s['images_without_mask']}  |  "
                f"masks without image: {s['masks_without_image']}  |  "
                f"stem conflicts (img/mask): "
                f"{s['stem_conflicts_images']}/{s['stem_conflicts_masks']}  |  "
                f"copy failures: {s['copy_failures']}"
            )

    print(
        f"\nDone.  Copied: {counts['copied']}  "
        f"Skipped: {counts['skipped']}  "
        f"Failed: {counts['failed']}"
    )

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()