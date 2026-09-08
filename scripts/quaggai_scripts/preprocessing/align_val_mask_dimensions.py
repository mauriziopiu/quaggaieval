"""
align_val_mask_dimensions.py

Fixes image/mask dimension mismatches in a dataset split (typically the
validation split) by resizing mismatched masks to match their paired
image's dimensions, using nearest-neighbor interpolation to preserve
binary mask values.

Background: build_dataset.py matches image/mask pairs by filename stem
only and never validates that the two files share the same dimensions.
For a meaningful fraction of this dataset they don't (see
mask_alignment_report.json produced by this script for specifics),
which causes downstream evaluation scripts (run_inference.py,
visualize_comparison.py, evaluate.py) to crash with a numpy broadcast
error. This script fixes the masks so those scripts can run against the
full split without modification.

Safety
------
- Dry-run by default. Nothing is written until --apply is passed.
- Every mask that gets resized has its original backed up first
  (unmodified) to --backup-dir, so the operation is fully reversible.
- Only resizes when image and mask aspect ratios match within
  --aspect-tolerance. Pairs whose aspect ratios differ by more than that
  are almost certainly a genuine mismatched pairing (wrong file, not a
  scale/rounding artifact) and are left untouched, flagged for manual
  review in the report instead.
- Already-matching pairs are left untouched. Re-running after a partial
  --apply is safe (idempotent).

Output
------
    <backup-dir>/<stem>_mask.png     original mask, backed up before overwrite
    <mask_path>                      overwritten in place with the resized mask
    <report>                         mask_alignment_report.json (see below)

Report JSON structure
----------------------
{
    "dataset_dir": "<path>",
    "split_file": "<path or null>",
    "aspect_tolerance": <float>,
    "applied": <bool>,
    "summary": {"already_matching": N, "resized": N, "flagged_manual_review": N, "missing_files": N},
    "resized": [
        {"stem": "...", "image_size": [w, h], "old_mask_size": [w, h], "new_mask_size": [w, h], "backup_path": "..."},
        ...
    ],
    "flagged_manual_review": [
        {"stem": "...", "image_size": [w, h], "mask_size": [w, h],
         "image_aspect": <float>, "mask_aspect": <float>, "relative_diff": <float>},
        ...
    ],
    "missing_files": [
        {"stem": "...", "reason": "..."},
        ...
    ]
}

Usage
-----
    # Dry run — see what would change, nothing is written
    python align_val_mask_dimensions.py \\
        --dataset ./dataset \\
        --split   ./dataset/val.txt

    # Apply the fix
    python align_val_mask_dimensions.py \\
        --dataset ./dataset \\
        --split   ./dataset/val.txt \\
        --apply

Options
-------
    --dataset          PATH    Dataset folder with dataset.json, images/, masks/ (required)
    --split            PATH    Optional .txt file of stems to process.
                                If omitted, ALL entries in dataset.json are processed —
                                pass the val/test split explicitly to scope this correctly.
    --aspect-tolerance  FLOAT  Max relative difference in image/mask aspect ratio to
                                treat a mismatch as safe to auto-resize. Default: 0.02 (2%).
    --backup-dir       PATH    Where to back up originals before overwriting.
                                Default: <dataset_dir>/masks_backup_pre_resize
    --report           PATH    Where to write the JSON report.
                                Default: <dataset_dir>/mask_alignment_report.json
    --apply                    Actually write changes. Default: dry run (report only).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_manifest(dataset_dir: Path) -> list[dict]:
    p = dataset_dir / "dataset.json"
    if not p.exists():
        sys.exit(f"Error: dataset.json not found in '{dataset_dir}'.")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    if not entries:
        sys.exit("Error: dataset.json contains no entries.")
    return entries


def load_split_filter(split_path: Path | None) -> set[str] | None:
    if split_path is None:
        return None
    if not split_path.exists():
        sys.exit(f"Error: --split file '{split_path}' does not exist.")
    stems = {l.strip() for l in split_path.read_text().splitlines() if l.strip()}
    if not stems:
        sys.exit(f"Error: --split file '{split_path}' is empty.")
    return stems


def filter_entries(entries: list[dict], stem_filter: set[str] | None) -> list[dict]:
    if stem_filter is None:
        return entries
    filtered = [e for e in entries if e["stem"] in stem_filter]
    missing  = stem_filter - {e["stem"] for e in filtered}
    if missing:
        print(f"[WARN] {len(missing)} stems from split file not in dataset.json:")
        for s in sorted(missing):
            print(f"  {s}")
    return filtered


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_entries(
    entries:           list[dict],
    dataset_dir:        Path,
    backup_dir:         Path,
    aspect_tolerance:   float,
    apply:              bool,
) -> dict:
    report = {
        "already_matching":      [],
        "resized":               [],
        "flagged_manual_review": [],
        "missing_files":         [],
    }

    for entry in entries:
        stem      = entry["stem"]
        img_path  = dataset_dir / entry["image_path"]
        mask_path = dataset_dir / entry["mask_path"]

        if not img_path.exists() or not mask_path.exists():
            report["missing_files"].append({
                "stem":   stem,
                "reason": f"image exists={img_path.exists()}, mask exists={mask_path.exists()}",
            })
            print(f"  [MISSING] {stem}")
            continue

        try:
            with Image.open(img_path) as img:
                img_size = img.size  # (w, h)
            with Image.open(mask_path) as mask:
                mask_size = mask.size
        except Exception as exc:
            report["missing_files"].append({"stem": stem, "reason": f"failed to open: {exc}"})
            print(f"  [FAIL]    {stem}: {exc}")
            continue

        if img_size == mask_size:
            report["already_matching"].append({"stem": stem, "size": list(img_size)})
            continue

        img_aspect  = img_size[0]  / img_size[1]
        mask_aspect = mask_size[0] / mask_size[1]
        rel_diff    = abs(img_aspect - mask_aspect) / img_aspect

        if rel_diff > aspect_tolerance:
            report["flagged_manual_review"].append({
                "stem":          stem,
                "image_size":    list(img_size),
                "mask_size":     list(mask_size),
                "image_aspect":  round(img_aspect, 4),
                "mask_aspect":   round(mask_aspect, 4),
                "relative_diff": round(rel_diff, 4),
            })
            print(f"  [FLAG]    {stem}  image={img_size} mask={mask_size}  "
                  f"aspect diff={rel_diff:.1%} — needs manual review, not auto-resized")
            continue

        backup_path = backup_dir / mask_path.name
        action = "RESIZE" if apply else "WOULD-RESIZE"
        print(f"  [{action}] {stem}  mask {mask_size} → {img_size}")

        if apply:
            backup_dir.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(mask_path, backup_path)
            with Image.open(mask_path) as mask:
                resized = mask.convert("L").resize(img_size, resample=RESAMPLE_NEAREST)
                resized.save(mask_path)

        report["resized"].append({
            "stem":          stem,
            "image_size":    list(img_size),
            "old_mask_size": list(mask_size),
            "new_mask_size": list(img_size),
            "backup_path":   str(backup_path),
        })

    return report


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    report_path:        Path,
    dataset_dir:         Path,
    split_path:          Path | None,
    aspect_tolerance:    float,
    apply:               bool,
    report:              dict,
) -> Path:
    full_report = {
        "dataset_dir":      str(dataset_dir.resolve()),
        "split_file":       str(split_path.resolve()) if split_path else None,
        "aspect_tolerance": aspect_tolerance,
        "applied":          apply,
        "summary": {k: len(v) for k, v in report.items()},
        **report,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize mismatched validation masks to match their image's dimensions (nearest-neighbor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",          required=True, type=Path,  metavar="PATH")
    parser.add_argument("--split",            default=None,  type=Path,  metavar="PATH")
    parser.add_argument("--aspect-tolerance", default=0.02,  type=float, metavar="FLOAT")
    parser.add_argument("--backup-dir",       default=None,  type=Path,  metavar="PATH")
    parser.add_argument("--report",           default=None,  type=Path,  metavar="PATH")
    parser.add_argument("--apply",            action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' does not exist.")
    if not (0.0 < args.aspect_tolerance < 1.0):
        sys.exit("Error: --aspect-tolerance must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    backup_dir = args.backup_dir or (args.dataset / "masks_backup_pre_resize")
    report_path = args.report or (args.dataset / "mask_alignment_report.json")

    print(f"Dataset          : {args.dataset}")
    print(f"Split            : {args.split if args.split else '(none — processing ALL entries)'}")
    print(f"Aspect tolerance : {args.aspect_tolerance:.1%}")
    print(f"Backup dir       : {backup_dir}")
    print(f"Report           : {report_path}")
    print(f"Mode             : {'APPLY (writing changes)' if args.apply else 'DRY RUN (no changes written)'}")
    print()

    entries     = load_manifest(args.dataset)
    stem_filter = load_split_filter(args.split)
    entries     = filter_entries(entries, stem_filter)

    if not entries:
        sys.exit("No entries to process.")

    print(f"Entries to process: {len(entries)}\n")

    report = process_entries(
        entries=entries,
        dataset_dir=args.dataset,
        backup_dir=backup_dir,
        aspect_tolerance=args.aspect_tolerance,
        apply=args.apply,
    )

    write_report(report_path, args.dataset, args.split, args.aspect_tolerance, args.apply, report)

    print(f"\nSummary:")
    print(f"  Already matching       : {len(report['already_matching'])}")
    print(f"  Resized{'' if args.apply else ' (would be)':<12}       : {len(report['resized'])}")
    print(f"  Flagged (manual review): {len(report['flagged_manual_review'])}")
    print(f"  Missing/failed         : {len(report['missing_files'])}")
    print(f"\nReport written to: {report_path}")

    if not args.apply and report["resized"]:
        print("\nThis was a dry run — re-run with --apply to actually resize the masks.")

    if report["flagged_manual_review"]:
        print(f"\n[WARN] {len(report['flagged_manual_review'])} pair(s) flagged for manual review "
              f"— aspect ratio mismatch too large to be a safe auto-resize:")
        for f in report["flagged_manual_review"]:
            print(f"    {f['stem']}: image={f['image_size']} mask={f['mask_size']}")


if __name__ == "__main__":
    main()
