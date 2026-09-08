"""
filter_mismatched_pairs.py

Scans a dataset for image/mask pairs whose pixel dimensions don't match
(a common cause of failed overlay/inference runs), writes a new dataset
folder containing only the correctly-matched pairs, and rewrites
dataset.json + train.txt/val.txt for the surviving pairs.

Dropping mismatched pairs shifts the train/val ratio away from 80/20, so
the split is rebalanced afterward: each surviving stem first keeps its
original train/val assignment (to minimize churn / avoid introducing
train-val leakage relative to prior runs), then the minimum number of
stems are moved between splits — chosen at random (seeded) — to restore
an 80/20 ratio. Stems absent from both original split files are treated
as unassigned and distributed toward whichever split needs more first.

Output structure
-----------------
<output>/
    images/                    copied image files, surviving pairs only
    masks/                     copied mask files, surviving pairs only
    dataset.json                only surviving entries
    train.txt                   rebalanced 80% split
    val.txt                     rebalanced 20% split
    filter_report.json          excluded pairs (with mismatched sizes) +
                                 split-rebalance details

Usage
-----
    python filter_mismatched_pairs.py \\
        --dataset ./dataset \\
        --output  ./dataset_filtered

Options
-------
    --dataset    PATH    Source dataset folder with dataset.json, images/,
                          masks/, train.txt, val.txt (required)
    --output     PATH    Folder to write the filtered dataset to (required)
    --seed       INT     Random seed for split rebalancing. Default: 42.
    --overwrite          Overwrite existing files in --output. Default: skip
                          copying files that already exist; dataset.json /
                          train.txt / val.txt / filter_report.json are
                          always (re)written.
    --dry-run            Report what would happen without writing anything.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_manifest(dataset_dir: Path) -> dict:
    p = dataset_dir / "dataset.json"
    if not p.exists():
        sys.exit(f"Error: dataset.json not found in '{dataset_dir}'.")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("entries"):
        sys.exit("Error: dataset.json contains no entries.")
    return data


def load_split(split_path: Path) -> list[str]:
    if not split_path.exists():
        sys.exit(f"Error: '{split_path}' does not exist.")
    return [l.strip() for l in split_path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Step 1: find size-mismatched pairs
# ---------------------------------------------------------------------------

def check_pairs(entries: list[dict], dataset_dir: Path) -> tuple[list[dict], list[dict]]:
    """Returns (matching_entries, mismatched_records)."""
    matching:    list[dict] = []
    mismatched:  list[dict] = []

    for entry in entries:
        img_path  = dataset_dir / entry["image_path"]
        mask_path = dataset_dir / entry["mask_path"]
        try:
            with Image.open(img_path) as im:
                image_size = im.size  # (w, h), header-only read
            with Image.open(mask_path) as im:
                mask_size = im.size
        except Exception as exc:
            mismatched.append({"stem": entry["stem"], "reason": f"failed_to_open: {exc}"})
            continue

        if image_size != mask_size:
            mismatched.append({
                "stem":       entry["stem"],
                "reason":     "size_mismatch",
                "image_size": list(image_size),
                "mask_size":  list(mask_size),
            })
        else:
            matching.append(entry)

    return matching, mismatched


# ---------------------------------------------------------------------------
# Step 2: copy surviving files
# ---------------------------------------------------------------------------

def copy_pairs(
    entries:     list[dict],
    dataset_dir: Path,
    output_dir:  Path,
    overwrite:   bool,
) -> None:
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    for entry in entries:
        for key in ("image_path", "mask_path"):
            src = dataset_dir / entry[key]
            dst = output_dir / entry[key]
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Step 3: rebalance the train/val split for surviving stems
# ---------------------------------------------------------------------------

def rebalance_split(
    surviving_stems: list[str],
    orig_train:      list[str],
    orig_val:        list[str],
    rng,
) -> tuple[list[str], list[str], dict]:
    surviving_set = set(surviving_stems)
    train_set = {s for s in orig_train if s in surviving_set}
    val_set   = {s for s in orig_val   if s in surviving_set}
    unassigned = sorted(surviving_set - train_set - val_set)

    total = len(surviving_stems)
    target_train = round(0.8 * total)
    target_val   = total - target_train

    # First place stems that had no original split assignment, favoring
    # whichever split is further below its target.
    rng.shuffle(unassigned)
    for stem in unassigned:
        if len(train_set) - target_train <= len(val_set) - target_val:
            train_set.add(stem)
        else:
            val_set.add(stem)

    moved_train_to_val: list[str] = []
    moved_val_to_train: list[str] = []

    if len(train_set) > target_train:
        surplus = len(train_set) - target_train
        candidates = sorted(train_set)
        rng.shuffle(candidates)
        moved_train_to_val = candidates[:surplus]
        train_set -= set(moved_train_to_val)
        val_set |= set(moved_train_to_val)
    elif len(train_set) < target_train:
        deficit = target_train - len(train_set)
        candidates = sorted(val_set)
        rng.shuffle(candidates)
        moved_val_to_train = candidates[:deficit]
        val_set -= set(moved_val_to_train)
        train_set |= set(moved_val_to_train)

    report = {
        "target_train":        target_train,
        "target_val":          target_val,
        "final_train":         len(train_set),
        "final_val":           len(val_set),
        "unassigned_in_orig":  unassigned,
        "moved_train_to_val":  moved_train_to_val,
        "moved_val_to_train":  moved_val_to_train,
    }
    return sorted(train_set), sorted(val_set), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop size-mismatched image/mask pairs and rewrite dataset.json + train/val splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output",  required=True, type=Path, metavar="PATH")
    parser.add_argument("--seed",    default=42,     type=int,  metavar="INT")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' does not exist.")


def main() -> None:
    import random

    args = parse_args()
    validate_args(args)

    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.output}")
    print(f"Seed    : {args.seed}")
    print()

    manifest = load_manifest(args.dataset)
    entries  = manifest["entries"]
    orig_train = load_split(args.dataset / "train.txt")
    orig_val   = load_split(args.dataset / "val.txt")

    print(f"Checking {len(entries)} pairs for size mismatches...")
    matching, mismatched = check_pairs(entries, args.dataset)
    print(f"  matching   : {len(matching)}")
    print(f"  mismatched : {len(mismatched)}")
    print()

    surviving_stems = [e["stem"] for e in matching]
    rng = random.Random(args.seed)
    new_train, new_val, split_report = rebalance_split(surviving_stems, orig_train, orig_val, rng)

    print(f"Split rebalanced: train={len(new_train)}  val={len(new_val)}  "
          f"(target train={split_report['target_train']}, target val={split_report['target_val']})")
    if split_report["unassigned_in_orig"]:
        print(f"  {len(split_report['unassigned_in_orig'])} surviving stem(s) had no original split assignment.")
    if split_report["moved_train_to_val"] or split_report["moved_val_to_train"]:
        print(f"  moved {len(split_report['moved_train_to_val'])} train→val, "
              f"{len(split_report['moved_val_to_train'])} val→train to restore ratio.")
    print()

    if args.dry_run:
        print("Dry run — no files written.")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(matching)} surviving pairs to {args.output}...")
    copy_pairs(matching, args.dataset, args.output, args.overwrite)

    new_manifest = {"total_pairs": len(matching), "entries": matching}
    with open(args.output / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2)

    (args.output / "train.txt").write_text("\n".join(new_train) + "\n")
    (args.output / "val.txt").write_text("\n".join(new_val) + "\n")

    report = {
        "source_dataset":     str(args.dataset.resolve()),
        "total_source_pairs": len(entries),
        "matching_pairs":     len(matching),
        "mismatched_pairs":   len(mismatched),
        "mismatched":         mismatched,
        "split_rebalance":    split_report,
    }
    with open(args.output / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone. Wrote filtered dataset ({len(matching)} pairs) to: {args.output}")
    print(f"Report: {args.output / 'filter_report.json'}")


if __name__ == "__main__":
    main()
