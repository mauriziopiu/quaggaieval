# rename_to_sa1b.py
"""
Renames images and their paired JSON annotations in an SA-1B formatted
directory to the numeric convention expected by SA1BRawDataset.

Before: IMG_S2_hit.jpg + IMG_S2_hit.json
After:  sa_000001.jpg  + sa_000001.json

Also rewrites train.txt / val.txt if present in the same directory,
replacing old stems with new ones.

Usage:
    python rename_to_sa1b.py --dir ./sa1b_train/images
    python rename_to_sa1b.py --dir ./sa1b_val/images
"""

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path,
                        help="Directory containing image + JSON pairs to rename.")
    parser.add_argument("--prefix", default="sa_", type=str,
                        help="Filename prefix. Default: 'sa_'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print renames without executing them.")
    return parser.parse_args()


def run(img_dir: Path, prefix: str = "sa_", dry_run: bool = False) -> dict[str, int]:
    """Rename image+JSON pairs in *img_dir* to numeric SA-1B convention.

    Also rewrites train.txt / val.txt in the parent directory if present.

    Returns a counts dict with key ``renamed`` (or ``would_rename`` in dry-run).
    """
    if not img_dir.is_dir():
        raise ValueError(f"'{img_dir}' is not a directory.")

    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No image files found in '{img_dir}'.")

    # Build rename mapping: old_stem → new_stem
    mapping: dict[str, str] = {}
    for idx, img_path in enumerate(images):
        mapping[img_path.stem] = f"{prefix}{idx:06d}"

    # Execute renames
    for old_stem, new_stem in mapping.items():
        matches = list(img_dir.glob(f"{old_stem}.*"))
        for src in matches:
            if src.suffix.lower() not in IMAGE_EXTENSIONS and src.suffix != ".json":
                continue
            dst = src.with_name(f"{new_stem}{src.suffix.lower()}")
            if dry_run:
                print(f"  {src.name}  →  {dst.name}")
            else:
                shutil.move(str(src), str(dst))

    # Update train.txt / val.txt one level up (dataset root)
    dataset_root = img_dir.parent
    for txt_name in ("train.txt", "val.txt"):
        txt_path = dataset_root / txt_name
        if not txt_path.exists():
            continue
        lines = txt_path.read_text().splitlines()
        new_lines = [mapping.get(line.strip(), line.strip()) for line in lines if line.strip()]
        if dry_run:
            print(f"\n  Would update {txt_path} ({len(new_lines)} stems)")
        else:
            txt_path.write_text("\n".join(new_lines))
            print(f"Updated {txt_path}")

    count_key = "would_rename" if dry_run else "renamed"
    return {count_key: len(mapping)}


def main():
    args = parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"Error: '{args.dir}' is not a directory.")

    print(f"Found images in: {args.dir}\n")
    counts = run(args.dir, prefix=args.prefix, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete — no files were modified.")
    else:
        renamed = counts.get("renamed", 0)
        print(f"\nDone. Renamed {renamed} image+JSON pair(s).")


if __name__ == "__main__":
    main()