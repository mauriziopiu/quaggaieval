# png_to_jpg.py
"""
Convert PNG images to JPEG format.

Usage:
    python png_to_jpg.py --input ./raw/cropped --output ./work/jpgs
    python png_to_jpg.py --input ./raw/cropped --output ./work/jpgs --quality 85
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


def process_batch(
    input_dir: Path,
    output_dir: Path,
    quality: int = 95,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert all PNG files in *input_dir* to JPEG and write to *output_dir*.

    Returns a counts dict with keys ``processed``, ``skipped``, ``failed``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(input_dir.glob("*.png"))
    counts = {"processed": 0, "skipped": 0, "failed": 0}

    for png in pngs:
        dest = output_dir / png.with_suffix(".jpg").name
        if dest.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            img = Image.open(png).convert("RGB")
            img.save(dest, "JPEG", quality=quality)
            counts["processed"] += 1
        except Exception as exc:
            print(f"  [FAIL] {png.name}: {exc}", file=sys.stderr)
            counts["failed"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PNG images to JPEG format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, type=Path, metavar="PATH",
        help="Folder containing source PNG files.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, metavar="PATH",
        help="Folder to write JPEG files to.",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality (1–95). Default: 95.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing JPEG files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_dir():
        sys.exit(f"Error: --input path '{args.input}' is not a directory or does not exist.")
    if not (1 <= args.quality <= 95):
        sys.exit(f"Error: --quality must be between 1 and 95, got {args.quality}.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Input folder  : {args.input}")
    print(f"Output folder : {args.output}")
    print(f"Quality       : {args.quality}")
    print(f"Overwrite     : {args.overwrite}")

    counts = process_batch(
        input_dir=args.input,
        output_dir=args.output,
        quality=args.quality,
        overwrite=args.overwrite,
    )

    print(
        f"\nDone. Processed: {counts['processed']}  "
        f"Skipped: {counts['skipped']}  "
        f"Failed: {counts['failed']}"
    )

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
