"""
png_to_mask.py

Converts PNG images with transparency into binary segmentation masks.
Transparent pixels → black (background, 0)
Opaque pixels      → white (foreground, 255)

Usage:
    python png_to_mask.py --input <input_dir> --output <output_dir> [options]

Options:
    --input   PATH     Folder containing source PNG images (required)
    --output  PATH     Folder to write binary mask PNGs (required)
    --alpha-threshold  INT   Alpha value below which a pixel is considered
                             transparent. Default: 128. Range: 0–255.
                             Lower = more pixels treated as foreground.
    --suffix  STR      Suffix appended to output filenames before extension.
                       Default: "_mask". Set to "" to keep original names.
    --recursive        If set, search input folder recursively for PNGs.
    --overwrite        If set, allow overwriting existing files in output folder.
                       By default, existing files are skipped with a warning.

Examples:
    # Basic usage
    python png_to_mask.py --input ./raw_pngs --output ./masks

    # Custom alpha threshold and no suffix
    python png_to_mask.py --input ./raw_pngs --output ./masks --alpha-threshold 10 --suffix ""

    # Recursive search, overwrite existing outputs
    python png_to_mask.py --input ./dataset --output ./masks --recursive --overwrite
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def png_to_binary_mask(source_path: Path, alpha_threshold: int) -> np.ndarray:
    """
    Load a PNG and return a binary mask as a uint8 numpy array (H×W).

    Pixels with alpha >= alpha_threshold are foreground (255).
    Pixels with alpha <  alpha_threshold are background (0).

    Raises ValueError if the image has no alpha channel.
    """
    img = Image.open(source_path)

    if img.mode not in ("RGBA", "LA", "PA"):
        raise ValueError(
            f"{source_path.name}: image has no alpha channel (mode={img.mode}). "
            "Only RGBA / LA / PA images are supported."
        )

    # Convert everything to RGBA for uniform alpha extraction
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]

    mask = np.where(alpha >= alpha_threshold, np.uint8(255), np.uint8(0))
    return mask


def save_mask(mask: np.ndarray, dest_path: Path) -> None:
    """Save a uint8 numpy array as a single-channel PNG."""
    Image.fromarray(mask, mode="L").save(dest_path)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def collect_sources(input_dir: Path, recursive: bool) -> list[Path]:
    """Return sorted list of PNG paths under input_dir."""
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(input_dir.glob(pattern))


def build_dest_path(source: Path, input_dir: Path, output_dir: Path, suffix: str) -> Path:
    """
    Mirror the source directory structure under output_dir and apply suffix.

    Example:
        source    = input_dir/subdir/img001.png
        dest      = output_dir/subdir/img001_mask.png
    """
    relative = source.relative_to(input_dir)
    dest = output_dir / relative.parent / f"{source.stem}{suffix}.png"
    return dest


def process_batch(
    input_dir: Path,
    output_dir: Path,
    alpha_threshold: int,
    suffix: str,
    recursive: bool,
    overwrite: bool,
) -> dict[str, int]:
    """
    Process all PNGs in input_dir and write masks to output_dir.

    Returns a summary dict with counts: processed, skipped, failed.
    """
    sources = collect_sources(input_dir, recursive)

    if not sources:
        print(f"No PNG files found in '{input_dir}'.")
        return {"processed": 0, "skipped": 0, "failed": 0}

    counts = {"processed": 0, "skipped": 0, "failed": 0}

    for source in sources:
        dest = build_dest_path(source, input_dir, output_dir, suffix)

        # Skip without overwriting unless --overwrite is set
        if dest.exists() and not overwrite:
            print(f"  [SKIP]  {source.name} → {dest} already exists")
            counts["skipped"] += 1
            continue

        # Ensure destination directory exists (mirrors source subdirectory)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            mask = png_to_binary_mask(source, alpha_threshold)
            save_mask(mask, dest)
            foreground_pct = 100.0 * (mask > 0).sum() / mask.size
            print(f"  [OK]    {source.name} → {dest.name}  ({foreground_pct:.1f}% foreground)")
            counts["processed"] += 1
        except ValueError as exc:
            print(f"  [WARN]  {exc} — skipping")
            counts["skipped"] += 1
        except Exception as exc:
            print(f"  [FAIL]  {source.name}: {exc}")
            counts["failed"] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert transparent PNG images to binary segmentation masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        metavar="PATH",
        help="Folder containing source PNG images.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        metavar="PATH",
        help="Folder to write binary mask PNGs into.",
    )
    parser.add_argument(
        "--alpha-threshold", type=int, default=128,
        metavar="INT",
        help="Alpha value (0–255) below which a pixel is background. Default: 128.",
    )
    parser.add_argument(
        "--suffix", type=str, default="_mask",
        metavar="STR",
        help='Suffix appended to output filenames. Default: "_mask". Use "" to keep original names.',
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search input folder recursively for PNG files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files. By default existing files are skipped.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_dir():
        sys.exit(f"Error: input path '{args.input}' is not a directory or does not exist.")
    if not 0 <= args.alpha_threshold <= 255:
        sys.exit(f"Error: --alpha-threshold must be between 0 and 255, got {args.alpha_threshold}.")
    if args.input.resolve() == args.output.resolve():
        sys.exit("Error: --input and --output must be different directories.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Input:           {args.input}")
    print(f"Output:          {args.output}")
    print(f"Alpha threshold: {args.alpha_threshold}")
    print(f"Suffix:          '{args.suffix}'")
    print(f"Recursive:       {args.recursive}")
    print(f"Overwrite:       {args.overwrite}")
    print()

    counts = process_batch(
        input_dir=args.input,
        output_dir=args.output,
        alpha_threshold=args.alpha_threshold,
        suffix=args.suffix,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )

    print()
    print(f"Done. Processed: {counts['processed']}  "
          f"Skipped: {counts['skipped']}  "
          f"Failed: {counts['failed']}")

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()