"""
analyse_density.py

Non-invasive analysis of foreground coverage distribution within a dataset.
Reads masks from a dataset produced by build_dataset.py (via dataset.json),
computes per-image foreground pixel coverage, bins results into fixed 10%
buckets, and writes:

    <output_dir>/
        density_report.json   -- per-image coverage stats + aggregate summary
        density_plot.png      -- histogram of coverage distribution

Coverage is defined as the fraction of pixels in a mask that are foreground
(pixel value > 0), expressed as a percentage in [0, 100].

Usage
-----
    python analyse_density.py --dataset <dir> --output <dir> [options]

Options
-------
    --dataset   PATH   Dataset folder containing dataset.json and masks/
                       (required)
    --output    PATH   Folder to write density_report.json and density_plot.png
                       (required)
    --overwrite        Overwrite existing output files. Default: skip with warning.

Examples
--------
    python analyse_density.py --dataset ./dataset --output ./analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe over SSH
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

# Fixed 10% bucket edges: [0, 10), [10, 20), ..., [90, 100]
BUCKET_EDGES = list(range(0, 110, 10))   # [0, 10, 20, ..., 100]
BUCKET_LABELS = [f"{lo}–{lo+10}%" for lo in range(0, 100, 10)]  # 10 labels


def coverage_from_mask(mask_path: Path) -> float:
    """
    Load a single-channel binary mask PNG and return foreground coverage
    as a percentage in [0.0, 100.0].
    """
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return 100.0 * float((mask > 0).sum()) / float(mask.size)


def assign_bucket(coverage_pct: float) -> int:
    """
    Return the bucket index (0–9) for a given coverage percentage.
    100% is clamped into the last bucket (index 9).
    """
    return min(int(coverage_pct // 10), 9)


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
        manifest = json.load(f)

    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        sys.exit(f"Error: dataset.json in '{dataset_dir}' contains no entries.")

    return entries


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(
    entries: list[dict],
    dataset_dir: Path,
) -> tuple[list[dict], dict]:
    """
    Compute coverage for every mask entry.

    Returns:
        per_image : list of per-image result dicts, sorted by coverage ascending
        summary   : aggregate statistics dict
    """
    per_image: list[dict] = []
    failures:  list[dict] = []

    for entry in entries:
        stem     = entry.get("stem", "<unknown>")
        mask_rel = entry.get("mask_path")

        if not mask_rel:
            failures.append({"stem": stem, "reason": "mask_path missing from manifest"})
            continue

        mask_path = dataset_dir / mask_rel

        if not mask_path.exists():
            failures.append({"stem": stem, "reason": f"mask file not found: {mask_path}"})
            continue

        try:
            coverage_pct = coverage_from_mask(mask_path)
        except Exception as exc:
            failures.append({"stem": stem, "reason": str(exc)})
            continue

        per_image.append({
            "stem":         stem,
            "mask_path":    mask_rel,
            "coverage_pct": round(coverage_pct, 3),
            "bucket":       assign_bucket(coverage_pct),
            "bucket_label": BUCKET_LABELS[assign_bucket(coverage_pct)],
        })

    per_image.sort(key=lambda x: x["coverage_pct"])

    # Bucket counts
    bucket_counts = [0] * 10
    for r in per_image:
        bucket_counts[r["bucket"]] += 1

    total = len(per_image)
    coverages = [r["coverage_pct"] for r in per_image]

    bucket_stats = []
    for i, (label, count) in enumerate(zip(BUCKET_LABELS, bucket_counts)):
        bucket_coverages = [r["coverage_pct"] for r in per_image if r["bucket"] == i]
        bucket_stats.append({
            "bucket_index": i,
            "range":        label,
            "count":        count,
            "percent":      round(100 * count / total, 2) if total else 0.0,
            "mean_coverage_pct": round(float(np.mean(bucket_coverages)), 3)
                                 if bucket_coverages else None,
        })

    summary = {
        "total_analysed": total,
        "total_failed":   len(failures),
        "overall_coverage": {
            "mean_pct":   round(float(np.mean(coverages)),   3) if coverages else None,
            "median_pct": round(float(np.median(coverages)), 3) if coverages else None,
            "std_pct":    round(float(np.std(coverages)),    3) if coverages else None,
            "min_pct":    round(float(np.min(coverages)),    3) if coverages else None,
            "max_pct":    round(float(np.max(coverages)),    3) if coverages else None,
        },
        "bucket_distribution": bucket_stats,
        "failures": failures,
    }

    return per_image, summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(per_image: list[dict], summary: dict, output_dir: Path, overwrite: bool) -> Path:
    dest = output_dir / "density_report.json"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest} already exists (use --overwrite to replace)")
        return dest
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
    return dest


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def build_plot(per_image: list[dict], summary: dict, output_dir: Path, overwrite: bool) -> Path:
    dest = output_dir / "density_plot.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest} already exists (use --overwrite to replace)")
        return dest

    coverages     = [r["coverage_pct"] for r in per_image]
    bucket_counts = [b["count"]        for b in summary["bucket_distribution"]]
    total         = summary["total_analysed"]
    ov            = summary["overall_coverage"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Coverage Distribution", fontsize=14, fontweight="bold")

    # --- Left: bucket bar chart ---
    ax = axes[0]
    x_pos = range(10)
    bars  = ax.bar(x_pos, bucket_counts, color="#4393c3", edgecolor="#2c5f8a", linewidth=0.7)

    for bar, count in zip(bars, bucket_counts):
        if count == 0:
            continue
        pct = 100 * count / total if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Foreground coverage range", labelpad=8)
    ax.set_ylabel("Image count")
    ax.set_title("Images per 10% coverage bucket")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(bucket_counts) * 1.20 if any(bucket_counts) else 1)
    ax.spines[["top", "right"]].set_visible(False)

    # Vertical lines for mean and median
    ax2_twin = ax.twiny()
    ax2_twin.set_xlim(0, 100)
    ax2_twin.set_xticks([])
    if ov["mean_pct"] is not None:
        ax.axvline(
            x=ov["mean_pct"] / 10 - 0.5,   # map pct → bar x position
            color="#d6604d", linestyle="--", linewidth=1.2, label=f"mean {ov['mean_pct']}%"
        )
    if ov["median_pct"] is not None:
        ax.axvline(
            x=ov["median_pct"] / 10 - 0.5,
            color="#4d9e4d", linestyle=":", linewidth=1.2, label=f"median {ov['median_pct']}%"
        )
    ax.legend(fontsize=8, framealpha=0.7)

    # --- Right: cumulative distribution ---
    ax2 = axes[1]
    if coverages:
        sorted_cov = np.sort(coverages)
        cdf = np.arange(1, len(sorted_cov) + 1) / len(sorted_cov) * 100
        ax2.plot(sorted_cov, cdf, color="#4393c3", linewidth=1.5)
        ax2.fill_between(sorted_cov, cdf, alpha=0.15, color="#4393c3")

        # Reference lines at 25th, 50th, 75th percentiles
        for pct_label, color in [(25, "#aaaaaa"), (50, "#4d9e4d"), (75, "#aaaaaa")]:
            val = float(np.percentile(sorted_cov, pct_label))
            ax2.axvline(x=val, color=color, linestyle=":", linewidth=1.0)
            ax2.text(val + 0.5, pct_label, f"p{pct_label}={val:.1f}%",
                     fontsize=7.5, color="#555555", va="center")

    ax2.set_xlabel("Foreground coverage (%)")
    ax2.set_ylabel("Cumulative % of images")
    ax2.set_title("Cumulative distribution")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.spines[["top", "right"]].set_visible(False)

    # --- Footer ---
    footer = (
        f"Total images: {total}   |   "
        f"Mean: {ov['mean_pct']}%   |   "
        f"Median: {ov['median_pct']}%   |   "
        f"Std: {ov['std_pct']}%   |   "
        f"Range: {ov['min_pct']}% – {ov['max_pct']}%"
    )
    fig.text(0.5, -0.03, footer, ha="center", fontsize=8.5, color="#444444")

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dest


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    ov = summary["overall_coverage"]
    print(f"\n{'='*55}")
    print(f"  Coverage analysis complete")
    print(f"{'='*55}")
    print(f"  Total analysed   : {summary['total_analysed']}")
    if summary["total_failed"]:
        print(f"  Failed / skipped : {summary['total_failed']}")
    print(f"\n  Overall coverage")
    print(f"    Mean   : {ov['mean_pct']}%")
    print(f"    Median : {ov['median_pct']}%")
    print(f"    Std    : {ov['std_pct']}%")
    print(f"    Range  : {ov['min_pct']}% – {ov['max_pct']}%")
    print(f"\n  Bucket distribution (10% intervals)")
    for b in summary["bucket_distribution"]:
        bar = "█" * int(b["percent"] / 2)   # simple ASCII bar
        print(f"    {b['range']:<10}  {b['count']:>4} images  ({b['percent']:>5.1f}%)  {bar}")
    if summary["failures"]:
        print(f"\n  [WARN] {summary['total_failed']} mask(s) could not be processed:")
        for f in summary["failures"]:
            print(f"    {f['stem']}: {f['reason']}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse foreground coverage distribution within a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, metavar="PATH",
        help="Dataset folder containing dataset.json and masks/.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, metavar="PATH",
        help="Folder to write density_report.json and density_plot.png.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' is not a directory or does not exist.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.output}")

    entries = load_manifest(args.dataset)
    print(f"\nLoaded {len(entries)} entries from dataset.json")
    print("Computing coverage per mask...")

    per_image, summary = analyse(entries, args.dataset)

    print_summary(summary)

    args.output.mkdir(parents=True, exist_ok=True)

    report_path = write_report(per_image, summary, args.output, args.overwrite)
    plot_path   = build_plot(per_image, summary, args.output, args.overwrite)

    print(f"Report : {report_path}")
    print(f"Plot   : {plot_path}")


if __name__ == "__main__":
    main()