"""
evaluate.py

Evaluates predicted segmentation masks against ground-truth masks and
produces a JSON report and PNG plots.

Supports single-checkpoint evaluation and optional side-by-side comparison
of two checkpoints (e.g. baseline vs. fine-tuned).

Metrics computed
----------------
    IoU                 Intersection over Union (primary metric)
    Dice                Dice coefficient (F1 score on pixels)
    Boundary F1         Precision/recall at mask boundaries (tolerance configurable)
    False Positive Rate Fraction of background pixels predicted as foreground
    False Negative Rate Fraction of foreground pixels predicted as background

All metrics are computed per image, then aggregated (mean, std, median, min, max)
and also broken down by coverage bucket (10% intervals, same as analyse_density.py).

Output structure
----------------
<output_dir>/
    evaluation_report.json      always written
    metrics_distributions.png   histograms of IoU / Dice / Boundary F1
    coverage_vs_iou.png         IoU per 10% coverage bucket
    visual_samples.png          grid of GT vs predicted masks
                                (best / worst / median N cases)
    comparison_report.json      written only in comparison mode
    comparison_metrics.png      side-by-side metric comparison plot
    comparison_visual_samples.png  GT / pred_A / pred_B grid

Usage
-----
    # Single checkpoint evaluation
    python evaluate.py \\
        --dataset  ./dataset \\
        --preds    ./predictions_sam2.1_hiera_base_plus \\
        --output   ./evaluation

    # Comparison mode
    python evaluate.py \\
        --dataset  ./dataset \\
        --preds    ./predictions_sam2.1_hiera_base_plus \\
        --preds-b  ./predictions_finetuned_step7000 \\
        --output   ./evaluation

Options
-------
    --dataset       PATH    Dataset folder with dataset.json and masks/ (required)
    --preds         PATH    Predictions folder (output of run_inference.py) (required)
    --preds-b       PATH    Second predictions folder for comparison mode (optional)
    --output        PATH    Folder to write reports and plots (required)
    --split         PATH    Optional .txt file to restrict evaluation to a subset of stems
    --boundary-tol  INT     Boundary F1 tolerance in pixels. Default: 2
    --num-samples   INT     Number of best/worst/median visual samples. Default: 5
    --label-a       STR     Label for first predictions in comparison plots. Default: auto
    --label-b       STR     Label for second predictions in comparison plots. Default: auto
    --overwrite             Overwrite existing output files
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_erosion, binary_dilation


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return float(intersection / union) if union > 0 else 1.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    denom        = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def compute_boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    def get_boundary(mask):
        if mask.sum() == 0:
            return mask
        return mask ^ binary_erosion(mask)

    pred_b = get_boundary(pred)
    gt_b   = get_boundary(gt)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    gt_dilated   = binary_dilation(gt_b,   iterations=tolerance)
    pred_dilated = binary_dilation(pred_b, iterations=tolerance)

    precision = float((pred_b & gt_dilated).sum())   / (pred_b.sum() + 1e-8)
    recall    = float((gt_b   & pred_dilated).sum())  / (gt_b.sum()   + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)


def compute_fpr(pred: np.ndarray, gt: np.ndarray) -> float:
    """False positive rate: FP / (FP + TN)"""
    fp = ( pred & ~gt).sum()
    tn = (~pred & ~gt).sum()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def compute_fnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """False negative rate: FN / (FN + TP)"""
    fn = (~pred & gt).sum()
    tp = ( pred & gt).sum()
    return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0


def coverage_bucket(gt: np.ndarray) -> int:
    """Return 0–9 bucket index based on foreground coverage percentage."""
    cov = 100.0 * gt.sum() / gt.size
    return min(int(cov // 10), 9)


def compute_all_metrics(
    pred: np.ndarray,
    gt:   np.ndarray,
    boundary_tol: int,
) -> dict:
    return {
        "iou":          compute_iou(pred, gt),
        "dice":         compute_dice(pred, gt),
        "boundary_f1":  compute_boundary_f1(pred, gt, boundary_tol),
        "fpr":          compute_fpr(pred, gt),
        "fnr":          compute_fnr(pred, gt),
        "coverage_pct": round(100.0 * gt.sum() / gt.size, 3),
        "bucket":       coverage_bucket(gt),
    }


# ---------------------------------------------------------------------------
# Dataset / prediction loading
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
        sys.exit(f"Error: --split '{split_path}' does not exist.")
    stems = {l.strip() for l in split_path.read_text().splitlines() if l.strip()}
    if not stems:
        sys.exit(f"Error: --split '{split_path}' is empty.")
    return stems


def resolve_pred_path(preds_dir: Path, stem: str) -> Path | None:
    """Find <stem>_pred.png in predictions dir."""
    p = preds_dir / f"{stem}_pred.png"
    return p if p.exists() else None


def auto_label(preds_dir: Path) -> str:
    """Derive a human label from the predictions folder name."""
    name = preds_dir.name
    if name.startswith("predictions_"):
        return name[len("predictions_"):]
    return name


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    entries:      list[dict],
    dataset_dir:  Path,
    preds_dir:    Path,
    stem_filter:  set[str] | None,
    boundary_tol: int,
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate predictions against GT masks.

    Returns (results, failures).
    results: one dict per image with metrics + paths.
    """
    results:  list[dict] = []
    failures: list[dict] = []

    for entry in entries:
        stem = entry["stem"]

        if stem_filter and stem not in stem_filter:
            continue

        gt_path   = dataset_dir / entry["mask_path"]
        pred_path = resolve_pred_path(preds_dir, stem)

        if pred_path is None:
            failures.append({"stem": stem, "reason": "prediction file not found"})
            continue

        if not gt_path.exists():
            failures.append({"stem": stem, "reason": f"GT mask not found: {gt_path}"})
            continue

        try:
            gt   = np.array(Image.open(gt_path).convert("L"),   dtype=np.uint8) > 0
            pred = np.array(Image.open(pred_path).convert("L"), dtype=np.uint8) > 0

            metrics = compute_all_metrics(pred, gt, boundary_tol)
            results.append({
                "stem":       stem,
                "image_path": entry["image_path"],
                "gt_path":    entry["mask_path"],
                "pred_path":  str(pred_path),
                **metrics,
            })
        except Exception as exc:
            failures.append({"stem": stem, "reason": str(exc)})

    return results, failures


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

BUCKET_LABELS = [f"{lo}–{lo+10}%" for lo in range(0, 100, 10)]
METRIC_KEYS   = ["iou", "dice", "boundary_f1", "fpr", "fnr"]


def aggregate(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    a = np.array(values)
    return {
        "mean":   round(float(a.mean()),   4),
        "std":    round(float(a.std()),    4),
        "median": round(float(np.median(a)), 4),
        "min":    round(float(a.min()),    4),
        "max":    round(float(a.max()),    4),
    }


def build_summary(results: list[dict]) -> dict:
    overall = {k: aggregate([r[k] for r in results]) for k in METRIC_KEYS}

    # Per coverage bucket breakdown
    buckets = {}
    for i, label in enumerate(BUCKET_LABELS):
        bucket_results = [r for r in results if r["bucket"] == i]
        buckets[label] = {
            "count": len(bucket_results),
            **{k: aggregate([r[k] for r in bucket_results]) for k in METRIC_KEYS},
        }

    return {
        "total_evaluated": len(results),
        "overall":         overall,
        "by_coverage_bucket": buckets,
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    output_dir:   Path,
    preds_dir:    Path,
    summary:      dict,
    results:      list[dict],
    failures:     list[dict],
    boundary_tol: int,
    overwrite:    bool,
    filename:     str = "evaluation_report.json",
) -> Path:
    dest = output_dir / filename
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest} already exists")
        return dest
    report = {
        "predictions_dir": str(preds_dir.resolve()),
        "boundary_tolerance_px": boundary_tol,
        "summary":  summary,
        "failures": failures,
        "per_image": sorted(results, key=lambda r: r["iou"]),
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return dest


def write_comparison_report(
    output_dir:   Path,
    label_a:      str,
    label_b:      str,
    summary_a:    dict,
    summary_b:    dict,
    results_a:    list[dict],
    results_b:    list[dict],
    boundary_tol: int,
    overwrite:    bool,
) -> Path:
    dest = output_dir / "comparison_report.json"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest} already exists")
        return dest

    # Compute per-image delta for stems present in both
    stems_a = {r["stem"]: r for r in results_a}
    stems_b = {r["stem"]: r for r in results_b}
    common  = sorted(set(stems_a) & set(stems_b))

    deltas = []
    for stem in common:
        ra, rb = stems_a[stem], stems_b[stem]
        deltas.append({
            "stem": stem,
            **{f"delta_{k}": round(rb[k] - ra[k], 4) for k in METRIC_KEYS},
        })

    report = {
        "boundary_tolerance_px": boundary_tol,
        label_a: {"summary": summary_a},
        label_b: {"summary": summary_b},
        "per_image_deltas": deltas,
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return dest


# ---------------------------------------------------------------------------
# Plots — single evaluation
# ---------------------------------------------------------------------------

PLOT_METRICS = [
    ("iou",         "IoU",          "#4393c3"),
    ("dice",        "Dice",         "#74c476"),
    ("boundary_f1", "Boundary F1",  "#fd8d3c"),
]


def plot_metric_distributions(
    results:    list[dict],
    output_dir: Path,
    label:      str,
    overwrite:  bool,
) -> Path:
    dest = output_dir / "metrics_distributions.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Metric Distributions — {label}", fontsize=12, fontweight="bold")

    for ax, (key, name, color) in zip(axes, PLOT_METRICS):
        values = [r[key] for r in results]
        ax.hist(values, bins=20, range=(0, 1), color=color,
                edgecolor="white", linewidth=0.5)
        mean = np.mean(values)
        ax.axvline(mean, color="#333333", linestyle="--", linewidth=1.2,
                   label=f"mean={mean:.3f}")
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Image count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_coverage_vs_iou(
    results:    list[dict],
    output_dir: Path,
    label:      str,
    overwrite:  bool,
) -> Path:
    dest = output_dir / "coverage_vs_iou.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    bucket_ious   = [[] for _ in range(10)]
    bucket_counts = [0] * 10
    for r in results:
        bucket_ious[r["bucket"]].append(r["iou"])
        bucket_counts[r["bucket"]] += 1

    means  = [np.mean(v) if v else None for v in bucket_ious]
    stds   = [np.std(v)  if v else None for v in bucket_ious]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"IoU by Coverage Bucket — {label}", fontsize=12, fontweight="bold")

    # Left: mean IoU per bucket with error bars
    ax = axes[0]
    x  = np.arange(10)
    y  = [m if m is not None else 0.0 for m in means]
    e  = [s if s is not None else 0.0 for s in stds]
    bars = ax.bar(x, y, yerr=e, color="#4393c3", edgecolor="#2c5f8a",
                  linewidth=0.7, capsize=3)

    for bar, mean, count in zip(bars, means, bucket_counts):
        if mean is None or count == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mean:.2f}\n(n={count})",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("Foreground coverage range")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.15)
    ax.set_title("Mean IoU per coverage bucket")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: scatter IoU vs coverage
    ax2 = axes[1]
    xs  = [r["coverage_pct"] for r in results]
    ys  = [r["iou"]          for r in results]
    ax2.scatter(xs, ys, alpha=0.4, s=15, color="#4393c3")
    ax2.set_xlabel("Foreground coverage (%)")
    ax2.set_ylabel("IoU")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("IoU vs coverage (scatter)")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Visual samples
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask_overlay(image: np.ndarray, mask_path: Path,
                       color: tuple, alpha: float = 0.45) -> np.ndarray:
    """Return image with mask overlaid in given RGB color."""
    out  = image.copy().astype(float)
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + v * alpha,
                                out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def _select_samples(results: list[dict], n: int, key: str = "iou") -> dict[str, list[dict]]:
    """Return best, worst and median n results by key."""
    sorted_r = sorted(results, key=lambda r: r[key])
    total    = len(sorted_r)
    mid      = total // 2
    half_n   = n // 2

    median_slice = sorted_r[max(0, mid - half_n): min(total, mid - half_n + n)]

    return {
        "worst":  sorted_r[:n],
        "median": median_slice,
        "best":   sorted_r[-n:][::-1],
    }


def plot_visual_samples(
    results:     list[dict],
    dataset_dir: Path,
    output_dir:  Path,
    label:       str,
    num_samples: int,
    overwrite:   bool,
    filename:    str = "visual_samples.png",
    results_b:   list[dict] | None = None,
    label_b:     str | None = None,
) -> Path:
    dest = output_dir / filename
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    samples = _select_samples(results, num_samples)
    comparison_mode = results_b is not None
    stems_b = {r["stem"]: r for r in results_b} if results_b else {}

    # Columns: image | GT | pred_A [| pred_B]
    n_cols = 4 if comparison_mode else 3
    groups = ["worst", "median", "best"]
    rows_per_group = num_samples
    n_rows = len(groups) * rows_per_group + len(groups)  # +1 header row per group

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.2))
    fig.suptitle(f"Visual Samples — {label}", fontsize=12, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.05, wspace=0.03)

    col_titles = ["Image", "GT mask", f"Pred: {label}"]
    if comparison_mode:
        col_titles.append(f"Pred: {label_b}")

    row_idx = 0
    for group_name in groups:
        group_samples = samples[group_name]

        # Group header row
        header_ax = fig.add_subplot(gs[row_idx, :])
        header_ax.axis("off")
        header_ax.text(
            0.5, 0.5,
            f"{group_name.upper()} {num_samples} by IoU",
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            transform=header_ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8", edgecolor="#aaaaaa"),
        )
        row_idx += 1

        for r in group_samples:
            stem      = r["stem"]
            img_path  = dataset_dir / r["image_path"]
            gt_path   = dataset_dir / r["gt_path"]
            pred_path = Path(r["pred_path"])

            img        = _load_rgb(img_path)
            gt_overlay = _load_mask_overlay(img, gt_path,   color=(50, 200, 50))
            pa_overlay = _load_mask_overlay(img, pred_path, color=(200, 50, 50))

            row_data = [img, gt_overlay, pa_overlay]
            if comparison_mode and stem in stems_b:
                pb_path = Path(stems_b[stem]["pred_path"])
                row_data.append(_load_mask_overlay(img, pb_path, color=(50, 100, 220)))

            for col_idx, (panel, title) in enumerate(zip(row_data, col_titles)):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.imshow(panel)
                ax.axis("off")
                if row_idx == 1:  # only label columns on first data row
                    ax.set_title(title, fontsize=8, pad=3)

            # Annotate with metrics
            iou_str = f"IoU={r['iou']:.3f}  Dice={r['dice']:.3f}  BF1={r['boundary_f1']:.3f}"
            fig.add_subplot(gs[row_idx, 0]).set_xlabel(
                f"{stem}\n{iou_str}", fontsize=6.5, labelpad=2
            )
            row_idx += 1

    plt.tight_layout()
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_comparison_metrics(
    results_a:  list[dict],
    results_b:  list[dict],
    label_a:    str,
    label_b:    str,
    output_dir: Path,
    overwrite:  bool,
) -> Path:
    dest = output_dir / "comparison_metrics.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    # Align on common stems
    stems_a = {r["stem"]: r for r in results_a}
    stems_b = {r["stem"]: r for r in results_b}
    common  = sorted(set(stems_a) & set(stems_b))

    if not common:
        print("[WARN] No common stems between the two prediction sets — skipping comparison plot.")
        return dest

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Comparison: {label_a}  vs  {label_b}",
                 fontsize=12, fontweight="bold")

    colors = {"a": "#4393c3", "b": "#d6604d"}

    for ax, (key, name, _) in zip(axes, PLOT_METRICS):
        vals_a = [stems_a[s][key] for s in common]
        vals_b = [stems_b[s][key] for s in common]

        # Paired histogram with 20 bins
        bins = np.linspace(0, 1, 21)
        ax.hist(vals_a, bins=bins, alpha=0.6, color=colors["a"],
                label=f"{label_a} (μ={np.mean(vals_a):.3f})", edgecolor="white")
        ax.hist(vals_b, bins=bins, alpha=0.6, color=colors["b"],
                label=f"{label_b} (μ={np.mean(vals_b):.3f})", edgecolor="white")

        # Mean lines
        ax.axvline(np.mean(vals_a), color=colors["a"], linestyle="--", linewidth=1.2)
        ax.axvline(np.mean(vals_b), color=colors["b"], linestyle="--", linewidth=1.2)

        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Image count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Inset: per-image IoU delta scatter
    ax_delta = axes[0].inset_axes([0.55, 0.55, 0.43, 0.43])
    deltas   = [stems_b[s]["iou"] - stems_a[s]["iou"] for s in common]
    ax_delta.hist(deltas, bins=15, color="#888888", edgecolor="white")
    ax_delta.axvline(0, color="black", linewidth=0.8)
    ax_delta.set_title(f"ΔIoU (μ={np.mean(deltas):+.3f})", fontsize=7)
    ax_delta.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(label: str, summary: dict) -> None:
    ov = summary["overall"]
    print(f"\n  {label}")
    print(f"  {'─'*50}")
    print(f"  Total evaluated : {summary['total_evaluated']}")
    for key, name in [("iou","IoU"),("dice","Dice"),
                      ("boundary_f1","Boundary F1"),("fpr","FPR"),("fnr","FNR")]:
        s = ov[key]
        if s["mean"] is None:
            continue
        print(f"  {name:<14}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"median={s['median']:.4f}  [{s['min']:.4f} – {s['max']:.4f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 predictions against ground-truth masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",      required=True,  type=Path, metavar="PATH")
    parser.add_argument("--preds",        required=True,  type=Path, metavar="PATH",
                        help="Predictions folder (output of run_inference.py).")
    parser.add_argument("--preds-b",      default=None,   type=Path, metavar="PATH",
                        help="Second predictions folder for comparison mode.")
    parser.add_argument("--output",       required=True,  type=Path, metavar="PATH")
    parser.add_argument("--split",        default=None,   type=Path, metavar="PATH")
    parser.add_argument("--boundary-tol", default=2,      type=int,  metavar="INT",
                        help="Boundary F1 tolerance in pixels. Default: 2.")
    parser.add_argument("--num-samples",  default=5,      type=int,  metavar="INT",
                        help="Best/worst/median visual samples per group. Default: 5.")
    parser.add_argument("--label-a",      default=None,   type=str,  metavar="STR",
                        help="Label for first predictions. Default: derived from folder name.")
    parser.add_argument("--label-b",      default=None,   type=str,  metavar="STR",
                        help="Label for second predictions. Default: derived from folder name.")
    parser.add_argument("--overwrite",    action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for attr, flag in [("dataset","--dataset"), ("preds","--preds")]:
        p = getattr(args, attr)
        if not p.is_dir():
            sys.exit(f"Error: {flag} '{p}' is not a directory or does not exist.")
    if args.preds_b is not None and not args.preds_b.is_dir():
        sys.exit(f"Error: --preds-b '{args.preds_b}' is not a directory or does not exist.")
    if args.num_samples < 1:
        sys.exit("Error: --num-samples must be >= 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    label_a = args.label_a or auto_label(args.preds)
    label_b = args.label_b or (auto_label(args.preds_b) if args.preds_b else None)
    comparison = args.preds_b is not None

    print(f"Dataset      : {args.dataset}")
    print(f"Predictions  : {args.preds}  [{label_a}]")
    if comparison:
        print(f"Predictions B: {args.preds_b}  [{label_b}]")
    print(f"Output       : {args.output}")
    print(f"Boundary tol : {args.boundary_tol} px")
    print(f"Num samples  : {args.num_samples}")
    print()

    entries     = load_manifest(args.dataset)
    stem_filter = load_split_filter(args.split)

    # ── Evaluate predictions A ──────────────────────────────────────────────
    print(f"Evaluating {label_a}...")
    results_a, failures_a = evaluate_predictions(
        entries, args.dataset, args.preds, stem_filter, args.boundary_tol
    )
    if not results_a:
        sys.exit("No results — check that predictions folder contains *_pred.png files.")

    summary_a = build_summary(results_a)

    # ── Evaluate predictions B (comparison mode) ────────────────────────────
    results_b = failures_b = summary_b = None
    if comparison:
        print(f"Evaluating {label_b}...")
        results_b, failures_b = evaluate_predictions(
            entries, args.dataset, args.preds_b, stem_filter, args.boundary_tol
        )
        summary_b = build_summary(results_b)

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print_summary(label_a, summary_a)
    if comparison:
        print_summary(label_b, summary_b)
    if failures_a:
        print(f"\n  [WARN] {len(failures_a)} failure(s) for {label_a}:")
        for f in failures_a:
            print(f"    {f['stem']}: {f['reason']}")
    print(f"{'='*55}\n")

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Reports ─────────────────────────────────────────────────────────────
    rp_a = write_report(args.output, args.preds, summary_a, results_a,
                        failures_a, args.boundary_tol, args.overwrite)
    print(f"Report       : {rp_a}")

    if comparison:
        rp_cmp = write_comparison_report(
            args.output, label_a, label_b,
            summary_a, summary_b, results_a, results_b,
            args.boundary_tol, args.overwrite,
        )
        print(f"Comparison   : {rp_cmp}")

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    p1 = plot_metric_distributions(results_a, args.output, label_a, args.overwrite)
    p2 = plot_coverage_vs_iou(results_a, args.output, label_a, args.overwrite)
    p3 = plot_visual_samples(
        results_a, args.dataset, args.output, label_a,
        args.num_samples, args.overwrite,
        results_b=results_b, label_b=label_b,
    )
    print(f"  {p1.name}")
    print(f"  {p2.name}")
    print(f"  {p3.name}")

    if comparison:
        p4 = plot_comparison_metrics(
            results_a, results_b, label_a, label_b, args.output, args.overwrite
        )
        print(f"  {p4.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()