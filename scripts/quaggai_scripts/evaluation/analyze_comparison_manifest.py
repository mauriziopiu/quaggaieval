"""
analyze_comparison_manifest.py

Reads a comparison_manifest.json (output of visualize_comparison.py) and
produces two plots for qualitative + quantitative analysis of the
baseline vs. fine-tuned comparison, without re-running inference or
touching any image/mask files.

Selection of "worst / median / best" cases is based on the FINE-TUNED
model's IoU (baseline IoU is still shown alongside for context).

Plots produced
---------------
1. coverage_vs_iou_finetuned.png
   Two panels:
     - Mean fine-tuned IoU per 10%-coverage bucket (bar chart w/ error bars),
       same style as evaluate.py's coverage_vs_iou.png.
     - Scatter of fine-tuned IoU vs. coverage for all images, with the
       selected worst/median/best cases highlighted and labeled by stem.

2. confusion_matrices_worst_median_best.png
   For each selected worst/median/best case: a text panel (stem, IoU for
   both models, coverage, and the corresponding diff image filenames) plus
   two 2x2 confusion matrices (baseline and fine-tuned, side by side),
   cell values expressed as % of total image pixels.

3. refinement_trajectory.png (only if the manifest has `rounds` data,
   i.e. it came from visualize_iterative_refinement.py)
   Two panels:
     - Per-image IoU trajectory across every round that was run (round 1
       through however many points it took), colored green (improved
       beyond round 1) / gray (unchanged), with the winning best round
       marked with a star. Any continuation past the best round —
       i.e. the process kept adding points after its own peak — is
       drawn as a dashed red overshoot segment.
     - A histogram of Δ(round1→best) IoU across multi-round images.

4. refinement_best_round_by_coverage.png (only if refinement data is present)
   Distribution of which round produced the best IoU (1..max_points),
   grouped by the same coverage buckets as coverage_vs_iou_finetuned.png:
   a box-and-whisker per bucket (median, IQR, whiskers) plus a jittered
   strip of every individual sample's best_round, so you can see e.g.
   whether low-coverage images typically need more corrective points
   than high-coverage ones.

Output structure
-----------------
<output>/
    coverage_vs_iou_finetuned.png
    confusion_matrices_worst_median_best.png
    selection_summary.json                       which stems were selected per group + their metrics
    refinement_trajectory.png                     only if refinement (rounds) data is present
    refinement_best_round_by_coverage.png         only if refinement data is present
    refinement_summary.json                       only if refinement data is present — per-image round
                                                    trajectory, delta from round 1 to best, and an
                                                    explicit overshoot list

Usage
-----
    python analyze_comparison_manifest.py \\
        --manifest ./comparison_visuals/comparison_manifest.json \\
        --output   ./comparison_analysis

Options
-------
    --manifest      PATH    Path to comparison_manifest.json (required)
    --output        PATH    Folder to write plots + summary (required)
    --num-samples   INT     Number of worst/median/best cases per group. Default: 5.
    --overwrite             Overwrite existing output files. Default: skip.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        sys.exit(f"Error: manifest '{manifest_path}' does not exist.")
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    if "predictions" not in data:
        sys.exit(f"Error: '{manifest_path}' does not look like a comparison_manifest.json (no 'predictions' key).")
    return data


def load_processed_entries(manifest: dict) -> list[dict]:
    entries = [p for p in manifest["predictions"] if p.get("status") == "processed"]
    if not entries:
        sys.exit("Error: manifest contains no entries with status == 'processed'.")
    return entries


def add_derived_fields(entries: list[dict]) -> list[dict]:
    """Add coverage_pct and bucket (0-9) derived from pixel counts."""
    for e in entries:
        counts = e["pixel_counts_finetuned"]
        total  = sum(counts.values())
        gt_fg  = counts["tp"] + counts["fn"]
        e["coverage_pct"] = 100.0 * gt_fg / total if total > 0 else 0.0
        e["bucket"]       = min(int(e["coverage_pct"] // 10), 9)
    return entries


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(entries: list[dict], n: int, key: str = "iou_finetuned") -> dict[str, list[dict]]:
    """Return worst, median and best n entries by key (fine-tuned IoU)."""
    sorted_e = sorted(entries, key=lambda e: e[key])
    total    = len(sorted_e)
    mid      = total // 2
    half_n   = n // 2

    median_slice = sorted_e[max(0, mid - half_n): min(total, mid - half_n + n)]

    return {
        "worst":  sorted_e[:n],
        "median": median_slice,
        "best":   sorted_e[-n:][::-1],
    }


# ---------------------------------------------------------------------------
# Coverage vs. IoU plot
# ---------------------------------------------------------------------------

BUCKET_LABELS = [f"{lo}–{lo+10}%" for lo in range(0, 100, 10)]

GROUP_STYLE = {
    "worst":  ("#d6604d", "D", "Worst"),
    "median": ("#f4a582", "s", "Median"),
    "best":   ("#2ca25f", "o", "Best"),
}


def plot_coverage_vs_iou(
    entries:    list[dict],
    samples:    dict[str, list[dict]],
    output_dir: Path,
    overwrite:  bool,
) -> Path:
    dest = output_dir / "coverage_vs_iou_finetuned.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    bucket_ious   = [[] for _ in range(10)]
    bucket_counts = [0] * 10
    for e in entries:
        bucket_ious[e["bucket"]].append(e["iou_finetuned"])
        bucket_counts[e["bucket"]] += 1

    means = [np.mean(v) if v else None for v in bucket_ious]
    stds  = [np.std(v)  if v else None for v in bucket_ious]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fine-tuned IoU vs. Coverage — worst/median/best highlighted",
                 fontsize=12, fontweight="bold")

    # Left: bucketed mean IoU
    ax = axes[0]
    x  = np.arange(10)
    y  = [m if m is not None else 0.0 for m in means]
    e_ = [s if s is not None else 0.0 for s in stds]
    bars = ax.bar(x, y, yerr=e_, color="#4393c3", edgecolor="#2c5f8a",
                  linewidth=0.7, capsize=3, alpha=0.85, zorder=2)
    for bar, mean, count in zip(bars, means, bucket_counts):
        if mean is None or count == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.2f}\n(n={count})", ha="center", va="bottom", fontsize=7)

    # Jittered strip of individual samples within each bin, to show the
    # within-bucket distribution the mean/error-bar alone would hide.
    jitter_rng = np.random.default_rng(0)
    for i, values in enumerate(bucket_ious):
        if not values:
            continue
        jitter = jitter_rng.uniform(-0.3, 0.3, size=len(values))
        ax.scatter(i + jitter, values, color="#1a1a1a", alpha=0.35, s=8,
                  edgecolor="none", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("Foreground coverage range")
    ax.set_ylabel("Mean IoU (fine-tuned)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Mean IoU per coverage bucket")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: scatter with worst/median/best highlighted
    ax2 = axes[1]
    xs = [e["coverage_pct"]  for e in entries]
    ys = [e["iou_finetuned"] for e in entries]
    ax2.scatter(xs, ys, alpha=0.25, s=15, color="#4393c3", zorder=1, label="all images")

    for group, sample_list in samples.items():
        color, marker, name = GROUP_STYLE[group]
        gx = [s["coverage_pct"]  for s in sample_list]
        gy = [s["iou_finetuned"] for s in sample_list]
        ax2.scatter(gx, gy, color=color, marker=marker, s=65, edgecolor="black",
                   linewidth=0.6, zorder=3, label=name)
        for s in sample_list:
            ax2.annotate(s["stem"], (s["coverage_pct"], s["iou_finetuned"]),
                        fontsize=5.5, xytext=(3, 3), textcoords="offset points")

    ax2.set_xlabel("Foreground coverage (%)")
    ax2.set_ylabel("IoU (fine-tuned)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("IoU vs coverage — worst/median/best highlighted")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def confusion_matrix_pct(counts: dict) -> np.ndarray:
    """2x2 array [[TN, FP], [FN, TP]] as % of total image pixels."""
    total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    arr = np.array([[counts["tn"], counts["fp"]],
                     [counts["fn"], counts["tp"]]], dtype=float)
    return 100.0 * arr / total if total > 0 else arr


def draw_confusion_ax(ax, matrix: np.ndarray, show_labels: bool) -> None:
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            val   = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    if show_labels:
        ax.set_xticklabels(["Pred: bg", "Pred: fg"], fontsize=6)
        ax.set_yticklabels(["GT: bg", "GT: fg"], fontsize=6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.tick_params(length=0)


def plot_confusion_grid(
    samples:     dict[str, list[dict]],
    output_dir:  Path,
    num_samples: int,
    overwrite:   bool,
) -> Path:
    dest = output_dir / "confusion_matrices_worst_median_best.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    groups  = ["worst", "median", "best"]
    n_cols  = 3  # label panel, baseline matrix, finetuned matrix
    n_rows  = len(groups) * num_samples + len(groups)  # +1 header row per group

    fig = plt.figure(figsize=(n_cols * 3.3, n_rows * 1.15))
    fig.suptitle("Confusion Matrices — Worst / Median / Best (by fine-tuned IoU)",
                 fontsize=12, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.4, wspace=0.15, width_ratios=[1.5, 1, 1])

    row_idx = 0
    first_data_row = True
    for group in groups:
        header_ax = fig.add_subplot(gs[row_idx, :])
        header_ax.axis("off")
        header_ax.text(
            0.5, 0.5, f"{group.upper()} {num_samples} by fine-tuned IoU",
            ha="center", va="center", fontsize=10, fontweight="bold",
            transform=header_ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8", edgecolor="#aaaaaa"),
        )
        row_idx += 1

        for s in samples[group]:
            label_ax = fig.add_subplot(gs[row_idx, 0])
            label_ax.axis("off")
            delta = s["iou_finetuned"] - s["iou_baseline"]
            text = (
                f"{s['stem']}\n"
                f"IoU  base={s['iou_baseline']:.3f}  ft={s['iou_finetuned']:.3f}  (Δ={delta:+.3f})\n"
                f"coverage={s['coverage_pct']:.1f}%\n"
                f"diff: {s['paths']['diff_baseline']}\n"
                f"      {s['paths']['diff_finetuned']}"
            )
            label_ax.text(0.0, 0.5, text, fontsize=6.5, va="center", ha="left")

            ax_b = fig.add_subplot(gs[row_idx, 1])
            draw_confusion_ax(ax_b, confusion_matrix_pct(s["pixel_counts_baseline"]), first_data_row)
            if first_data_row:
                ax_b.set_title("Baseline", fontsize=8)

            ax_f = fig.add_subplot(gs[row_idx, 2])
            draw_confusion_ax(ax_f, confusion_matrix_pct(s["pixel_counts_finetuned"]), first_data_row)
            if first_data_row:
                ax_f.set_title("Fine-tuned", fontsize=8)

            first_data_row = False
            row_idx += 1

    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Refinement (round trajectory / best-round) analysis
# ---------------------------------------------------------------------------

REFINEMENT_DELTA_EPS = 1e-9


def has_refinement_data(entries: list[dict]) -> bool:
    return any(e.get("rounds") for e in entries)


def refined_entries(entries: list[dict]) -> list[dict]:
    """All entries that carry `rounds` data (every processed entry from
    visualize_iterative_refinement.py)."""
    return [e for e in entries if e.get("rounds")]


def multi_round_entries(entries: list[dict]) -> list[dict]:
    """Entries where more than one round actually ran (i.e. a correction was attempted)."""
    return [e for e in refined_entries(entries) if len(e["rounds"]) > 1]


def entry_correction_status(e: dict) -> str:
    if e.get("stop_reason") == "perfect_after_round1":
        return "not_applicable"
    round1_iou = e["rounds"][0]["iou_finetuned"]
    best_iou   = e["best_round"]["iou_finetuned"]
    return "improved" if best_iou > round1_iou + REFINEMENT_DELTA_EPS else "unchanged"


REFINEMENT_STATUS_COLOR = {
    "improved":  "#2ca25f",
    "unchanged": "#999999",
}
OVERSHOOT_COLOR = "#d6604d"


def plot_refinement_trajectory(
    entries:    list[dict],
    output_dir: Path,
    overwrite:  bool,
) -> Path | None:
    dest = output_dir / "refinement_trajectory.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    multi = multi_round_entries(entries)
    if not multi:
        print("[WARN] No multi-round entries found — skipping refinement_trajectory.png")
        return None

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("IoU Trajectory Across Refinement Rounds", fontsize=12, fontweight="bold")

    # Left: per-image round-by-round IoU trajectory
    ax = axes[0]
    for e in multi:
        xs = [r["round"] for r in e["rounds"]]
        ys = [r["iou_finetuned"] for r in e["rounds"]]
        color     = REFINEMENT_STATUS_COLOR[entry_correction_status(e)]
        best_round = e["best_round"]["round"]
        best_idx   = xs.index(best_round)

        ax.plot(xs[:best_idx + 1], ys[:best_idx + 1], color=color, alpha=0.5, linewidth=1.1,
                marker="o", markersize=2.5, zorder=1)
        ax.scatter([xs[best_idx]], [ys[best_idx]], color=color, edgecolor="black",
                   linewidth=0.5, s=35, marker="*", zorder=3)

        if best_idx < len(xs) - 1:  # overshoot: continued past the best round
            ax.plot(xs[best_idx:], ys[best_idx:], color=OVERSHOOT_COLOR, alpha=0.7, linewidth=1.1,
                    linestyle="--", marker="o", markersize=2.5, zorder=2)

    ax.set_xlabel("Round (cumulative points placed)")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-image IoU trajectory (n={len(multi)})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([0], [0], color=REFINEMENT_STATUS_COLOR["improved"],  lw=2, label="improved (round1→best)"),
        Line2D([0], [0], color=REFINEMENT_STATUS_COLOR["unchanged"], lw=2, label="unchanged"),
        Line2D([0], [0], color=OVERSHOOT_COLOR, lw=2, linestyle="--", label="overshoot (past best)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=9, label="best round"),
    ], fontsize=7, loc="lower right")

    # Right: Δ(round1 -> best) histogram
    ax2 = axes[1]
    deltas = [e["best_round"]["iou_finetuned"] - e["rounds"][0]["iou_finetuned"] for e in multi]
    lo, hi = min(deltas + [-0.01]), max(deltas + [0.01])
    ax2.hist(deltas, bins=np.linspace(lo, hi, 25), color="#4393c3", edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1)
    mean_delta = float(np.mean(deltas))
    ax2.axvline(mean_delta, color="#333333", linestyle="--", linewidth=1.2,
               label=f"mean Δ={mean_delta:+.3f}")

    n_overshoot = sum(1 for e in multi if e["best_round"]["round"] != e["rounds"][-1]["round"])
    ax2.set_title(f"Δ(round1→best) IoU distribution  (overshoot in {n_overshoot}/{len(multi)})", fontsize=9)
    ax2.set_xlabel("ΔIoU (best round − round 1)")
    ax2.set_ylabel("Image count")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def write_refinement_summary(
    entries:       list[dict],
    output_dir:    Path,
    manifest_path: Path,
) -> Path | None:
    refined = refined_entries(entries)
    if not refined:
        return None

    dest = output_dir / "refinement_summary.json"

    rows = []
    for e in refined:
        round1_iou     = e["rounds"][0]["iou_finetuned"]
        best_iou       = e["best_round"]["iou_finetuned"]
        last_round_num = e["rounds"][-1]["round"]
        last_iou       = e["rounds"][-1]["iou_finetuned"]
        overshoot      = e["best_round"]["round"] != last_round_num
        rows.append({
            "stem":                  e["stem"],
            "num_rounds":            len(e["rounds"]),
            "stop_reason":           e.get("stop_reason"),
            "correction_status":     entry_correction_status(e),
            "iou_round1":            round1_iou,
            "iou_best":              best_iou,
            "best_round":            e["best_round"]["round"],
            "last_round":            last_round_num,
            "delta_round1_to_best":  round(best_iou - round1_iou, 4),
            "overshoot":             overshoot,
            "overshoot_magnitude":   round(best_iou - last_iou, 4),
            "diff_finetuned":        e["paths"].get("diff_finetuned"),
        })

    overshoots = sorted([r for r in rows if r["overshoot"]], key=lambda r: -r["overshoot_magnitude"])
    multi_rows = [r for r in rows if r["num_rounds"] > 1]

    summary = {
        "source_manifest":           str(manifest_path.resolve()),
        "images_with_rounds_data":   len(rows),
        "no_correction_possible":    sum(1 for r in rows if r["correction_status"] == "not_applicable"),
        "improved_count":            sum(1 for r in rows if r["correction_status"] == "improved"),
        "unchanged_count":           sum(1 for r in rows if r["correction_status"] == "unchanged"),
        "overshoot_count":           len(overshoots),
        "mean_delta_round1_to_best": round(float(np.mean([r["delta_round1_to_best"] for r in multi_rows])), 4)
                                      if multi_rows else None,
        "overshoots":  overshoots,
        "per_image":   rows,
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return dest


def plot_coverage_vs_iou_round1(
    entries:    list[dict],
    output_dir: Path,
    overwrite:  bool,
) -> Path | None:
    """Same bucketed mean-IoU + jittered-strip panel as plot_coverage_vs_iou's
    left panel, but using ROUND 1 (first point only) IoU instead of the best
    round — a before/after comparison point for the refinement process."""
    dest = output_dir / "refinement_coverage_vs_iou_round1.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    refined = refined_entries(entries)
    if not refined:
        print("[WARN] No entries with rounds data found — skipping refinement_coverage_vs_iou_round1.png")
        return None

    bucket_ious   = [[] for _ in range(10)]
    bucket_counts = [0] * 10
    for e in refined:
        round1_iou = e["rounds"][0]["iou_finetuned"]
        bucket_ious[e["bucket"]].append(round1_iou)
        bucket_counts[e["bucket"]] += 1

    means = [np.mean(v) if v else None for v in bucket_ious]
    stds  = [np.std(v)  if v else None for v in bucket_ious]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Round-1 (First Point Only) IoU vs. Coverage", fontsize=12, fontweight="bold")

    x  = np.arange(10)
    y  = [m if m is not None else 0.0 for m in means]
    e_ = [s if s is not None else 0.0 for s in stds]
    bars = ax.bar(x, y, yerr=e_, color="#4393c3", edgecolor="#2c5f8a",
                  linewidth=0.7, capsize=3, alpha=0.85, zorder=2)
    for bar, mean, count in zip(bars, means, bucket_counts):
        if mean is None or count == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.2f}\n(n={count})", ha="center", va="bottom", fontsize=7)

    jitter_rng = np.random.default_rng(0)
    for i, values in enumerate(bucket_ious):
        if not values:
            continue
        jitter = jitter_rng.uniform(-0.3, 0.3, size=len(values))
        ax.scatter(i + jitter, values, color="#1a1a1a", alpha=0.35, s=8,
                  edgecolor="none", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("Foreground coverage range")
    ax.set_ylabel("Mean IoU (round 1 only)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Mean IoU per coverage bucket — round 1 (first point) only (n={len(refined)})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_best_round_by_coverage(
    entries:    list[dict],
    output_dir: Path,
    overwrite:  bool,
    max_points: int | None = None,
) -> Path | None:
    """Box-and-whisker + jittered strip of best_round, grouped by coverage bucket."""
    dest = output_dir / "refinement_best_round_by_coverage.png"
    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest}")
        return dest

    refined = refined_entries(entries)
    if not refined:
        print("[WARN] No entries with rounds data found — skipping refinement_best_round_by_coverage.png")
        return None

    bucket_best_rounds: list[list[int]] = [[] for _ in range(10)]
    for e in refined:
        bucket_best_rounds[e["bucket"]].append(e["best_round"]["round"])

    if max_points is None:
        max_points = max(r for values in bucket_best_rounds for r in values)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("Best Round Reached, by Coverage Bucket", fontsize=12, fontweight="bold")

    box_positions = [i for i, values in enumerate(bucket_best_rounds) if values]
    box_data      = [values for values in bucket_best_rounds if values]

    if box_data:
        ax.boxplot(
            box_data, positions=box_positions, widths=0.6, showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#4393c3", alpha=0.55, edgecolor="#2c5f8a"),
            medianprops=dict(color="#0b3d5c", linewidth=1.6),
            whiskerprops=dict(color="#2c5f8a"),
            capprops=dict(color="#2c5f8a"),
        )

    # Jittered strip of individual samples' best_round within each bin.
    jitter_rng = np.random.default_rng(0)
    for i, values in enumerate(bucket_best_rounds):
        if not values:
            continue
        jitter = jitter_rng.uniform(-0.3, 0.3, size=len(values))
        ax.scatter(i + jitter, values, color="#1a1a1a", alpha=0.35, s=8,
                  edgecolor="none", zorder=4)
        ax.text(i, max_points + 0.4, f"n={len(values)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(range(10)))
    ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_xlim(-0.6, 9.6)
    ax.set_xlabel("Foreground coverage range")
    ax.set_yticks(range(1, max_points + 1))
    ax.set_ylim(0.5, max_points + 1.0)
    ax.set_ylabel("Round the best IoU was reached at")
    ax.set_title(f"Distribution of best_round per coverage bucket (n={len(refined)})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Selection summary
# ---------------------------------------------------------------------------

def write_selection_summary(
    samples:     dict[str, list[dict]],
    output_dir:  Path,
    manifest_path: Path,
    num_samples: int,
) -> Path:
    dest = output_dir / "selection_summary.json"
    summary = {
        "source_manifest": str(manifest_path.resolve()),
        "selection_key":   "iou_finetuned",
        "num_samples":     num_samples,
        "groups": {
            group: [
                {
                    "stem":           s["stem"],
                    "iou_baseline":   s["iou_baseline"],
                    "iou_finetuned":  s["iou_finetuned"],
                    "coverage_pct":   round(s["coverage_pct"], 3),
                    "diff_baseline":  s["paths"]["diff_baseline"],
                    "diff_finetuned": s["paths"]["diff_finetuned"],
                }
                for s in sample_list
            ]
            for group, sample_list in samples.items()
        },
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot IoU-vs-coverage and confusion matrices from a comparison_manifest.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest",    required=True, type=Path, metavar="PATH")
    parser.add_argument("--output",      required=True, type=Path, metavar="PATH")
    parser.add_argument("--num-samples", default=5,      type=int, metavar="INT")
    parser.add_argument("--overwrite",   action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        sys.exit(f"Error: --manifest '{args.manifest}' is not a file or does not exist.")
    if args.num_samples < 1:
        sys.exit("Error: --num-samples must be >= 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Manifest    : {args.manifest}")
    print(f"Output      : {args.output}")
    print(f"Num samples : {args.num_samples}")
    print()

    manifest = load_manifest(args.manifest)
    entries  = load_processed_entries(manifest)
    entries  = add_derived_fields(entries)

    print(f"Entries loaded: {len(entries)}")

    samples = select_samples(entries, args.num_samples, key="iou_finetuned")

    args.output.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    p1 = plot_coverage_vs_iou(entries, samples, args.output, args.overwrite)
    print(f"  {p1.name}")
    p2 = plot_confusion_grid(samples, args.output, args.num_samples, args.overwrite)
    print(f"  {p2.name}")

    s1 = write_selection_summary(samples, args.output, args.manifest, args.num_samples)
    print(f"  {s1.name}")

    if has_refinement_data(entries):
        print("\nRefinement data detected — generating round-trajectory analysis...")
        p3 = plot_refinement_trajectory(entries, args.output, args.overwrite)
        if p3 is not None:
            print(f"  {p3.name}")
        p4 = plot_best_round_by_coverage(entries, args.output, args.overwrite,
                                         max_points=manifest.get("max_points"))
        if p4 is not None:
            print(f"  {p4.name}")
        p5 = plot_coverage_vs_iou_round1(entries, args.output, args.overwrite)
        if p5 is not None:
            print(f"  {p5.name}")
        s2 = write_refinement_summary(entries, args.output, args.manifest)
        if s2 is not None:
            print(f"  {s2.name}")
            if s2.exists():
                overshoot_count = sum(1 for e in refined_entries(entries)
                                      if e["best_round"]["round"] != e["rounds"][-1]["round"])
                if overshoot_count > 0:
                    print(f"  [WARN] {overshoot_count} image(s) overshot past their best round — "
                          f"see refinement_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
