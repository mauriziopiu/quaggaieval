"""
analyze_finetuned_metrics.py

Reads a comparison_manifest.json (output of visualize_comparison.py, or
visualize_iterative_refinement.py) and computes per-image pixel-level
metrics for the FINE-TUNED model only, writing a single JSON report.
Does not re-run inference or touch any image/mask files.

Per image, TP/TN/FP/FN are expressed as a percentage of the image's
total pixel count (summing to 100%), alongside precision, recall,
accuracy and F1 score computed from the raw pixel counts. These
top-level per-image fields always describe the manifest's BEST/final
prediction (for iterative-refinement manifests: the highest-IoU round).

If the manifest comes from visualize_iterative_refinement.py (has a
`rounds` list per entry), each per-image entry also gets a full
`round_trajectory` (metrics for every round that was run), the winning
`best_round`, and `delta_round1_to_best` — how much the corrective
points changed IoU overall relative to the single-point round 1. The
report gains a top-level `refinement` block summarizing this across the
dataset, including an explicit **overshoot** list: images where a later
round ended up worse than the best one reached along the way (i.e. the
process kept adding points past the peak). This section is omitted
entirely for manifests without round data (e.g. from
visualize_comparison.py).

Output JSON structure
----------------------
{
    "source_manifest": "<path>",
    "total_evaluated": <int>,          # entries with status == "processed"
    "excluded": {"failed": <int>, "skipped": <int>},
    "average": {                       # macro-average over all evaluated images (best round)
        "precision": <float>,
        "recall":    <float>,
        "accuracy":  <float>,
        "f1":        <float>
    },
    "rankings": {                      # redundant highest/median/lowest per metric (best round)
        "tp_pct":     {"highest": {...}, "median": {...}, "lowest": {...}},
        "tn_pct":     {...},
        "fp_pct":     {...},
        "fn_pct":     {...},
        "precision":  {...},
        "recall":     {...},
        "accuracy":   {...},
        "f1":         {...}
    },
    "refinement": {                    # only present for iterative-refinement manifests
        "images_with_rounds_data": <int>,
        "no_correction_possible":  <int>,   # round 1 was already pixel-perfect
        "improved_count":  <int>,           # best round beat round 1
        "unchanged_count": <int>,           # no round ever beat round 1
        "overshoot_count": <int>,           # the process continued past its own best round
        "average_delta_round1_to_best":   <float>,   # over multi-round images only
        "average_overshoot_magnitude":    <float>,   # best_iou - last_round_iou, over multi-round images
        "delta_round1_to_best_rankings":  {"lowest": {...}, "median": {...}, "highest": {...}},
        "stop_reason_breakdown": {"iou_threshold_met": <int>, "max_points_reached": <int>,
                                   "perfect_after_round1": <int>},
        "overshoots": [
            {"stem": "...", "best_round": <int>, "last_round": <int>,
             "overshoot_magnitude": <float>, "diff_finetuned": "<path>"},
            ...   # worst (largest wasted gain) first
        ]
    },
    "per_image": [
        {
            "stem": "...",
            "tp_pct": <float>, "tn_pct": <float>, "fp_pct": <float>, "fn_pct": <float>,
            "precision": <float>, "recall": <float>, "accuracy": <float>, "f1": <float>,
            "diff_finetuned": "<relative path, for cross-referencing the diff image — best round>",

            # only present for iterative-refinement manifests:
            "num_rounds":  <int>,
            "stop_reason": "iou_threshold_met" | "max_points_reached" | "perfect_after_round1",
            "best_round":  {"round": <int>, "iou": <float>, "is_last_round": <bool>},
            "round_trajectory": [{"round": 1, "iou": <float>, "tp_pct": ..., "precision": ..., ...}, ...],
            "delta_round1_to_best": <float>,   # always >= 0
            "overshoot": <bool>,
            "overshoot_magnitude": <float>,    # best_iou - last_round_iou, >= 0
            "correction_status": "improved" | "unchanged" | "not_applicable"
        },
        ...
    ]
}

Usage
-----
    python analyze_finetuned_metrics.py \\
        --manifest ./comparison_visuals/comparison_manifest.json \\
        --output   ./comparison_analysis/finetuned_metrics_report.json

Options
-------
    --manifest   PATH   Path to comparison_manifest.json (required)
    --output     PATH   Path to write the JSON report (required)
    --overwrite         Overwrite the output file if it already exists. Default: skip.
"""

import argparse
import json
import sys
from pathlib import Path


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


def split_by_status(manifest: dict) -> tuple[list[dict], int, int]:
    processed = [p for p in manifest["predictions"] if p.get("status") == "processed"]
    failed    = sum(1 for p in manifest["predictions"] if p.get("status") == "failed")
    skipped   = sum(1 for p in manifest["predictions"] if p.get("status") == "skipped")
    if not processed:
        sys.exit("Error: manifest contains no entries with status == 'processed'.")
    return processed, failed, skipped


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

DELTA_EPS = 1e-9


def compute_metrics_from_counts(counts: dict) -> dict:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1_denom  = 2 * tp + fp + fn
    f1        = (2 * tp / f1_denom) if f1_denom > 0 else 1.0
    accuracy  = (tp + tn) / total if total > 0 else 1.0

    return {
        "tp_pct":    round(100.0 * tp / total, 4) if total > 0 else 0.0,
        "tn_pct":    round(100.0 * tn / total, 4) if total > 0 else 0.0,
        "fp_pct":    round(100.0 * fp / total, 4) if total > 0 else 0.0,
        "fn_pct":    round(100.0 * fn / total, 4) if total > 0 else 0.0,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "accuracy":  round(accuracy, 4),
        "f1":        round(f1, 4),
    }


def has_refinement_data(entry: dict) -> bool:
    return bool(entry.get("rounds"))


def compute_image_metrics(entry: dict) -> dict:
    metrics = compute_metrics_from_counts(entry["pixel_counts_finetuned"])
    metrics["stem"]           = entry["stem"]
    metrics["diff_finetuned"] = entry["paths"]["diff_finetuned"]

    if has_refinement_data(entry):
        round_trajectory = []
        for r in entry["rounds"]:
            rm = compute_metrics_from_counts(r["pixel_counts_finetuned"])
            rm["round"] = r["round"]
            rm["iou"]   = r["iou_finetuned"]
            round_trajectory.append(rm)

        round1_iou = round_trajectory[0]["iou"]
        last_round = round_trajectory[-1]
        best       = entry["best_round"]
        best_iou   = best["iou_finetuned"]
        overshoot  = best["round"] != last_round["round"]

        if entry.get("stop_reason") == "perfect_after_round1":
            correction_status = "not_applicable"
        elif best_iou > round1_iou + DELTA_EPS:
            correction_status = "improved"
        else:
            correction_status = "unchanged"

        metrics["num_rounds"]  = len(round_trajectory)
        metrics["stop_reason"] = entry.get("stop_reason")
        metrics["best_round"] = {
            "round":         best["round"],
            "iou":           best_iou,
            "is_last_round": not overshoot,
        }
        metrics["round_trajectory"]        = round_trajectory
        metrics["delta_round1_to_best"]    = round(best_iou - round1_iou, 4)
        metrics["overshoot"]               = overshoot
        metrics["overshoot_magnitude"]     = round(best_iou - last_round["iou"], 4)
        metrics["correction_status"]       = correction_status

    return metrics


# ---------------------------------------------------------------------------
# Rankings and averages
# ---------------------------------------------------------------------------

RANKING_KEYS = ["tp_pct", "tn_pct", "fp_pct", "fn_pct", "precision", "recall", "accuracy", "f1"]


def ranking_entry(image_metrics: dict, key: str) -> dict:
    return {
        "stem":           image_metrics["stem"],
        "value":          image_metrics[key],
        "diff_finetuned": image_metrics["diff_finetuned"],
    }


def build_rankings(per_image: list[dict]) -> dict:
    rankings = {}
    for key in RANKING_KEYS:
        sorted_images = sorted(per_image, key=lambda m: m[key])
        median_img = sorted_images[(len(sorted_images) - 1) // 2]
        rankings[key] = {
            "lowest":  ranking_entry(sorted_images[0],  key),
            "median":  ranking_entry(median_img,        key),
            "highest": ranking_entry(sorted_images[-1], key),
        }
    return rankings


def build_average(per_image: list[dict]) -> dict:
    n = len(per_image)
    return {
        "precision": round(sum(m["precision"] for m in per_image) / n, 4),
        "recall":    round(sum(m["recall"]    for m in per_image) / n, 4),
        "accuracy":  round(sum(m["accuracy"]  for m in per_image) / n, 4),
        "f1":        round(sum(m["f1"]        for m in per_image) / n, 4),
    }


# ---------------------------------------------------------------------------
# Refinement (round trajectory / best-round) summary
# ---------------------------------------------------------------------------

def _delta_entry(m: dict) -> dict:
    return {"stem": m["stem"], "delta_round1_to_best": m["delta_round1_to_best"],
            "diff_finetuned": m["diff_finetuned"]}


def _overshoot_entry(m: dict) -> dict:
    return {
        "stem":                m["stem"],
        "best_round":          m["best_round"]["round"],
        "last_round":          m["round_trajectory"][-1]["round"],
        "overshoot_magnitude": m["overshoot_magnitude"],
        "diff_finetuned":      m["diff_finetuned"],
    }


def build_refinement_summary(per_image: list[dict]) -> dict | None:
    """None if this manifest has no `rounds` data at all."""
    refined = [m for m in per_image if "num_rounds" in m]
    if not refined:
        return None

    not_applicable = [m for m in refined if m["correction_status"] == "not_applicable"]
    improved       = [m for m in refined if m["correction_status"] == "improved"]
    unchanged      = [m for m in refined if m["correction_status"] == "unchanged"]
    overshoots     = [m for m in refined if m["overshoot"]]
    multi_round    = [m for m in refined if m["num_rounds"] > 1]  # excludes not_applicable images

    average_delta          = None
    average_overshoot_mag  = None
    delta_rankings         = None
    if multi_round:
        n = len(multi_round)
        average_delta         = round(sum(m["delta_round1_to_best"] for m in multi_round) / n, 4)
        average_overshoot_mag = round(sum(m["overshoot_magnitude"]  for m in multi_round) / n, 4)
        sorted_by_delta = sorted(multi_round, key=lambda m: m["delta_round1_to_best"])
        delta_rankings = {
            "lowest":  _delta_entry(sorted_by_delta[0]),
            "median":  _delta_entry(sorted_by_delta[(n - 1) // 2]),
            "highest": _delta_entry(sorted_by_delta[-1]),
        }

    stop_reason_breakdown: dict[str, int] = {}
    for m in refined:
        sr = m.get("stop_reason") or "unknown"
        stop_reason_breakdown[sr] = stop_reason_breakdown.get(sr, 0) + 1

    overshoots_sorted = sorted(overshoots, key=lambda m: -m["overshoot_magnitude"])  # worst first

    return {
        "images_with_rounds_data": len(refined),
        "no_correction_possible":  len(not_applicable),
        "improved_count":  len(improved),
        "unchanged_count": len(unchanged),
        "overshoot_count": len(overshoots),
        "average_delta_round1_to_best":  average_delta,
        "average_overshoot_magnitude":   average_overshoot_mag,
        "delta_round1_to_best_rankings": delta_rankings,
        "stop_reason_breakdown":         stop_reason_breakdown,
        "overshoots":                    [_overshoot_entry(m) for m in overshoots_sorted],
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def build_report(manifest_path: Path, per_image: list[dict], failed: int, skipped: int) -> dict:
    report = {
        "source_manifest":  str(manifest_path.resolve()),
        "total_evaluated":  len(per_image),
        "excluded":         {"failed": failed, "skipped": skipped},
        "average":          build_average(per_image),
        "rankings":         build_rankings(per_image),
    }
    refinement = build_refinement_summary(per_image)
    if refinement is not None:
        report["refinement"] = refinement
    report["per_image"] = per_image
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-image pixel-level metrics for the fine-tuned model from a comparison_manifest.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output",   required=True, type=Path, metavar="PATH")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        sys.exit(f"Error: --manifest '{args.manifest}' is not a file or does not exist.")
    if args.output.exists() and not args.overwrite:
        sys.exit(f"Error: --output '{args.output}' already exists. Pass --overwrite to replace it.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Manifest: {args.manifest}")
    print(f"Output  : {args.output}")
    print()

    manifest = load_manifest(args.manifest)
    entries, failed, skipped = split_by_status(manifest)

    print(f"Processed entries: {len(entries)}  (failed: {failed}, skipped: {skipped})")

    per_image = [compute_image_metrics(e) for e in entries]
    report     = build_report(args.manifest, per_image, failed, skipped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    avg = report["average"]
    print(f"\nAverage (macro)  precision={avg['precision']:.4f}  "
          f"recall={avg['recall']:.4f}  accuracy={avg['accuracy']:.4f}  f1={avg['f1']:.4f}")

    refinement = report.get("refinement")
    if refinement is not None:
        print(f"\nRefinement (round trajectory, best round wins):")
        print(f"  Images with rounds data : {refinement['images_with_rounds_data']}  "
              f"(already perfect at round 1: {refinement['no_correction_possible']})")
        print(f"  Improved beyond round 1 : {refinement['improved_count']}")
        print(f"  Unchanged (no round beat round 1): {refinement['unchanged_count']}")
        print(f"  Overshot past best round: {refinement['overshoot_count']}")
        if refinement["average_delta_round1_to_best"] is not None:
            print(f"  Mean Δ(round1→best) IoU : {refinement['average_delta_round1_to_best']:+.4f}")
            print(f"  Mean overshoot magnitude: {refinement['average_overshoot_magnitude']:+.4f}")
        print(f"  Stop reasons            : {refinement['stop_reason_breakdown']}")
        if refinement["overshoot_count"] > 0:
            print(f"  [WARN] {refinement['overshoot_count']} image(s) passed their best round and kept going:")
            for o in refinement["overshoots"]:
                print(f"    {o['stem']}: best=round{o['best_round']}  last=round{o['last_round']}  "
                      f"wasted ΔIoU={o['overshoot_magnitude']:+.4f}")

    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
