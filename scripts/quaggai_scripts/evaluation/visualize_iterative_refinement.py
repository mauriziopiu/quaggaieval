"""
visualize_iterative_refinement.py

Runs SAM2 inference with two checkpoints (baseline and fine-tuned),
modeling a multi-click interactive refinement workflow for the
fine-tuned model: a first random point, then up to (max_points - 1)
additional corrective points, each chosen from wherever the previous
prediction erred most, stopping once IoU reaches a target threshold.

Per image
---------
1. Point 1: a single random positive point sampled from the GT
   foreground (seeded).
2. Baseline model runs ONCE with point 1 only → model_baseline.png.
   No diff is produced for the baseline (out of scope for this workflow).
3. Fine-tuned model runs with point 1 only (round 1).
4. Point 2 is ALWAYS added next (round 2), chosen from round 1's
   errors — unless round 1 is already pixel-perfect (FP=FN=0), in
   which case there is no valid region for a corrective point and the
   image stops at round 1 (stop_reason: "perfect_after_round1").
5. From round 3 onward: after each round, if IoU is still below
   --iou-threshold and --max-points hasn't been reached, another
   corrective point is added (same FP/FN-dominant-region rule, applied
   to the latest prediction) and another round is run. Stops as soon as
   IoU >= --iou-threshold (stop_reason: "iou_threshold_met") or
   --max-points is reached (stop_reason: "max_points_reached").
6. Each round K writes model_finetuned_K.png + diff_finetuned_K.png,
   showing all K cumulative points (point 1 white, all others magenta).
7. Once every round for that image is computed, the round with the
   highest IoU (earliest round wins ties) is additionally saved as
   model_finetuned_best.png + diff_finetuned_best.png.
8. ground_truth.png is rendered last, showing every point that was
   ultimately placed for that image.

Output structure
-----------------
<output>/
    <stem>_ground_truth.png        GT overlay (green), all points used (point 1 white, rest magenta)
    <stem>_model_baseline.png      baseline prediction overlay (red), point 1 only
    <stem>_model_finetuned_1.png   round-1 prediction overlay (blue)
    <stem>_diff_finetuned_1.png    round-1 TP/FP/FN error map
    <stem>_model_finetuned_2.png   round-2 outputs (always produced, unless round 1 was
    <stem>_diff_finetuned_2.png    already pixel-perfect)
    <stem>_model_finetuned_3.png   round-3+ outputs, only as many as actually ran
    <stem>_diff_finetuned_3.png
    ...
    <stem>_model_finetuned_best.png  duplicate of whichever round had the highest IoU
    <stem>_diff_finetuned_best.png   (earliest round wins ties)
    ...
    comparison_manifest.json       metadata (incl. max_points, iou_threshold) + a `rounds`
                                    list per image (one entry per round: cumulative point
                                    count, the point added that round, IoU, score, pixel
                                    counts, file paths), a `best_round` block, and
                                    `stop_reason`. Fields iou_finetuned / score_finetuned /
                                    pixel_counts_finetuned / paths.model_finetuned /
                                    paths.diff_finetuned alias the BEST round, for
                                    compatibility with analyze_comparison_manifest.py /
                                    analyze_finetuned_metrics.py. paths.diff_baseline is
                                    not produced in this workflow.

Usage
-----
    python visualize_iterative_refinement.py \\
        --dataset             ./dataset \\
        --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \\
        --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \\
        --output              ./refinement_visuals

Options
-------
    --dataset               PATH    Dataset folder with dataset.json, images/, masks/
                                     (required)
    --baseline-checkpoint   PATH    Baseline SAM2 checkpoint (.pt) (required)
    --finetuned-checkpoint  PATH    Fine-tuned SAM2 checkpoint (.pt) (required)
    --baseline-model-cfg    STR     Model config for the baseline checkpoint.
                                     Default: auto-detected from filename.
    --finetuned-model-cfg   STR     Model config for the fine-tuned checkpoint.
                                     Default: auto-detected from filename.
    --split                 PATH    Optional .txt file of stems to process.
                                     If omitted, all entries in dataset.json are used.
    --seed                  INT     Random seed for point sampling. Default: 42.
    --max-points            INT     Maximum TOTAL points per image, point 1 included
                                     (round K has K points). Must be >= 2, since point 2
                                     is always placed. Default: 5.
    --iou-threshold         FLOAT   Stop adding points once IoU reaches this value.
                                     Default: 0.9.
    --output                PATH    Folder to write overlay images + manifest (required)
    --alpha                 FLOAT   Overlay alpha for GT/baseline/finetuned masks. Default: 0.45
    --diff-alpha            FLOAT   Overlay alpha for TP/FP/FN regions in diff images. Default: 0.55
    --diff-dim              FLOAT   Brightness factor for the dimmed background in diff
                                     images (0-1). Default: 0.5
    --marker-alpha          FLOAT   Alpha for the prompt point markers. Default: 0.75
    --overwrite             Overwrite existing output files. Default: skip.
    --dry-run               Print what would be run without writing anything.
    --device                STR     Device to run inference on. Default: cuda.

Example
-------
    python visualize_iterative_refinement.py \\
        --dataset ./dataset \\
        --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \\
        --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \\
        --split ./dataset/val.txt \\
        --max-points 5 --iou-threshold 0.9 \\
        --output ./refinement_visuals
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    sys.exit(
        "Error: SAM2 is not installed.\n"
        "Install with: pip install -e '.[dev]' from the sam2 repo root."
    )


# ---------------------------------------------------------------------------
# Overlay colors
# ---------------------------------------------------------------------------

GT_COLOR        = (50, 200, 50)     # green
BASELINE_COLOR  = (200, 50, 50)     # red
FINETUNED_COLOR = (50, 100, 220)    # blue

# Colorblind-safe diff palette (avoids relying on red/green distinction)
DIFF_TP_COLOR = (60, 120, 220)      # blue    — predicted and in GT
DIFF_FP_COLOR = (255, 140, 0)       # orange  — predicted, not in GT
DIFF_FN_COLOR = (240, 220, 40)      # yellow  — in GT, missed

POINT1_MARKER_COLOR = (255, 255, 255)  # white — point 1
POINT2_MARKER_COLOR = (255, 0, 255)    # magenta — every corrective point (2, 3, 4, ...)


def marker_colors_for(num_points: int) -> list[tuple[int, int, int]]:
    """Point 1 is white; every subsequent (corrective) point is magenta."""
    if num_points <= 0:
        return []
    return [POINT1_MARKER_COLOR] + [POINT2_MARKER_COLOR] * (num_points - 1)


# ---------------------------------------------------------------------------
# Model config auto-detection
# ---------------------------------------------------------------------------

CHECKPOINT_TO_CFG = {
    "hiera_tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
    "hiera_small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "hiera_large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def auto_detect_cfg(checkpoint_path: Path) -> str:
    name = checkpoint_path.stem.lower()
    for key, cfg in CHECKPOINT_TO_CFG.items():
        if key.replace("_", "") in name.replace("_", "").replace("+", "plus"):
            return cfg
    print(
        f"[WARN] Could not auto-detect model config from '{checkpoint_path.name}'. "
        f"Falling back to 'configs/sam2.1/sam2.1_hiera_b+.yaml'. "
        f"Use --baseline-model-cfg/--finetuned-model-cfg to override if this is wrong."
    )
    return "configs/sam2.1/sam2.1_hiera_b+.yaml"


# ---------------------------------------------------------------------------
# Point selection
# ---------------------------------------------------------------------------

def sample_random_point1(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """A single random positive point from the GT foreground. None if mask is empty."""
    fg_pixels = np.argwhere(mask)  # (N, 2) in (y, x)
    if len(fg_pixels) == 0:
        return None
    y, x = fg_pixels[rng.integers(len(fg_pixels))]
    return np.array([x, y])


def prompt_image_center(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([w // 2, h // 2])


def sample_from_region(region: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A single random point from a non-empty boolean region. Caller guarantees non-empty."""
    pixels = np.argwhere(region)  # (N, 2) in (y, x)
    y, x = pixels[rng.integers(len(pixels))]
    return np.array([x, y])


def select_corrective_point(
    pred: np.ndarray,
    gt:   np.ndarray,
    rng:  np.random.Generator,
) -> dict | None:
    """
    Choose a corrective point from the current prediction's errors.

    Used for point 2 and every subsequent point. Returns None if the
    current prediction is already pixel-perfect (FP == FN == 0) —
    nothing left to correct. Otherwise returns a dict with the sampled
    point, its label (1=positive/grow, 0=negative/shrink), and a reason
    tag.
    """
    fp_region = pred & ~gt
    fn_region = ~pred & gt
    fp_area   = int(fp_region.sum())
    fn_area   = int(fn_region.sum())

    if fp_area == 0 and fn_area == 0:
        return None

    if fp_area > fn_area:
        region, label, reason = fp_region, 0, "shrink_fp_dominant"
    elif fn_area > fp_area:
        region, label, reason = fn_region, 1, "grow_fn_dominant"
    else:
        region, label, reason = fn_region, 1, "grow_tie"  # nonzero tie → default to grow

    point = sample_from_region(region, rng)
    return {"point": point, "label": label, "reason": reason, "fp_area": fp_area, "fn_area": fn_area}


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
# Overlay rendering
# ---------------------------------------------------------------------------

def colorize_mask_overlay(
    image: np.ndarray,
    mask:  np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend `color` into `image` wherever `mask` is True."""
    out = image.astype(np.float32).copy()
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + v * alpha,
                                out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_prompt_markers(
    image:  np.ndarray,
    points: np.ndarray,
    colors: list[tuple[int, int, int]],
    radius: int | None = None,
    alpha:  float = 0.75,
) -> np.ndarray:
    """Draw small semi-transparent circular markers, one color per point."""
    h, w = image.shape[:2]
    if radius is None:
        radius = max(3, min(h, w) // 150)

    base    = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    fill_a  = int(255 * alpha)

    for (x, y), color in zip(points, colors):
        bbox = [int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius]
        draw.ellipse(bbox, fill=(*color, fill_a), outline=(0, 0, 0, fill_a))

    composited = Image.alpha_composite(base, overlay)
    return np.array(composited.convert("RGB"))


def build_diff_overlay(
    image:     np.ndarray,
    pred:      np.ndarray,
    gt:        np.ndarray,
    dim:       float,
    alpha:     float,
    tp_color:  tuple[int, int, int] = DIFF_TP_COLOR,
    fp_color:  tuple[int, int, int] = DIFF_FP_COLOR,
    fn_color:  tuple[int, int, int] = DIFF_FN_COLOR,
) -> np.ndarray:
    """TP/FP/FN error map on top of a dimmed copy of the original image."""
    out = (image.astype(np.float32) * dim)
    tp  = pred & gt
    fp  = pred & ~gt
    fn  = ~pred & gt

    for region, color in [(tp, tp_color), (fp, fp_color), (fn, fn_color)]:
        for c, v in enumerate(color):
            out[:, :, c] = np.where(region, out[:, :, c] * (1 - alpha) + v * alpha,
                                    out[:, :, c])

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_mask(
    predictor: "SAM2ImagePredictor",
    image:     np.ndarray,
    points:    np.ndarray,
    labels:    np.ndarray,
    device:    str,
) -> tuple[np.ndarray, float]:
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )
    return masks[0] > 0, float(scores[0])


def pixel_counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "tp": int((pred & gt).sum()),
        "fp": int((pred & ~gt).sum()),
        "fn": int((~pred & gt).sum()),
        "tn": int((~pred & ~gt).sum()),
    }


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return float(intersection / union) if union > 0 else 1.0


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

STOP_REASONS = ["iou_threshold_met", "max_points_reached", "perfect_after_round1"]


def run_refinement(
    entries:              list[dict],
    dataset_dir:           Path,
    output_dir:            Path,
    baseline_predictor:    "SAM2ImagePredictor",
    finetuned_predictor:   "SAM2ImagePredictor",
    rng:                   np.random.Generator,
    max_points:            int,
    iou_threshold:         float,
    alpha:                 float,
    diff_alpha:            float,
    diff_dim:              float,
    marker_alpha:          float,
    overwrite:             bool,
    dry_run:               bool,
    device:                str,
) -> tuple[list[dict], dict[str, int]]:
    results: list[dict] = []
    counts = {
        "processed": 0, "skipped": 0, "failed": 0, "empty_mask": 0,
        "overshoot": 0,
        **{f"stop_{r}": 0 for r in STOP_REASONS},
    }

    for entry in entries:
        stem      = entry["stem"]
        img_path  = dataset_dir / entry["image_path"]
        mask_path = dataset_dir / entry["mask_path"]

        core_paths = {
            "ground_truth":         output_dir / f"{stem}_ground_truth.png",
            "model_baseline":       output_dir / f"{stem}_model_baseline.png",
            "model_finetuned_1":    output_dir / f"{stem}_model_finetuned_1.png",
            "diff_finetuned_1":     output_dir / f"{stem}_diff_finetuned_1.png",
            "model_finetuned_best": output_dir / f"{stem}_model_finetuned_best.png",
            "diff_finetuned_best":  output_dir / f"{stem}_diff_finetuned_best.png",
        }

        if not overwrite and all(p.exists() for p in core_paths.values()):
            print(f"  [SKIP]  {stem}")
            counts["skipped"] += 1
            results.append({"stem": stem, "status": "skipped"})
            continue

        if dry_run:
            print(f"  [DRY]   {stem}  →  {output_dir}/{stem}_*.png")
            counts["processed"] += 1
            continue

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            gt    = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0

            point1 = sample_random_point1(gt, rng)
            if point1 is None:
                print(f"  [EMPTY] {stem} — GT mask is empty, using image center")
                point1 = prompt_image_center(image)
                counts["empty_mask"] += 1

            # ---- Baseline: single point, unaffected by the round loop ----
            points1 = np.array([point1])
            labels1 = np.array([1])
            pred_baseline, score_baseline = predict_mask(baseline_predictor, image, points1, labels1, device)
            iou_baseline    = compute_iou(pred_baseline, gt)
            counts_baseline = pixel_counts(pred_baseline, gt)

            # ---- Fine-tuned: iterative round loop ----
            points_so_far: list[np.ndarray] = [point1]
            labels_so_far: list[int]        = [1]
            rounds: list[dict] = []

            def run_round(point_added: dict) -> dict:
                pts  = np.array(points_so_far)
                lbls = np.array(labels_so_far)
                pred, score = predict_mask(finetuned_predictor, image, pts, lbls, device)
                iou       = compute_iou(pred, gt)
                counts_ft = pixel_counts(pred, gt)
                overlay = draw_prompt_markers(
                    colorize_mask_overlay(image, pred, FINETUNED_COLOR, alpha),
                    pts, marker_colors_for(len(pts)), alpha=marker_alpha,
                )
                diff_img = build_diff_overlay(image, pred, gt, diff_dim, diff_alpha)
                return {
                    "round":                  len(points_so_far),
                    "num_points":             len(points_so_far),
                    "point_added":            point_added,
                    "iou_finetuned":          iou,
                    "score_finetuned":        score,
                    "pixel_counts_finetuned": counts_ft,
                    "pred_mask":              pred,
                    "overlay_img":            overlay,
                    "diff_img":               diff_img,
                }

            # Round 1
            rounds.append(run_round({"coords": point1.tolist(), "label": 1, "reason": "initial_random_fg"}))

            # Round 2: always placed, unless round 1 is already pixel-perfect
            stop_reason = None
            corrective  = select_corrective_point(rounds[-1]["pred_mask"], gt, rng)
            if corrective is None:
                stop_reason = "perfect_after_round1"
            else:
                points_so_far.append(corrective["point"])
                labels_so_far.append(corrective["label"])
                rounds.append(run_round({
                    "coords": corrective["point"].tolist(),
                    "label":  corrective["label"],
                    "reason": corrective["reason"],
                }))

                # Rounds 3..max_points, gated by the IoU threshold
                while rounds[-1]["iou_finetuned"] < iou_threshold and len(points_so_far) < max_points:
                    nxt = select_corrective_point(rounds[-1]["pred_mask"], gt, rng)
                    if nxt is None:
                        break  # became perfect; the loop guard above will also see this
                    points_so_far.append(nxt["point"])
                    labels_so_far.append(nxt["label"])
                    rounds.append(run_round({
                        "coords": nxt["point"].tolist(),
                        "label":  nxt["label"],
                        "reason": nxt["reason"],
                    }))

                if stop_reason is None:
                    stop_reason = "iou_threshold_met" if rounds[-1]["iou_finetuned"] >= iou_threshold \
                        else "max_points_reached"

            best_iou   = max(r["iou_finetuned"] for r in rounds)
            best_round = next(r for r in rounds if r["iou_finetuned"] == best_iou)  # earliest tie wins
            overshoot  = best_round["round"] != rounds[-1]["round"]

            # ---- Write everything now that we know which round is best ----
            gt_points_arr = np.array(points_so_far)
            gt_img = draw_prompt_markers(
                colorize_mask_overlay(image, gt, GT_COLOR, alpha),
                gt_points_arr, marker_colors_for(len(gt_points_arr)), alpha=marker_alpha,
            )
            baseline_img = draw_prompt_markers(
                colorize_mask_overlay(image, pred_baseline, BASELINE_COLOR, alpha),
                points1, [POINT1_MARKER_COLOR], alpha=marker_alpha,
            )
            Image.fromarray(baseline_img).save(core_paths["model_baseline"])

            for r in rounds:
                model_path = output_dir / f"{stem}_model_finetuned_{r['round']}.png"
                diff_path  = output_dir / f"{stem}_diff_finetuned_{r['round']}.png"
                Image.fromarray(r["overlay_img"]).save(model_path)
                Image.fromarray(r["diff_img"]).save(diff_path)
                r["path_model"] = str(model_path.relative_to(output_dir))
                r["path_diff"]  = str(diff_path.relative_to(output_dir))

            best_model_path = output_dir / f"{stem}_model_finetuned_best.png"
            best_diff_path  = output_dir / f"{stem}_diff_finetuned_best.png"
            Image.fromarray(best_round["overlay_img"]).save(best_model_path)
            Image.fromarray(best_round["diff_img"]).save(best_diff_path)

            Image.fromarray(gt_img).save(core_paths["ground_truth"])

            counts[f"stop_{stop_reason}"] += 1
            if overshoot:
                counts["overshoot"] += 1

            print(f"  [OK]    {stem}  rounds={len(rounds)}  IoU baseline={iou_baseline:.3f}  "
                  f"best=round{best_round['round']} (IoU={best_round['iou_finetuned']:.3f})  "
                  f"last=round{rounds[-1]['round']} (IoU={rounds[-1]['iou_finetuned']:.3f})  "
                  f"[{stop_reason}]" + ("  [OVERSHOOT]" if overshoot else ""))

            counts["processed"] += 1

            paths = {
                "ground_truth":         str(core_paths["ground_truth"].relative_to(output_dir)),
                "model_baseline":       str(core_paths["model_baseline"].relative_to(output_dir)),
                "model_finetuned":      str(best_model_path.relative_to(output_dir)),
                "diff_finetuned":       str(best_diff_path.relative_to(output_dir)),
                "model_finetuned_best": str(best_model_path.relative_to(output_dir)),
                "diff_finetuned_best":  str(best_diff_path.relative_to(output_dir)),
                "diff_baseline":        None,
            }

            results.append({
                "stem":   stem,
                "status": "processed",

                "iou_baseline":          iou_baseline,
                "score_baseline":        score_baseline,
                "pixel_counts_baseline": counts_baseline,

                "rounds": [
                    {
                        "round":                  r["round"],
                        "num_points":             r["num_points"],
                        "point_added":            r["point_added"],
                        "iou_finetuned":          r["iou_finetuned"],
                        "score_finetuned":        r["score_finetuned"],
                        "pixel_counts_finetuned": r["pixel_counts_finetuned"],
                        "path_model":             r["path_model"],
                        "path_diff":              r["path_diff"],
                    }
                    for r in rounds
                ],

                "best_round": {
                    "round":          best_round["round"],
                    "iou_finetuned":  best_round["iou_finetuned"],
                    "is_last_round":  not overshoot,
                },
                "stop_reason": stop_reason,

                # Compatibility aliases for analyze_comparison_manifest.py / analyze_finetuned_metrics.py:
                # point to the BEST round.
                "prompt_pts":             [p.tolist() for p in points_so_far[:best_round["round"]]],
                "prompt_lbl":             labels_so_far[:best_round["round"]],
                "iou_finetuned":          best_round["iou_finetuned"],
                "score_finetuned":        best_round["score_finetuned"],
                "pixel_counts_finetuned": best_round["pixel_counts_finetuned"],

                "paths": paths,
            })

        except Exception as exc:
            print(f"  [FAIL]  {stem}: {exc}")
            counts["failed"] += 1
            results.append({"stem": stem, "status": "failed", "reason": str(exc)})

    return results, counts


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    output_dir:            Path,
    dataset_dir:           Path,
    baseline_checkpoint:   Path,
    finetuned_checkpoint:  Path,
    baseline_model_cfg:    str,
    finetuned_model_cfg:   str,
    seed:                  int,
    max_points:            int,
    iou_threshold:         float,
    alpha:                 float,
    diff_alpha:            float,
    diff_dim:              float,
    split_path:            Path | None,
    results:               list[dict],
    counts:                dict[str, int],
) -> Path:
    manifest = {
        "dataset_dir":            str(dataset_dir.resolve()),
        "baseline_checkpoint":    str(baseline_checkpoint.resolve()),
        "finetuned_checkpoint":   str(finetuned_checkpoint.resolve()),
        "baseline_model_cfg":     baseline_model_cfg,
        "finetuned_model_cfg":    finetuned_model_cfg,
        "workflow":               "iterative_refinement_npoint",
        "seed":                   seed,
        "max_points":             max_points,
        "iou_threshold":          iou_threshold,
        "alpha":                  alpha,
        "diff_alpha":             diff_alpha,
        "diff_dim":               diff_dim,
        "split_file":             str(split_path.resolve()) if split_path else None,
        "summary":                counts,
        "predictions":            results,
    }
    dest = output_dir / "comparison_manifest.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an N-point iterative refinement comparison (baseline vs. fine-tuned SAM2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",               required=True, type=Path, metavar="PATH")
    parser.add_argument("--baseline-checkpoint",    required=True, type=Path, metavar="PATH")
    parser.add_argument("--finetuned-checkpoint",   required=True, type=Path, metavar="PATH")
    parser.add_argument("--baseline-model-cfg",     default=None,  type=str,  metavar="STR")
    parser.add_argument("--finetuned-model-cfg",    default=None,  type=str,  metavar="STR")
    parser.add_argument("--split",                  default=None,  type=Path, metavar="PATH")
    parser.add_argument("--seed",          default=42,   type=int,   metavar="INT")
    parser.add_argument("--max-points",    default=5,    type=int,   metavar="INT",
                        help="Maximum total points per image, point 1 included. Must be >= 2. Default: 5.")
    parser.add_argument("--iou-threshold", default=0.9,  type=float, metavar="FLOAT",
                        help="Stop adding points once IoU reaches this value. Default: 0.9.")
    parser.add_argument("--output",       required=True, type=Path, metavar="PATH")
    parser.add_argument("--alpha",        default=0.45, type=float, metavar="FLOAT")
    parser.add_argument("--diff-alpha",   default=0.55, type=float, metavar="FLOAT")
    parser.add_argument("--diff-dim",     default=0.5,  type=float, metavar="FLOAT")
    parser.add_argument("--marker-alpha", default=0.75, type=float, metavar="FLOAT")
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--device",       default="cuda", type=str, metavar="STR")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' does not exist.")
    if not args.baseline_checkpoint.exists():
        sys.exit(f"Error: --baseline-checkpoint '{args.baseline_checkpoint}' does not exist.")
    if not args.finetuned_checkpoint.exists():
        sys.exit(f"Error: --finetuned-checkpoint '{args.finetuned_checkpoint}' does not exist.")
    if args.max_points < 2:
        sys.exit("Error: --max-points must be >= 2 (point 2 is always placed).")
    if not (0.0 < args.iou_threshold <= 1.0):
        sys.exit("Error: --iou-threshold must be between 0 (exclusive) and 1 (inclusive).")
    if not (0.0 <= args.alpha <= 1.0):
        sys.exit("Error: --alpha must be between 0 and 1.")
    if not (0.0 <= args.diff_alpha <= 1.0):
        sys.exit("Error: --diff-alpha must be between 0 and 1.")
    if not (0.0 <= args.diff_dim <= 1.0):
        sys.exit("Error: --diff-dim must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    baseline_model_cfg  = args.baseline_model_cfg  or auto_detect_cfg(args.baseline_checkpoint)
    finetuned_model_cfg = args.finetuned_model_cfg or auto_detect_cfg(args.finetuned_checkpoint)

    print(f"Dataset             : {args.dataset}")
    print(f"Baseline checkpoint : {args.baseline_checkpoint}  [{baseline_model_cfg}]")
    print(f"Finetuned checkpoint: {args.finetuned_checkpoint}  [{finetuned_model_cfg}]")
    print(f"Output dir          : {args.output}")
    print(f"Seed                : {args.seed}")
    print(f"Max points          : {args.max_points}")
    print(f"IoU threshold       : {args.iou_threshold}")
    print(f"Device              : {args.device}")
    print()

    entries     = load_manifest(args.dataset)
    stem_filter = load_split_filter(args.split)
    entries     = filter_entries(entries, stem_filter)

    if not entries:
        sys.exit("No entries to process.")

    print(f"Entries to process: {len(entries)}\n")

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"Loading baseline model from {args.baseline_checkpoint}...")
        baseline_sam2      = build_sam2(baseline_model_cfg, str(args.baseline_checkpoint), device=args.device)
        baseline_predictor = SAM2ImagePredictor(baseline_sam2)
        print(f"Loading fine-tuned model from {args.finetuned_checkpoint}...")
        finetuned_sam2      = build_sam2(finetuned_model_cfg, str(args.finetuned_checkpoint), device=args.device)
        finetuned_predictor = SAM2ImagePredictor(finetuned_sam2)
        print("Models loaded.\n")
    else:
        baseline_predictor = finetuned_predictor = None
        print("Dry run — no models loaded, no files will be written.\n")

    rng = np.random.default_rng(args.seed)

    results, counts = run_refinement(
        entries=entries,
        dataset_dir=args.dataset,
        output_dir=args.output,
        baseline_predictor=baseline_predictor,
        finetuned_predictor=finetuned_predictor,
        rng=rng,
        max_points=args.max_points,
        iou_threshold=args.iou_threshold,
        alpha=args.alpha,
        diff_alpha=args.diff_alpha,
        diff_dim=args.diff_dim,
        marker_alpha=args.marker_alpha,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        device=args.device,
    )

    if not args.dry_run:
        manifest_path = write_manifest(
            output_dir=args.output,
            dataset_dir=args.dataset,
            baseline_checkpoint=args.baseline_checkpoint,
            finetuned_checkpoint=args.finetuned_checkpoint,
            baseline_model_cfg=baseline_model_cfg,
            finetuned_model_cfg=finetuned_model_cfg,
            seed=args.seed,
            max_points=args.max_points,
            iou_threshold=args.iou_threshold,
            alpha=args.alpha,
            diff_alpha=args.diff_alpha,
            diff_dim=args.diff_dim,
            split_path=args.split,
            results=results,
            counts=counts,
        )
        print(f"\nManifest written to: {manifest_path}")

    if counts.get("empty_mask", 0) > 0:
        print(f"\n[WARN] {counts['empty_mask']} image(s) had empty GT masks — used image center as fallback.")

    if counts.get("overshoot", 0) > 0:
        print(f"\n[INFO] {counts['overshoot']} image(s) had their best IoU in an earlier round than "
              f"where the process stopped (best_round.is_last_round == false) — extra points didn't help there.")

    print(
        f"\nDone.  Processed: {counts['processed']}  "
        f"(stopped: threshold met={counts['stop_iou_threshold_met']}, "
        f"max points={counts['stop_max_points_reached']}, "
        f"already perfect={counts['stop_perfect_after_round1']})  "
        f"Skipped: {counts['skipped']}  "
        f"Failed: {counts['failed']}"
    )

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
