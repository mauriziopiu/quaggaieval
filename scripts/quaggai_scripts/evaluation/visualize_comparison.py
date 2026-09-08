"""
visualize_comparison.py

Runs SAM2 inference with two checkpoints (typically a baseline and a
fine-tuned model) on the same labelled dataset and writes per-image
overlay visualizations for qualitative comparison.

For each dataset entry, the SAME prompt point(s) are used for both
checkpoints so the two predictions are directly comparable. Five PNG
files are written per image, all sharing the entry's stem:

    <stem>_ground_truth.png    image + GT mask overlay (green)
    <stem>_model_baseline.png  image + baseline prediction overlay (red)
    <stem>_model_finetuned.png image + fine-tuned prediction overlay (blue)
    <stem>_diff_baseline.png   TP/FP/FN error map for the baseline model
    <stem>_diff_finetuned.png  TP/FP/FN error map for the fine-tuned model

All overlay images additionally show the prompt point(s) as small
semi-transparent markers (except the diff images, which are left
uncluttered).

Diff color scheme (colorblind-safe): TP=blue, FP=orange, FN=yellow, on
top of the original image dimmed to make the error regions stand out.

Output structure
-----------------
<output>/
    <stem>_ground_truth.png
    <stem>_model_baseline.png
    <stem>_model_finetuned.png
    <stem>_diff_baseline.png
    <stem>_diff_finetuned.png
    ...
    comparison_manifest.json   metadata + per-image prompts, IoU, pixel counts

Usage
-----
    python visualize_comparison.py \\
        --dataset             ./dataset \\
        --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \\
        --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \\
        --output              ./comparison_visuals

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
    --prompt-mode           STR     random_fg (default) | gt_centroid | image_center
    --seed                  INT     Random seed for random_fg prompt mode. Default: 42.
    --num-prompts           INT     Number of prompt points per image (1 or 2). Default: 1.
    --output                PATH    Folder to write overlay images + manifest (required)
    --alpha                 FLOAT   Overlay alpha for GT/baseline/finetuned masks. Default: 0.45
    --diff-alpha             FLOAT   Overlay alpha for TP/FP/FN regions in diff images. Default: 0.55
    --diff-dim              FLOAT   Brightness factor for the dimmed background in diff
                                     images (0-1). Default: 0.5
    --marker-alpha          FLOAT   Alpha for the prompt point marker. Default: 0.75
    --overwrite             Overwrite existing output files. Default: skip.
    --dry-run               Print what would be run without writing anything.
    --device                STR     Device to run inference on. Default: cuda.

Examples
--------
    # Basic comparison with default random_fg prompts
    python visualize_comparison.py \\
        --dataset ./dataset \\
        --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \\
        --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \\
        --output ./comparison_visuals

    # Restrict to val split, use GT centroid prompts for determinism
    python visualize_comparison.py \\
        --dataset ./dataset \\
        --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \\
        --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \\
        --split ./dataset/val.txt \\
        --prompt-mode gt_centroid \\
        --output ./comparison_visuals
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
# Prompt generation strategies
# ---------------------------------------------------------------------------

def prompt_gt_centroid(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Centroid of the foreground region. Returns None if mask is empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = int(xs.mean()), int(ys.mean())
    return np.array([[cx, cy]]), np.array([1])


def prompt_image_center(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed center of the image."""
    h, w = image.shape[:2]
    return np.array([[w // 2, h // 2]]), np.array([1])


def prompt_random_fg(
    mask:       np.ndarray,
    rng:        np.random.Generator,
    num_points: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sample num_points random foreground pixels.
    Returns None if mask is empty.
    When num_points > 1, attempts to sample from spatially distinct regions.
    """
    fg_pixels = np.argwhere(mask > 0)  # (N, 2) in (y, x)
    if len(fg_pixels) == 0:
        return None

    if num_points == 1 or len(fg_pixels) < num_points:
        indices = rng.choice(len(fg_pixels), size=min(num_points, len(fg_pixels)),
                             replace=False)
        selected = fg_pixels[indices]
    else:
        h, w = mask.shape
        quadrants = []
        for qy in [(0, h // 2), (h // 2, h)]:
            for qx in [(0, w // 2), (w // 2, w)]:
                q_pixels = fg_pixels[
                    (fg_pixels[:, 0] >= qy[0]) & (fg_pixels[:, 0] < qy[1]) &
                    (fg_pixels[:, 1] >= qx[0]) & (fg_pixels[:, 1] < qx[1])
                ]
                if len(q_pixels) > 0:
                    quadrants.append(q_pixels)

        if len(quadrants) >= num_points:
            selected = np.array([
                q[rng.integers(len(q))] for q in quadrants[:num_points]
            ])
        else:
            indices  = rng.choice(len(fg_pixels), size=num_points, replace=False)
            selected = fg_pixels[indices]

    points = selected[:, ::-1].copy()  # (y, x) → (x, y)
    labels = np.ones(len(points), dtype=np.int32)
    return points, labels


def generate_prompt(
    mode:       str,
    mask:       np.ndarray,
    image:      np.ndarray,
    rng:        np.random.Generator,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if mode == "gt_centroid":
        return prompt_gt_centroid(mask)
    elif mode == "image_center":
        return prompt_image_center(image)
    elif mode == "random_fg":
        return prompt_random_fg(mask, rng, num_points)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")


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
    radius: int | None = None,
    alpha:  float = 0.75,
) -> np.ndarray:
    """Draw small semi-transparent circular markers at each prompt point."""
    h, w = image.shape[:2]
    if radius is None:
        radius = max(3, min(h, w) // 150)

    base    = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    fill_a  = int(255 * alpha)

    for x, y in points:
        bbox = [int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius]
        draw.ellipse(bbox, fill=(255, 255, 255, fill_a), outline=(0, 0, 0, fill_a))

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

def run_comparison(
    entries:              list[dict],
    dataset_dir:           Path,
    output_dir:            Path,
    baseline_predictor:    "SAM2ImagePredictor",
    finetuned_predictor:   "SAM2ImagePredictor",
    prompt_mode:           str,
    rng:                   np.random.Generator,
    num_prompts:           int,
    alpha:                 float,
    diff_alpha:            float,
    diff_dim:              float,
    marker_alpha:          float,
    overwrite:             bool,
    dry_run:               bool,
    device:                str,
) -> tuple[list[dict], dict[str, int]]:
    results: list[dict] = []
    counts = {"processed": 0, "skipped": 0, "failed": 0, "empty_mask": 0}

    for entry in entries:
        stem      = entry["stem"]
        img_path  = dataset_dir / entry["image_path"]
        mask_path = dataset_dir / entry["mask_path"]

        out_paths = {
            "ground_truth":    output_dir / f"{stem}_ground_truth.png",
            "model_baseline":  output_dir / f"{stem}_model_baseline.png",
            "model_finetuned": output_dir / f"{stem}_model_finetuned.png",
            "diff_baseline":   output_dir / f"{stem}_diff_baseline.png",
            "diff_finetuned":  output_dir / f"{stem}_diff_finetuned.png",
        }

        if not overwrite and all(p.exists() for p in out_paths.values()):
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

            prompt = generate_prompt(prompt_mode, gt.astype(np.uint8) * 255, image, rng, num_prompts)
            if prompt is None:
                print(f"  [EMPTY] {stem} — GT mask is empty, using image center")
                prompt = prompt_image_center(image)
                counts["empty_mask"] += 1
            points, labels = prompt

            pred_baseline,  score_baseline  = predict_mask(baseline_predictor,  image, points, labels, device)
            pred_finetuned, score_finetuned = predict_mask(finetuned_predictor, image, points, labels, device)

            gt_img        = draw_prompt_markers(colorize_mask_overlay(image, gt,              GT_COLOR,        alpha), points, alpha=marker_alpha)
            baseline_img  = draw_prompt_markers(colorize_mask_overlay(image, pred_baseline,    BASELINE_COLOR,  alpha), points, alpha=marker_alpha)
            finetuned_img = draw_prompt_markers(colorize_mask_overlay(image, pred_finetuned,    FINETUNED_COLOR, alpha), points, alpha=marker_alpha)
            diff_baseline_img  = build_diff_overlay(image, pred_baseline,  gt, diff_dim, diff_alpha)
            diff_finetuned_img = build_diff_overlay(image, pred_finetuned, gt, diff_dim, diff_alpha)

            Image.fromarray(gt_img).save(out_paths["ground_truth"])
            Image.fromarray(baseline_img).save(out_paths["model_baseline"])
            Image.fromarray(finetuned_img).save(out_paths["model_finetuned"])
            Image.fromarray(diff_baseline_img).save(out_paths["diff_baseline"])
            Image.fromarray(diff_finetuned_img).save(out_paths["diff_finetuned"])

            iou_baseline  = compute_iou(pred_baseline,  gt)
            iou_finetuned = compute_iou(pred_finetuned, gt)

            print(f"  [OK]    {stem}  (IoU baseline={iou_baseline:.3f}  finetuned={iou_finetuned:.3f})")
            counts["processed"] += 1
            results.append({
                "stem":               stem,
                "status":             "processed",
                "prompt_pts":         points.tolist(),
                "prompt_lbl":         labels.tolist(),
                "iou_baseline":       iou_baseline,
                "iou_finetuned":      iou_finetuned,
                "score_baseline":     score_baseline,
                "score_finetuned":    score_finetuned,
                "pixel_counts_baseline":  pixel_counts(pred_baseline,  gt),
                "pixel_counts_finetuned": pixel_counts(pred_finetuned, gt),
                "paths": {k: str(v.relative_to(output_dir)) for k, v in out_paths.items()},
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
    prompt_mode:           str,
    seed:                  int,
    num_prompts:           int,
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
        "prompt_mode":            prompt_mode,
        "seed":                   seed,
        "num_prompts":            num_prompts,
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
        description="Run baseline vs. fine-tuned SAM2 inference and write comparison overlay visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",               required=True, type=Path, metavar="PATH")
    parser.add_argument("--baseline-checkpoint",    required=True, type=Path, metavar="PATH")
    parser.add_argument("--finetuned-checkpoint",   required=True, type=Path, metavar="PATH")
    parser.add_argument("--baseline-model-cfg",     default=None,  type=str,  metavar="STR")
    parser.add_argument("--finetuned-model-cfg",    default=None,  type=str,  metavar="STR")
    parser.add_argument("--split",                  default=None,  type=Path, metavar="PATH")
    parser.add_argument("--prompt-mode", default="random_fg",
                        choices=["random_fg", "gt_centroid", "image_center"], metavar="STR")
    parser.add_argument("--seed",         default=42,   type=int,   metavar="INT")
    parser.add_argument("--num-prompts",  default=1,    type=int,   choices=[1, 2], metavar="INT")
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
    print(f"Prompt mode         : {args.prompt_mode}")
    print(f"Seed                : {args.seed}")
    print(f"Num prompts         : {args.num_prompts}")
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

    results, counts = run_comparison(
        entries=entries,
        dataset_dir=args.dataset,
        output_dir=args.output,
        baseline_predictor=baseline_predictor,
        finetuned_predictor=finetuned_predictor,
        prompt_mode=args.prompt_mode,
        rng=rng,
        num_prompts=args.num_prompts,
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
            prompt_mode=args.prompt_mode,
            seed=args.seed,
            num_prompts=args.num_prompts,
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

    print(
        f"\nDone.  Processed: {counts['processed']}  "
        f"Skipped: {counts['skipped']}  "
        f"Failed: {counts['failed']}"
    )

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
