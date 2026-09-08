"""
run_inference.py

Runs SAM2 inference on a dataset and writes predicted masks to disk.
Predictions are stored in a folder named after the checkpoint, placed
alongside the dataset directory.

Output structure
----------------
<dataset_dir>/../predictions_<checkpoint_stem>/
    <stem>_pred.png     # binary predicted mask (0 / 255)
    ...
    inference_manifest.json   # metadata: checkpoint, prompt mode, seed, per-image results

Usage
-----
    python run_inference.py --dataset <dir> --checkpoint <path> [options]

Options
-------
    --dataset       PATH    Dataset folder containing dataset.json, images/, masks/
                            (required)
    --checkpoint    PATH    Path to SAM2 model checkpoint (.pt file) (required)
    --model-cfg     STR     SAM2 model config name. Default: auto-detected from
                            checkpoint filename. Override if needed, e.g.
                            "configs/sam2.1/sam2.1_hiera_b+.yaml"
    --split         PATH    Optional .txt file of stems to run inference on.
                            If omitted, all entries in dataset.json are used.
    --prompt-mode   STR     How to generate the prompt point. One of:
                              random_fg    (default) random foreground pixel from GT mask
                              gt_centroid  centroid of GT mask foreground region
                              image_center fixed center of the image
    --seed          INT     Random seed for random_fg prompt mode. Default: 42.
    --num-prompts   INT     Number of prompt points per image (1 or 2). Default: 1.
    --output-dir    PATH    Override the default output directory. By default,
                            predictions are written to
                            <dataset_dir>/../predictions_<checkpoint_stem>/
    --overwrite             Overwrite existing prediction files. Default: skip.
    --dry-run               Print what would be run without writing anything.
    --device        STR     Device to run inference on. Default: cuda.

Examples
--------
    # Basic inference with default settings
    python run_inference.py \\
        --dataset ./dataset \\
        --checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt

    # Use GT centroid prompts, val split only
    python run_inference.py \\
        --dataset ./dataset \\
        --checkpoint ./checkpoints/finetuned_step7000.pt \\
        --split ./dataset/val.txt \\
        --prompt-mode gt_centroid

    # Two prompt points, custom seed
    python run_inference.py \\
        --dataset ./dataset \\
        --checkpoint ./checkpoints/finetuned_step7000.pt \\
        --num-prompts 2 \\
        --seed 123
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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
# Model config auto-detection
# ---------------------------------------------------------------------------

# Maps checkpoint filename substrings to config paths
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
    # Fall back to base_plus as the most common fine-tuning target
    print(
        f"[WARN] Could not auto-detect model config from '{checkpoint_path.name}'. "
        f"Falling back to 'configs/sam2.1/sam2.1_hiera_b+.yaml'. "
        f"Use --model-cfg to override if this is wrong."
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
    mask: np.ndarray,
    rng: np.random.Generator,
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
        # Split mask into quadrants, sample one point per quadrant if possible
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

    # Convert (y, x) → (x, y) as SAM2 expects
    points = selected[:, ::-1].copy()
    labels = np.ones(len(points), dtype=np.int32)
    return points, labels


def generate_prompt(
    mode:       str,
    mask:       np.ndarray,
    image:      np.ndarray,
    rng:        np.random.Generator,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Dispatch to the correct prompt strategy."""
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
# Output directory
# ---------------------------------------------------------------------------

def resolve_output_dir(dataset_dir: Path, checkpoint_path: Path,
                       output_dir_override: Path | None) -> Path:
    if output_dir_override is not None:
        return output_dir_override
    # Place predictions/ sibling to dataset dir, named after checkpoint stem
    parent = dataset_dir.resolve().parent
    return parent / f"predictions_{checkpoint_path.stem}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    entries:      list[dict],
    dataset_dir:  Path,
    output_dir:   Path,
    predictor:    "SAM2ImagePredictor",
    prompt_mode:  str,
    rng:          np.random.Generator,
    num_prompts:  int,
    overwrite:    bool,
    dry_run:      bool,
    device:       str,
) -> tuple[list[dict], dict[str, int]]:
    """
    Run inference on all entries. Returns (per_image_results, counts).
    """
    results: list[dict] = []
    counts = {"predicted": 0, "skipped": 0, "failed": 0, "empty_mask": 0}

    for entry in entries:
        stem      = entry["stem"]
        img_path  = dataset_dir / entry["image_path"]
        mask_path = dataset_dir / entry["mask_path"]
        pred_path = output_dir / f"{stem}_pred.png"

        if pred_path.exists() and not overwrite:
            print(f"  [SKIP]  {stem}")
            counts["skipped"] += 1
            results.append({"stem": stem, "status": "skipped",
                             "pred_path": str(pred_path.relative_to(output_dir.parent))})
            continue

        if dry_run:
            print(f"  [DRY]   {stem}  →  {pred_path}")
            counts["predicted"] += 1
            continue

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

            prompt = generate_prompt(prompt_mode, mask, image, rng, num_prompts)

            if prompt is None:
                print(f"  [EMPTY] {stem} — GT mask is empty, using image center")
                prompt = prompt_image_center(image)
                counts["empty_mask"] += 1

            points, labels = prompt

            with torch.inference_mode(), \
                 torch.autocast(device, dtype=torch.bfloat16):
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )

            pred_mask = (masks[0] > 0).astype(np.uint8) * 255
            Image.fromarray(pred_mask, mode="L").save(pred_path)

            print(f"  [OK]    {stem}  (score: {scores[0]:.3f})")
            counts["predicted"] += 1
            results.append({
                "stem":       stem,
                "status":     "predicted",
                "pred_path":  str(pred_path.relative_to(output_dir.parent)),
                "iou_score":  float(scores[0]),
                "prompt_pts": points.tolist(),
                "prompt_lbl": labels.tolist(),
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
    output_dir:      Path,
    checkpoint_path: Path,
    model_cfg:       str,
    prompt_mode:     str,
    seed:            int,
    num_prompts:     int,
    dataset_dir:     Path,
    split_path:      Path | None,
    results:         list[dict],
    counts:          dict[str, int],
) -> Path:
    manifest = {
        "checkpoint":    str(checkpoint_path.resolve()),
        "checkpoint_stem": checkpoint_path.stem,
        "model_cfg":     model_cfg,
        "prompt_mode":   prompt_mode,
        "seed":          seed,
        "num_prompts":   num_prompts,
        "dataset_dir":   str(dataset_dir.resolve()),
        "split_file":    str(split_path.resolve()) if split_path else None,
        "summary":       counts,
        "predictions":   results,
    }
    dest = output_dir / "inference_manifest.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 inference and write predicted masks to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",    required=True,  type=Path, metavar="PATH")
    parser.add_argument("--checkpoint", required=True,  type=Path, metavar="PATH")
    parser.add_argument("--model-cfg",  default=None,   type=str,  metavar="STR",
                        help="SAM2 model config. Auto-detected from checkpoint name if omitted.")
    parser.add_argument("--split",      default=None,   type=Path, metavar="PATH")
    parser.add_argument("--prompt-mode", default="random_fg",
                        choices=["random_fg", "gt_centroid", "image_center"],
                        metavar="STR",
                        help="Prompt strategy: random_fg (default), gt_centroid, image_center.")
    parser.add_argument("--seed",       default=42,     type=int,  metavar="INT",
                        help="Random seed for random_fg prompt mode. Default: 42.")
    parser.add_argument("--num-prompts", default=1,     type=int,  metavar="INT",
                        choices=[1, 2],
                        help="Number of prompt points per image (1 or 2). Default: 1.")
    parser.add_argument("--output-dir", default=None,   type=Path, metavar="PATH",
                        help="Override default output directory.")
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--device",     default="cuda", type=str,  metavar="STR")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        sys.exit(f"Error: --dataset '{args.dataset}' does not exist.")
    if not args.checkpoint.exists():
        sys.exit(f"Error: --checkpoint '{args.checkpoint}' does not exist.")
    if args.num_prompts not in (1, 2):
        sys.exit("Error: --num-prompts must be 1 or 2.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    model_cfg  = args.model_cfg or auto_detect_cfg(args.checkpoint)
    output_dir = resolve_output_dir(args.dataset, args.checkpoint, args.output_dir)

    print(f"Dataset      : {args.dataset}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Model config : {model_cfg}")
    print(f"Output dir   : {output_dir}")
    print(f"Prompt mode  : {args.prompt_mode}")
    print(f"Seed         : {args.seed}")
    print(f"Num prompts  : {args.num_prompts}")
    print(f"Device       : {args.device}")
    print()

    entries     = load_manifest(args.dataset)
    stem_filter = load_split_filter(args.split)
    entries     = filter_entries(entries, stem_filter)

    if not entries:
        sys.exit("No entries to process.")

    print(f"Entries to process: {len(entries)}\n")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading model from {args.checkpoint}...")
        sam2      = build_sam2(model_cfg, str(args.checkpoint), device=args.device)
        predictor = SAM2ImagePredictor(sam2)
        print("Model loaded.\n")
    else:
        predictor = None
        print("Dry run — no model loaded, no files will be written.\n")

    rng = np.random.default_rng(args.seed)

    results, counts = run_inference(
        entries=entries,
        dataset_dir=args.dataset,
        output_dir=output_dir,
        predictor=predictor,
        prompt_mode=args.prompt_mode,
        rng=rng,
        num_prompts=args.num_prompts,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        device=args.device,
    )

    if not args.dry_run:
        manifest_path = write_manifest(
            output_dir=output_dir,
            checkpoint_path=args.checkpoint,
            model_cfg=model_cfg,
            prompt_mode=args.prompt_mode,
            seed=args.seed,
            num_prompts=args.num_prompts,
            dataset_dir=args.dataset,
            split_path=args.split,
            results=results,
            counts=counts,
        )
        print(f"\nManifest written to: {manifest_path}")

    if counts.get("empty_mask", 0) > 0:
        print(f"\n[WARN] {counts['empty_mask']} image(s) had empty GT masks — used image center as fallback.")

    print(
        f"\nDone.  Predicted: {counts['predicted']}  "
        f"Skipped: {counts['skipped']}  "
        f"Failed: {counts['failed']}"
    )

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()