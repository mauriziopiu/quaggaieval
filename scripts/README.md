# scripts

This folder contains scripts contributing to model training, fine-tuning and data processing,
organized as an installable package (`quaggai_scripts`) under `preprocessing/`, `training/`, and
`evaluation/`. See the repo root README for how this fits alongside `quaggaieval_desktop/` and
`eval_viewer/`.

## Tasklist

- Alpha Mask Extraction: export binary masks from semi-transparent images
- Density Class Estimation: for proper dataset balancing, estimate density in sample image
- Dataset Generation: finding pairs of filenames and also those that do not match and arrange in dataset structure
- Evaluation Script: how good is a model? (to be used for baseline reference)
- Dataset Augmentation?
- Training Script: execute model training

## Model Training Process Outline

A brief outline of relevant preprocessing steps to generate dataset suitable and ready for training.

Assumed Input

- (Folder) PNG/JPG/... Image files representing baseline image capture. Often these are the `*_cropped.jpg` images.
- (Folder) PNG Image files containing only coverage mask pixels (full-color!) and transparency everywhere else.
- The above files should be named so that the filename stem (without suffix), will determine matching pairs of (baseline, coverage)
    - example: "lake_bed_capture_3691_cropped.jpg" will match with "lake_bed_capture_3691_area.png"
- Matching pairs should have exactly the same dimensions in order to ensure a proper GT match.

Preprocessing

- Binary Mask Generation:
    - from the png images with partial transparency, a binary mask needs to be extracted. For this we have the `png_to_mask.py` script as described in more detail below.
- Build Dataset:
    - create a basic dataset folder structure
- (Optional) Analyse Density Distribution:
    - analyse the distribution of dense and sparsely populated masks in the dataset (in order to avoid strong bias)
- Split Dataset:
    - for training and validation
- Format Dataset in SA-1B:
    - bring the dataset into meta's required dataset format for model finetuning.
- Rename files for training:
    - renames files so that single common stem remains and then only a counter as suffix.
    - not ideal but required for training.
- Convert png to jpg for training:
    - convert baseline image captures to jpg for (required) file format consistency in model training.

## Data Preprocessing Scripts

### batch_preprocess.py

Batch preprocessing orchestrator for the QuaggAI dataset pipeline.

Chains all preprocessing steps in order, driven by a single YAML config file -> see `configs/batch_config.yaml`

Pipeline:

1. png_to_mask — alpha-channel PNG → binary mask PNG
2. png_to_jpg — baseline PNG → JPEG
3. build_dataset — match image/mask pairs, build structured dataset/
4. split_dataset — produce train.txt + val.txt
5. convert_to_sa1b — convert to SA-1B JSON annotation format
6. rename_to_sa1b — rename files to numeric convention (sa_000001.\*)

Usage:

```bash
python -m quaggai_scripts.preprocessing.batch_preprocess --config configs/batch_config.yaml
```

### png_to_mask.py

Converts PNG images with transparency into binary segmentation masks.
Transparent pixels → black (background, 0)
Opaque pixels → white (foreground, 255)

Usage:
`python -m quaggai_scripts.preprocessing.png_to_mask --input <input_dir> --output <output_dir> [options]`

Options:

- `--input PATH`
    - Folder containing source PNG images (required)
- `--output PATH`
    - Folder to write binary mask PNGs (required)
- `--alpha-threshold INT`
    - Alpha value below which a pixel is considered transparent.
    - Default: 128. Range: 0–255.
    - Lower = more pixels treated as foreground.
- `--suffix STR`
    - Suffix appended to output filenames before extension.
    - Default: "\_mask". Set to "" to keep original names.
- `--recursive`
    - If set, search input folder recursively for PNGs.
- `--overwrite`
    - If set, allow overwriting existing files in output folder.
    - By default, existing files are skipped with a warning.

Examples:

```bash
# Basic usage
python -m quaggai_scripts.preprocessing.png_to_mask --input ./raw_pngs --output ./masks

# Custom alpha threshold and no suffix
python -m quaggai_scripts.preprocessing.png_to_mask --input ./raw_pngs --output ./masks --alpha-threshold 10 --suffix ""

# Recursive search, overwrite existing outputs
python -m quaggai_scripts.preprocessing.png_to_mask --input ./dataset --output ./masks --recursive --overwrite
```

### build_dataset.py

Scans an image folder and a mask folder, matches pairs by base stem,
and consolidates them into a structured dataset directory with a manifest.

**Matching rules**

Images:

- extensions .jpg .jpeg .png .tiff .tif are considered.
- Known image suffixes stripped to recover base stem: \_cropped
- If none match, the full filename stem is used as-is.

Masks :

- only .png files are considered.
- Known mask suffixes stripped to recover base stem (longest first): \_area_mask, \_mask

A pair is matched when both sides resolve to the same base stem.

**Output structure**

```
<output_dir>/
    images/
        <stem>.<ext> # copied from source
        ...
    masks/
        <stem>\_mask.png # copied from source, renamed to canonical form
        ...
    dataset.json # manifest with one entry per matched pair
```

**dataset.json schema**

```json
{
    "total_pairs": <int>,
    "entries": [
        {
            "stem": "<base stem>",
            "image_path": "images/<stem>.<ext>", # relative to output_dir
            "mask_path": "masks/<stem>_mask.png", # relative to output_dir
            "image_source": "<absolute source path>",
            "mask_source": "<absolute source path>"
        },
        ...
    ]
}
```

unmatched_report.json schema (written when --unmatched-report is set)

```json
{
    "images_without_mask": [
        {"stem": "...", "source_path": "..."},
        ...
    ],
    "masks_without_image": [
        {"stem": "...", "source_path": "..."},
        ...
    ],
    "stem_conflicts": {
        "images": ["Stem conflict 'foo': keeping foo.jpg, ignoring foo_cropped.jpg", ...],
        "masks":  ["Stem conflict 'bar': keeping bar_mask.png, ignoring bar_area_mask.png", ...]
    },
    "copy_failures": [
        {"stem": "...", "reason": "..."},
        ...
    ],
    "summary": {
        "images_without_mask":   <int>,
        "masks_without_image":   <int>,
        "stem_conflicts_images": <int>,
        "stem_conflicts_masks":  <int>,
        "copy_failures":         <int>
    }
}
```

Usage

```bash
python -m quaggai_scripts.preprocessing.build_dataset --images <dir> --masks <dir> --output <dir> [options]
```

Options

- `--images PATH` Folder containing source images (required)
- `--masks PATH` Folder containing source masks (required)
- `--output PATH `Destination dataset folder (required)
- `--unmatched-report PATH`
    - Write a JSON report of all unmatched/failed files to this path.
    - Optional: issues are only printed to console if omitted.
- `--overwrite` Overwrite existing output files. Default: skip with warning.
- `--dry-run` Print matched pairs and exit without writing anything.

Examples

```bash
python -m quaggai_scripts.preprocessing.build_dataset --images ./raw_images --masks ./raw_masks --output ./dataset

python -m quaggai_scripts.preprocessing.build_dataset --images ./raw_images --masks ./raw_masks --output ./dataset --unmatched-report ./unmatched_report.json

python -m quaggai_scripts.preprocessing.build_dataset --images ./raw_images --masks ./raw_masks --output ./dataset --dry-run
```

### analyse_density.py

Non-invasive analysis of foreground coverage distribution within a dataset.

Reads masks from a dataset produced by build_dataset.py (via dataset.json), computes per-image foreground pixel coverage, bins results into fixed 10% buckets, and writes:

```
<output_dir>/
    density_report.json   -- per-image coverage stats + aggregate summary
    density_plot.png      -- histogram of coverage distribution
```

Coverage is defined as the fraction of pixels in a mask that are foreground
(pixel value > 0), expressed as a percentage in [0, 100].

Usage

```bash
python -m quaggai_scripts.preprocessing.analyse_density --dataset <dir> --output <dir> [options]
```

Options

- `--dataset PATH` Dataset folder containing dataset.json and masks/ (required)
- `--output PATH` Folder to write density_report.json and density_plot.png (required)
- `--overwrite` Overwrite existing output files. Default: skip with warning.

Examples

```bash
python -m quaggai_scripts.preprocessing.analyse_density --dataset ./dataset --output ./analysis
```

### split_dataset.py

tbd, very simple tho

### convert_to_sa1b.py

Converts a dataset produced by build_dataset.py into SA-1B format, as
required by the SAM2 official training code (SA1BRawDataset).

SA-1B format places a JSON annotation file next to each image:

    <output_dir>/
        images/
            <stem>.<ext>        # copied from source dataset
            <stem>.json         # generated annotation in SA-1B schema
        train.txt               # copied from source dataset (if found)
        val.txt                 # copied from source dataset (if found)

SA-1B JSON schema (fields required by SA1BRawDataset)

```json
{
    "image": {
        "image_id": <int>,
        "width": <int>,
        "height": <int>
    },
    "annotations": [
        {
            "id": 1,
            "segmentation": {
                "size": [height, width],
                "counts": "<RLE-encoded binary mask string>"
            },
            "area": <int>,
            "bbox": [x, y, w, h],
            "predicted_iou": 1.0,
            "stability_score": 1.0,
            "crop_box": [0, 0, width, height]
        }
    ]
}
```

The mask is encoded using COCO RLE (pycocotools). Only pixels with value > 0
in the source mask PNG are treated as foreground.

Usage

```bash
    python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset <dir> --output <dir> [options]
```

Options

- `--dataset PATH` Dataset folder containing dataset.json, images/, masks/ (required)
- `--output PATH` Destination folder for SA-1B formatted data (required)
- `--split PATH`
    - Optional path to a .txt file listing stems to process (one stem per line).
    - If omitted, all entries in dataset.json are converted.
- `--fg-threshold INT`
    - Pixel value threshold above which a pixel is foreground.
    - Default: 0 (any non-zero pixel = foreground).
- `--overwrite`
    - Overwrite existing output files.
    - Default: skip with warning.
- `--dry-run` Print what would be converted without writing anything.
- `--copy-splits`
    - Copy train.txt / val.txt from dataset dir to output dir if found.
    - Default: true (pass --no-copy-splits to disable).

Examples

```bash
# Convert full dataset
python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset ./dataset --output ./dataset_sa1b

# Convert only the training split
python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset ./dataset --output ./dataset_sa1b --split ./dataset/train.txt

# Convert validation split into a separate output folder
python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset ./dataset --output ./dataset_sa1b_val --split ./dataset/val.txt

# Dry run to verify matching before writing
python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset ./dataset --output ./dataset_sa1b --dry-run

# * Recommended Approach: Run Conversion Separately for each Split * #

# Train split
python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset ./dataset --output ./sa1b_train \
    --split ./dataset/train.txt

# Val split
python -m quaggai_scripts.preprocessing.convert_to_sa1b --dataset ./dataset --output ./sa1b_val \
    --split ./dataset/val.txt
```

### rename_to_sa1b.py 📍

### png_to_jpg.py 📍

## Training 📍

### model_finetune.yaml 📍

Hydra Config

### launch_training.py 📍

Training Script

## Inference and Evaluation Scripts

### run_inference.py

Runs SAM2 inference on a dataset and writes predicted masks to disk.
Predictions are stored in a folder named after the checkpoint, placed
alongside the dataset directory.

Output structure

```
<dataset_dir>/../predictions_<checkpoint_stem>/
    <stem>_pred.png     # binary predicted mask (0 / 255)
    ...
    inference_manifest.json   # metadata: checkpoint, prompt mode, seed, per-image results
```

Usage

```bash
python -m quaggai_scripts.evaluation.run_inference --dataset <dir> --checkpoint <path> [options]
```

Options

- `--dataset PATH`
    - Dataset folder containing dataset.json, images/, masks/ (required)
- `--checkpoint PATH`
    - Path to SAM2 model checkpoint (.pt file) (required)
- `--model-cfg STR`
    - SAM2 model config name.
    - Default: auto-detected from checkpoint filename.
    - Override if needed, e.g. "configs/sam2.1/sam2.1_hiera_b+.yaml"
- `--split PATH`
    - Optional .txt file of stems to run inference on.
    - If omitted, all entries in dataset.json are used.
- `--prompt-mode STR`
    - How to generate the prompt point. One of:
        - `random_fg` (default) random foreground pixel from GT mask
        - `gt_centroid` centroid of GT mask foreground region
        - `image_center` fixed center of the image
- `--seed INT`
    - Random seed for random_fg prompt mode.
    - Default: 42.
- `--num-prompts INT`
    - Number of prompt points per image (1 or 2).
    - Default: 1.
- `--output-dir PATH`
    - Override the default output directory.
    - By default, predictions are written to `<dataset*dir>/../predictions*<checkpoint_stem>/`
- `--overwrite`
    - Overwrite existing prediction files.
    - Default: skip.
- `--dry-run`
    - Print what would be run without writing anything.
- `--device STR`
    - Device to run inference on.
    - Default: cuda.

Examples

```bash
# Basic inference with default settings
python -m quaggai_scripts.evaluation.run_inference \\
    --dataset ./dataset \\
    --checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt

# Use GT centroid prompts, val split only
python -m quaggai_scripts.evaluation.run_inference \\
    --dataset ./dataset \\
    --checkpoint ./checkpoints/finetuned_step7000.pt \\
    --split ./dataset/val.txt \\
    --prompt-mode gt_centroid

# Two prompt points, custom seed
python -m quaggai_scripts.evaluation.run_inference \\
    --dataset ./dataset \\
    --checkpoint ./checkpoints/finetuned_step7000.pt \\
    --num-prompts 2 \\
    --seed 123
```

### evaluate.py

Evaluates predicted segmentation masks against ground-truth masks and
produces a JSON report and PNG plots.

Supports single-checkpoint evaluation and optional side-by-side comparison
of two checkpoints (e.g. baseline vs. fine-tuned).

Metrics computed

- IoU Intersection over Union (primary metric)
- Dice Dice coefficient (F1 score on pixels)
- Boundary F1 Precision/recall at mask boundaries (tolerance configurable)
- False Positive Rate Fraction of background pixels predicted as foreground
- False Negative Rate Fraction of foreground pixels predicted as background

All metrics are computed per image, then aggregated (mean, std, median, min, max)
and also broken down by coverage bucket (10% intervals, same as analyse_density.py).

Output structure

```
<output_dir>/
    evaluation_report.json      always written
    metrics_distributions.png   histograms of IoU / Dice / Boundary F1
    coverage_vs_iou.png         IoU per 10% coverage bucket
    visual_samples.png          grid of GT vs predicted masks
                                (best / worst / median N cases)
    comparison_report.json      written only in comparison mode
    comparison_metrics.png      side-by-side metric comparison plot
    comparison_visual_samples.png  GT / pred_A / pred_B grid
```

Usage

```bash
# Single checkpoint evaluation
python -m quaggai_scripts.evaluation.evaluate \\
    --dataset  ./dataset \\
    --preds    ./predictions_sam2.1_hiera_base_plus \\
    --output   ./evaluation

# Comparison mode
python -m quaggai_scripts.evaluation.evaluate \\
    --dataset  ./dataset \\
    --preds    ./predictions_sam2.1_hiera_base_plus \\
    --preds-b  ./predictions_finetuned_step7000 \\
    --output   ./evaluation
```

Options

- `--dataset PATH`
    - Dataset folder with dataset.json and masks/ (required)
- `--preds PATH`
    - Predictions folder (output of run_inference.py) (required)
- ` --preds-b PATH`
    - Second predictions folder for comparison mode (optional)
- `--output PATH`
    - Folder to write reports and plots (required)
- `--split PATH`
    - Optional .txt file to restrict evaluation to a subset of stems
- `--boundary-tol INT`
    - Boundary F1 tolerance in pixels. Default: 2
- `--num-samples INT`
    - Number of best/worst/median visual samples. Default: 5
- `--label-a STR`
    - Label for first predictions in comparison plots. Default: auto
- `--label-b STR`
    - Label for second predictions in comparison plots. Default: auto
- `--overwrite`
    - Overwrite existing output files

### visualize_comparison.py

Runs SAM2 inference with two checkpoints (typically a baseline and a
fine-tuned model) on the same labelled dataset and writes per-image
overlay visualizations for qualitative, side-by-side inspection.

The same prompt point(s) are used for both checkpoints per image so the
two predictions are directly comparable. Five PNGs are written per
image, all sharing the entry's stem:

```
<output>/
    <stem>_ground_truth.png    image + GT mask overlay (green)
    <stem>_model_baseline.png  image + baseline prediction overlay (red)
    <stem>_model_finetuned.png image + fine-tuned prediction overlay (blue)
    <stem>_diff_baseline.png   TP/FP/FN error map for the baseline model
    <stem>_diff_finetuned.png  TP/FP/FN error map for the fine-tuned model
    ...
    comparison_manifest.json   metadata + per-image prompts, IoU, pixel counts
```

All overlay images also show the prompt point(s) as small semi-transparent
markers (omitted on the diff images to keep them uncluttered).

Diff color scheme is colorblind-safe: TP=blue, FP=orange, FN=yellow, on top
of the original image dimmed to make error regions stand out.

Usage

```bash
python -m quaggai_scripts.evaluation.visualize_comparison \
    --dataset              <dir> \
    --baseline-checkpoint  <path> \
    --finetuned-checkpoint <path> \
    --output               <dir> \
    [options]
```

Options

- `--dataset PATH` Dataset folder with dataset.json, images/, masks/ (required)
- `--baseline-checkpoint PATH` Baseline SAM2 checkpoint (.pt) (required)
- `--finetuned-checkpoint PATH` Fine-tuned SAM2 checkpoint (.pt) (required)
- `--baseline-model-cfg STR` / `--finetuned-model-cfg STR`
    - SAM2 model config for each checkpoint. Default: auto-detected from filename.
- `--split PATH` Optional .txt file of stems to process. Default: all entries.
- `--prompt-mode STR` `random_fg` (default) | `gt_centroid` | `image_center`
- `--seed INT` Random seed for random_fg prompt mode. Default: 42.
- `--num-prompts INT` Number of prompt points per image (1 or 2). Default: 1.
- `--output PATH` Folder to write overlay images + manifest (required)
- `--alpha FLOAT` Overlay alpha for GT/baseline/finetuned masks. Default: 0.45
- `--diff-alpha FLOAT` Overlay alpha for TP/FP/FN regions in diff images. Default: 0.55
- `--diff-dim FLOAT` Brightness factor for dimmed background in diff images. Default: 0.5
- `--marker-alpha FLOAT` Alpha for the prompt point marker. Default: 0.75
- `--overwrite` Overwrite existing output files. Default: skip.
- `--dry-run` Print what would be run without writing anything.
- `--device STR` Device to run inference on. Default: cuda.

Example

```bash
python -m quaggai_scripts.evaluation.visualize_comparison \
    --dataset ./dataset \
    --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
    --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \
    --split ./dataset/val.txt \
    --output ./comparison_visuals
```

Numeric IoU/Dice aggregation, distribution plots and coverage-bucket
breakdowns are intentionally out of scope here — use `evaluate.py` for
that. The manifest's per-image IoU and pixel counts are included mainly
so future analysis doesn't require re-running inference.

### analyze_comparison_manifest.py

Reads a `comparison_manifest.json` (output of `visualize_comparison.py`)
and produces IoU/coverage and confusion-matrix plots, without re-running
inference or touching any image/mask files.

Worst/median/best selection is based on the **fine-tuned** model's IoU
(baseline IoU shown alongside for context).

Output structure

```
<output>/
    coverage_vs_iou_finetuned.png                bucketed mean IoU (bar + jittered strip of
                                                  individual samples) + scatter, worst/median/
                                                  best highlighted
    confusion_matrices_worst_median_best.png
                                                  baseline vs fine-tuned 2x2 confusion
                                                  matrices (% of image area) per selected
                                                  case, labelled with stem + diff image
                                                  filenames for cross-reference
    selection_summary.json                       which stems were selected per group
    refinement_trajectory.png                    only if the manifest has `rounds` data (from
                                                  visualize_iterative_refinement.py) — per-image
                                                  IoU trajectory across every round, green=improved
                                                  beyond round 1 / gray=unchanged, best round marked
                                                  with a star, any continuation past the best round
                                                  drawn as a dashed red overshoot segment, + a
                                                  Δ(round1→best) histogram
    refinement_best_round_by_coverage.png        only if refinement data is present — box-and-
                                                  whisker + jittered strip of which round produced
                                                  the best IoU (1..max_points), per coverage bucket
    refinement_coverage_vs_iou_round1.png        only if refinement data is present — same
                                                  bucketed mean-IoU (bar + jittered strip) panel
                                                  as coverage_vs_iou_finetuned.png's left side,
                                                  but using ROUND 1 (first point only) IoU instead
                                                  of the best round — a before/after comparison
    refinement_summary.json                      only if refinement data is present — per-image
                                                  round trajectory, delta from round 1 to best,
                                                  and an explicit, worst-first overshoot list
```

If the manifest comes from `visualize_iterative_refinement.py`, this
script automatically also runs the round-trajectory analysis above —
nothing extra to pass. Manifests from `visualize_comparison.py` (no
`rounds` field) produce exactly the same output as before, unaffected.

Usage

```bash
python -m quaggai_scripts.evaluation.analyze_comparison_manifest \
    --manifest <path to comparison_manifest.json> \
    --output   <dir>
```

Options

- `--manifest PATH` Path to comparison_manifest.json (required)
- `--output PATH` Folder to write plots + summary (required)
- `--num-samples INT` Number of worst/median/best cases per group. Default: 5.
- `--overwrite` Overwrite existing output files. Default: skip.

Example

```bash
python -m quaggai_scripts.evaluation.analyze_comparison_manifest \
    --manifest ./comparison_visuals/comparison_manifest.json \
    --output   ./comparison_analysis
```

### analyze_finetuned_metrics.py

Reads a `comparison_manifest.json` (output of `visualize_comparison.py`)
and computes per-image pixel-level metrics for the **fine-tuned model
only**, writing a single JSON report. Does not re-run inference or touch
any image/mask files.

Per image, TP/TN/FP/FN are expressed as a percentage of the image's total
pixel count (summing to 100%), alongside precision, recall, accuracy and
F1 score computed from the raw pixel counts.

If the manifest comes from `visualize_iterative_refinement.py` (has a
`rounds` list per entry), each per-image entry also gets a full
`round_trajectory`, the winning `best_round`, and `delta_round1_to_best`
— how much the corrective points changed IoU overall relative to the
single-point round 1. The report gains a top-level `refinement` block
summarizing this across the dataset, including an explicit **overshoot**
list: images where a later round ended up worse than the best one
reached along the way (i.e. the process kept adding points past its own
peak). This section is omitted entirely for manifests without round
data — output is unchanged otherwise.

Output JSON structure

```json
{
    "source_manifest": "<path>",
    "total_evaluated": 97,
    "excluded": {"failed": 0, "skipped": 0},
    "average": {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0},   // macro-average, best round
    "rankings": {
        "tp_pct":    {"highest": {...}, "median": {...}, "lowest": {...}},
        "tn_pct":    {...},
        "fp_pct":    {...},
        "fn_pct":    {...},
        "precision": {...},
        "recall":    {...},
        "accuracy":  {...},
        "f1":        {...}
    },
    "refinement": {                    // only present for iterative-refinement manifests
        "images_with_rounds_data": 20,
        "no_correction_possible":  2,     // round 1 was already pixel-perfect
        "improved_count":  18,            // best round beat round 1
        "unchanged_count": 0,             // no round ever beat round 1
        "overshoot_count": 4,             // process continued past its own best round
        "average_delta_round1_to_best": 0.33,
        "average_overshoot_magnitude":  0.04,
        "delta_round1_to_best_rankings": {"lowest": {...}, "median": {...}, "highest": {...}},
        "stop_reason_breakdown": {"iou_threshold_met": 6, "max_points_reached": 12, "perfect_after_round1": 2},
        "overshoots": [
            {"stem": "...", "best_round": 2, "last_round": 4, "overshoot_magnitude": 0.19, "diff_finetuned": "<path>"},
            ...   // worst (largest wasted gain) first
        ]
    },
    "per_image": [
        {
            "stem": "...",
            "tp_pct": 0.0, "tn_pct": 0.0, "fp_pct": 0.0, "fn_pct": 0.0,
            "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0,
            "diff_finetuned": "<relative path, for cross-referencing the diff image — best round>",

            // only present for iterative-refinement manifests:
            "num_rounds":  4,
            "stop_reason": "iou_threshold_met",
            "best_round":  {"round": 4, "iou": 0.92, "is_last_round": true},
            "round_trajectory": [{"round": 1, "iou": 0.5, "tp_pct": 0.0, "precision": 0.0, "..."}, ...],
            "delta_round1_to_best": 0.42,   // always >= 0
            "overshoot": false,
            "overshoot_magnitude": 0.0,     // best_iou - last_round_iou, >= 0
            "correction_status": "improved"  // | "unchanged" | "not_applicable"
        },
        ...
    ]
}
```

`rankings` and `average` are redundant (fully derivable from `per_image`),
kept at the top purely for quick access — each ranking entry carries the
`diff_finetuned` path so a metric extreme can be traced straight back to
its diff visualization.

Usage

```bash
python -m quaggai_scripts.evaluation.analyze_finetuned_metrics \
    --manifest <path to comparison_manifest.json> \
    --output   <path to write JSON report>
```

Options

- `--manifest PATH` Path to comparison_manifest.json (required)
- `--output PATH` Path to write the JSON report (required)
- `--overwrite` Overwrite the output file if it already exists. Default: skip.

Example

```bash
python -m quaggai_scripts.evaluation.analyze_finetuned_metrics \
    --manifest ./comparison_visuals/comparison_manifest.json \
    --output   ./comparison_analysis/finetuned_metrics_report.json
```

### align_val_mask_dimensions.py

Fixes image/mask dimension mismatches in a dataset split (typically the
validation split) by resizing mismatched masks to match their paired
image's dimensions, using nearest-neighbor interpolation to preserve
binary mask values.

Background: `build_dataset.py` matches image/mask pairs by filename stem
only and never validates that the two files share the same dimensions.
For a meaningful fraction of pairs they don't — this causes
`run_inference.py` / `visualize_comparison.py` / `evaluate.py` to crash
with a numpy broadcast error. This script fixes the masks in place so
those scripts can run against the full split without modification —
`dataset.json` and split files are untouched, only the mask *pixel
content* changes.

Safety

- Dry-run by default — nothing is written until `--apply` is passed.
- Every resized mask has its original backed up first (unmodified), so
  the operation is fully reversible.
- Only auto-resizes when image/mask aspect ratios match within
  `--aspect-tolerance` (default 2%). A pair whose aspect ratios differ by
  more is almost certainly a genuine mismatched pairing (wrong file, not
  a scale/rounding artifact) and gets flagged for manual review instead
  of being silently resized.
- Already-matching pairs are left untouched; re-running after a partial
  `--apply` is safe.

Output structure

```
<backup-dir>/<stem>_mask.png     original mask, backed up before overwrite
<mask_path>                      overwritten in place with the resized mask
<report>                         mask_alignment_report.json — per-stem before/after
                                  sizes, flagged pairs, missing files, and a summary
```

Usage

```bash
# Dry run — see what would change, nothing is written
python -m quaggai_scripts.preprocessing.align_val_mask_dimensions --dataset <dir> --split <dir>/val.txt

# Apply the fix
python -m quaggai_scripts.preprocessing.align_val_mask_dimensions --dataset <dir> --split <dir>/val.txt --apply
```

Options

- `--dataset PATH` Dataset folder with dataset.json, images/, masks/ (required)
- `--split PATH` Optional .txt file of stems to process. If omitted, ALL entries
  in dataset.json are processed — pass the val/test split explicitly to scope this.
- `--aspect-tolerance FLOAT` Max relative aspect-ratio difference treated as a
  safe auto-resize. Default: 0.02 (2%).
- `--backup-dir PATH` Where to back up originals. Default: `<dataset_dir>/masks_backup_pre_resize`
- `--report PATH` Where to write the JSON report. Default: `<dataset_dir>/mask_alignment_report.json`
- `--apply` Actually write changes. Default: dry run (report only).

Example

```bash
python -m quaggai_scripts.preprocessing.align_val_mask_dimensions \
    --dataset ./dataset \
    --split   ./dataset/val.txt \
    --apply
```

### visualize_iterative_refinement.py

Runs SAM2 inference with two checkpoints (baseline and fine-tuned),
modeling a multi-click interactive refinement workflow for the
**fine-tuned model**: a first random point, then up to `--max-points - 1`
additional corrective points, each chosen from wherever the previous
prediction erred most — imitating how a user would actually correct a
segmentation mask, stopping once IoU reaches `--iou-threshold`.

Per image:

1. Point 1: a single random positive point from the GT foreground (seeded).
2. Baseline model runs once with point 1 only → `model_baseline.png`. No
   diff is produced for the baseline in this workflow.
3. Fine-tuned model runs with point 1 only (round 1).
4. Point 2 is **always** added next (round 2), chosen from round 1's
   errors — unless round 1 is already pixel-perfect (FP=FN=0), in which
   case there's no valid region for a corrective point and the image
   stops at round 1 (`stop_reason: "perfect_after_round1"`):
   - FP area > FN area → a **negative** point sampled from the FP region
     (over-predicted — shrink it)
   - FN area > FP area → a **positive** point sampled from the FN region
     (missed real coverage — grow it)
   - FP == FN > 0 → tie, defaults to the grow/positive branch
5. From round 3 onward: after each round, if IoU is still below
   `--iou-threshold` and `--max-points` hasn't been reached, another
   corrective point is added (same rule, applied to the latest
   prediction) and another round is run. Stops as soon as IoU reaches
   the threshold (`stop_reason: "iou_threshold_met"`) or the point
   budget is exhausted (`stop_reason: "max_points_reached"`).
6. Each round *K* writes `model_finetuned_K.png` + `diff_finetuned_K.png`,
   showing all *K* cumulative points (point 1 white, every corrective
   point magenta).
7. Once every round for an image is computed, whichever round had the
   highest IoU (earliest round wins ties) is additionally saved as
   `model_finetuned_best.png` + `diff_finetuned_best.png`.
8. `ground_truth.png` is rendered last, showing every point that was
   ultimately placed.

Output structure

```
<output>/
    <stem>_ground_truth.png          GT overlay, point 1 white + every corrective point magenta
    <stem>_model_baseline.png        baseline prediction overlay, point 1 only
    <stem>_model_finetuned_1.png     round-1 prediction overlay
    <stem>_diff_finetuned_1.png      round-1 TP/FP/FN error map
    <stem>_model_finetuned_2.png     round-2 outputs (always produced, unless round 1
    <stem>_diff_finetuned_2.png      was already pixel-perfect)
    <stem>_model_finetuned_3.png     round-3+ outputs, only as many as actually ran
    <stem>_diff_finetuned_3.png
    ...
    <stem>_model_finetuned_best.png  duplicate of whichever round had the highest IoU
    <stem>_diff_finetuned_best.png   (earliest round wins ties)
    ...
    comparison_manifest.json         metadata (incl. max_points, iou_threshold) + a
                                      `rounds` list per image (cumulative point count, the
                                      point added that round, IoU, score, pixel counts,
                                      file paths), a `best_round` block, and `stop_reason`
```

The manifest's field names are chosen to stay compatible with
`analyze_comparison_manifest.py` / `analyze_finetuned_metrics.py`:
`iou_finetuned` / `score_finetuned` / `pixel_counts_finetuned` /
`paths.model_finetuned` / `paths.diff_finetuned` alias the **best**
round (not necessarily the last one run), alongside the full `rounds`
list and `best_round` block with per-round detail. `paths.diff_baseline`
is `null` — no baseline diff is produced in this workflow.

Because the process can overshoot (a later round ending up worse than
an earlier peak — see `analyze_finetuned_metrics.py` /
`analyze_comparison_manifest.py` below), `best_round.is_last_round`
tells you whether the run's own final round was actually its best one.

Usage

```bash
python -m quaggai_scripts.evaluation.visualize_iterative_refinement \
    --dataset              <dir> \
    --baseline-checkpoint  <path> \
    --finetuned-checkpoint <path> \
    --output               <dir> \
    [options]
```

Options

- `--dataset PATH` Dataset folder with dataset.json, images/, masks/ (required)
- `--baseline-checkpoint PATH` Baseline SAM2 checkpoint (.pt) (required)
- `--finetuned-checkpoint PATH` Fine-tuned SAM2 checkpoint (.pt) (required)
- `--baseline-model-cfg STR` / `--finetuned-model-cfg STR`
    - SAM2 model config for each checkpoint. Default: auto-detected from filename.
- `--split PATH` Optional .txt file of stems to process. Default: all entries.
- `--seed INT` Random seed for point sampling. Default: 42.
- `--max-points INT` Maximum total points per image, point 1 included (round *K* has *K*
  points). Must be >= 2, since point 2 is always placed. Default: 5.
- `--iou-threshold FLOAT` Stop adding points once IoU reaches this value. Default: 0.9.
- `--output PATH` Folder to write overlay images + manifest (required)
- `--alpha FLOAT` Overlay alpha for GT/baseline/finetuned masks. Default: 0.45
- `--diff-alpha FLOAT` Overlay alpha for TP/FP/FN regions in diff images. Default: 0.55
- `--diff-dim FLOAT` Brightness factor for dimmed background in diff images. Default: 0.5
- `--marker-alpha FLOAT` Alpha for the prompt point markers. Default: 0.75
- `--overwrite` Overwrite existing output files. Default: skip.
- `--dry-run` Print what would be run without writing anything.
- `--device STR` Device to run inference on. Default: cuda.

Example

```bash
python -m quaggai_scripts.evaluation.visualize_iterative_refinement \
    --dataset ./dataset \
    --baseline-checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
    --finetuned-checkpoint ./checkpoints/quaggai_v2.pt \
    --split ./dataset/val.txt \
    --max-points 5 --iou-threshold 0.9 \
    --output ./refinement_visuals
```

### add_raw_images.py

Copies the untouched raw source image for each processed entry in a
`comparison_manifest.json` into that same output directory, and records
the path in the manifest — needed because
`visualize_iterative_refinement.py` only ever writes overlaid images
(ground truth, baseline, per-round predictions/diffs), never a plain
copy of the original photo. Uses the dataset the evaluation was
originally run against, whose `dataset.json` already records each
stem's exact `image_path` (and therefore its real extension) — no
guessing, and no need to re-run inference.

Output structure

```
<output>/
    <stem>_raw.<ext>              copy of <dataset>/<image_path>, extension preserved
    comparison_manifest.json      updated in place: each processed entry's
                                   paths.raw_image = "<stem>_raw.<ext>"
    comparison_manifest.json.bak  backup of the manifest as it was before this script ran
                                   (written once, never overwritten by later runs)
```

Usage

```bash
python -m quaggai_scripts.evaluation.add_raw_images --dataset <dir> --output <dir> [options]
```

Options

- `--dataset PATH` Dataset folder containing dataset.json (required)
- `--output PATH` Evaluation output folder containing comparison_manifest.json (required)
- `--overwrite` Overwrite already-copied raw images. Default: skip existing files.
- `--dry-run` Print what would be copied/updated without writing anything.

Example

```bash
python -m quaggai_scripts.evaluation.add_raw_images --dataset ./dataset --output ./refinement_visuals
```

### /eval_viewer/ (repo root)

A small local web app (`index.html` + `style.css` + `app.js`, no build
step, no framework) for browsing a `visualize_iterative_refinement.py`
run visually: overall macro-average stats, worst/median/best per
metric, a sortable + searchable table of every sample, and a detail
view per sample showing the raw image, ground truth, baseline, and
every round's prediction + diff with the best round highlighted, each
with its own TP/TN/FP/FN/IoU/Precision/Recall/Accuracy/F1 stats box.
Every image in the detail view — reference cards and each round's
prediction/diff alike — renders at the same fixed width, so cards are
directly size-comparable.

Lives at `/eval_viewer/` in the repo root (not under `scripts/`),
since it's a standalone tool rather than a Python script.

Everything is computed client-side directly from the raw
`comparison_manifest.json` — no dependency on `analyze_finetuned_metrics.py`
or `analyze_comparison_manifest.py` having been run first. One shared
copy of the viewer lives in the repo; point it at any evaluation run via
a `?data=` query parameter rather than copying it into every output folder.

Because it reads files by relative path via `fetch()`, it needs to be
served over `http://`, not opened directly as a `file://` URL (browsers
block that). From the repo root:

```bash
python3 -m http.server 8000
```

Then open, e.g.:

```
http://localhost:8000/eval_viewer/index.html?data=../output/output_vis_v4/comparison_manifest.json
```

(`data` is resolved relative to `index.html`'s own location — adjust the
path to match where your output folder actually sits relative to
`eval_viewer/`; if you copy the output folder in next to the viewer
itself, as in `eval_viewer/output_vis_v4/`, `data` can just be
`output_vis_v4/comparison_manifest.json`.) You can also just type/paste
a path into the "Manifest" field at the top of the page instead of
editing the URL.

The raw-image card will show a "not available" placeholder until you've
run `add_raw_images.py` against that output folder. Manifests without
per-round data (i.e. not from `visualize_iterative_refinement.py`) will
load but show a notice — this viewer is built specifically around the
`rounds`/`best_round` schema.

### Typical Evaluation and Comparison Flow

Step 1 — Baseline inference (stock checkpoint):

```bash
# writes → ./predictions_sam2.1_hiera_base_plus/
python -m quaggai_scripts.evaluation.run_inference \
    --dataset ./dataset \
    --checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
    --split ./dataset/val.txt
```

Step 2 — Fine-tuned inference:

```bash
# writes → ./predictions_checkpoint_7000/
python -m quaggai_scripts.evaluation.run_inference \
    --dataset ./dataset \
    --checkpoint ./sam2_logs/checkpoints/checkpoint_7000.pt \
    --split ./dataset/val.txt
```

Step 3 — Comparison evaluation:

```bash
python -m quaggai_scripts.evaluation.evaluate \
    --dataset ./dataset \
    --preds   ./predictions_sam2.1_hiera_base_plus \
    --preds-b ./predictions_checkpoint_7000 \
    --output  ./evaluation \
    --label-a "Baseline" \
    --label-b "Fine-tuned step 7000"
```
