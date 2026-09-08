# batch_preprocess.py
"""
Batch preprocessing orchestrator for the QuaggAI dataset pipeline.

Chains all preprocessing steps in order, driven by a single YAML config file.

Pipeline:
    1. png_to_mask     — alpha-channel PNG → binary mask PNG
    2. png_to_jpg      — baseline PNG → JPEG
    3. build_dataset   — match image/mask pairs, build structured dataset/
    4. split_dataset   — produce train.txt + val.txt
    5. convert_to_sa1b — convert to SA-1B JSON annotation format
    6. rename_to_sa1b  — rename files to numeric convention (sa_000001.*)

Usage:
    python -m quaggai_scripts.preprocessing.batch_preprocess --config configs/batch_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from . import png_to_mask as pngmask
from . import png_to_jpg as pngjpg
from . import build_dataset as bds
from . import split_dataset as sds
from . import convert_to_sa1b as conv_sa1b
from . import rename_to_sa1b as rename_sa1b

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when the YAML config is missing required fields or has invalid values."""


class StepError(RuntimeError):
    """Raised when a pipeline step fails."""

    def __init__(self, step_name: str, cause: Exception):
        super().__init__(f"Pipeline failed at step '{step_name}': {cause}")
        self.step_name = step_name
        self.cause = cause


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load and parse the YAML config file."""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: '{path}'")
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse config YAML: {exc}")
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a YAML mapping at the top level.")
    return cfg


def validate_config(cfg: dict) -> None:
    """Validate config structure and values; raise ConfigError listing all problems."""
    errors: list[str] = []

    # ── Top-level sections ────────────────────────────────────────────────
    for section in ("global", "paths", "steps"):
        if section not in cfg:
            errors.append(f"Missing required top-level section: '{section}'")

    if errors:
        raise ConfigError("Config validation failed:\n  " + "\n  ".join(errors))

    g = cfg["global"]
    p = cfg["paths"]
    s = cfg["steps"]

    # ── Global flags ──────────────────────────────────────────────────────
    for key in ("overwrite", "dry_run"):
        if key not in g:
            errors.append(f"global.{key} is required.")
        elif not isinstance(g[key], bool):
            errors.append(f"global.{key} must be a boolean, got {type(g[key]).__name__}.")

    # ── Paths ─────────────────────────────────────────────────────────────
    required_paths = (
        "input_alpha_masks", "input_baseline_imgs",
        "output_masks", "output_jpgs", "output_dataset", "output_sa1b",
    )
    for key in required_paths:
        if key not in p:
            errors.append(f"paths.{key} is required.")

    # Input paths must exist
    for key in ("input_alpha_masks", "input_baseline_imgs"):
        if key in p and p[key] is not None:
            pth = Path(p[key])
            if not pth.is_dir():
                errors.append(f"paths.{key} '{pth}' does not exist or is not a directory.")

    # ── Steps ─────────────────────────────────────────────────────────────
    step_names = (
        "png_to_mask", "png_to_jpg", "build_dataset",
        "split_dataset", "convert_to_sa1b", "rename_to_sa1b",
    )
    for name in step_names:
        if name not in s:
            errors.append(f"steps.{name} section is required.")
            continue
        sc = s[name]
        if "enabled" not in sc:
            errors.append(f"steps.{name}.enabled is required.")
        elif not isinstance(sc["enabled"], bool):
            errors.append(f"steps.{name}.enabled must be a boolean.")

    # Per-step numeric range checks (only if the section exists)
    def _check_range(step, key, lo, hi):
        sc = s.get(step, {})
        if key in sc and sc[key] is not None:
            v = sc[key]
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                errors.append(
                    f"steps.{step}.{key} must be a number in [{lo}, {hi}], got {v!r}."
                )

    _check_range("png_to_mask", "alpha_threshold", 0, 255)
    _check_range("png_to_jpg", "quality", 1, 95)
    _check_range("split_dataset", "split_ratio", 0.0, 1.0)
    _check_range("convert_to_sa1b", "fg_threshold", 0, 254)

    if "split_dataset" in s and "seed" in s["split_dataset"]:
        seed = s["split_dataset"]["seed"]
        if not isinstance(seed, int):
            errors.append(f"steps.split_dataset.seed must be an integer, got {type(seed).__name__}.")

    if "rename_to_sa1b" in s and "prefix" in s["rename_to_sa1b"]:
        prefix = s["rename_to_sa1b"]["prefix"]
        if not isinstance(prefix, str) or not prefix:
            errors.append("steps.rename_to_sa1b.prefix must be a non-empty string.")

    if errors:
        raise ConfigError("Config validation failed:\n  " + "\n  ".join(errors))

    if g.get("dry_run"):
        logger.warning("dry_run is enabled globally — no files will be written.")


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------

def resolve_flag(step_val, global_val: bool) -> bool:
    """Return *step_val* if explicitly set, otherwise fall back to *global_val*."""
    if step_val is None:
        return global_val
    return bool(step_val)


# ---------------------------------------------------------------------------
# Step runner helpers
# ---------------------------------------------------------------------------

def _step_banner(name: str) -> None:
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)


def _step_summary(counts: dict, elapsed: float) -> None:
    parts = "  ".join(f"{k}: {v}" for k, v in counts.items())
    logger.info("Result  — %s  (%.1fs)", parts, elapsed)


def _check_failure(counts: dict, step_name: str) -> None:
    if counts.get("failed", 0) > 0:
        raise StepError(step_name, RuntimeError(f"{counts['failed']} item(s) failed."))


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_png_to_mask(step_cfg: dict, global_cfg: dict, paths: dict) -> None:
    overwrite = resolve_flag(step_cfg.get("overwrite"), global_cfg["overwrite"])
    counts = pngmask.process_batch(
        input_dir=Path(paths["input_alpha_masks"]),
        output_dir=Path(paths["output_masks"]),
        alpha_threshold=step_cfg.get("alpha_threshold", 128),
        suffix=step_cfg.get("suffix", "_mask"),
        recursive=step_cfg.get("recursive", False),
        overwrite=overwrite,
    )
    _check_failure(counts, "png_to_mask")
    return counts


def run_png_to_jpg(step_cfg: dict, global_cfg: dict, paths: dict) -> None:
    overwrite = resolve_flag(step_cfg.get("overwrite"), global_cfg["overwrite"])
    counts = pngjpg.process_batch(
        input_dir=Path(paths["input_baseline_imgs"]),
        output_dir=Path(paths["output_jpgs"]),
        quality=step_cfg.get("quality", 95),
        overwrite=overwrite,
    )
    _check_failure(counts, "png_to_jpg")
    return counts


def run_build_dataset(step_cfg: dict, global_cfg: dict, paths: dict) -> None:
    overwrite = resolve_flag(step_cfg.get("overwrite"), global_cfg["overwrite"])
    img_dir = Path(paths["output_jpgs"])
    mask_dir = Path(paths["output_masks"])
    out_dir = Path(paths["output_dataset"])
    unmatched_report = (
        Path(paths["unmatched_report"]) if paths.get("unmatched_report") else None
    )

    logger.info("Images folder    : %s", img_dir)
    logger.info("Masks folder     : %s", mask_dir)
    logger.info("Output folder    : %s", out_dir)

    image_index, image_conflicts = bds.index_images(img_dir)
    mask_index, mask_conflicts = bds.index_masks(mask_dir)

    logger.info("Found %d image(s), %d mask(s).", len(image_index), len(mask_index))

    matched, unmatched_images, unmatched_masks = bds.match_pairs(image_index, mask_index)
    logger.info("Matched pairs    : %d", len(matched))

    bds.print_unmatched(unmatched_images, unmatched_masks)

    if not matched:
        raise StepError(
            "build_dataset",
            RuntimeError("No matched pairs found. Check folder contents and naming conventions."),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, copy_failures, counts = bds.build_dataset(matched, out_dir, overwrite=overwrite)

    manifest_path = bds.write_manifest(manifest, out_dir)
    logger.info("Manifest written : %s", manifest_path)

    if unmatched_report is not None:
        report = bds.build_unmatched_report(
            unmatched_images=unmatched_images,
            unmatched_masks=unmatched_masks,
            image_index=image_index,
            mask_index=mask_index,
            image_conflicts=image_conflicts,
            mask_conflicts=mask_conflicts,
            copy_failures=copy_failures,
        )
        bds.write_unmatched_report(report, unmatched_report)
        logger.info("Unmatched report : %s", unmatched_report)

        s = report["summary"]
        if any(s.values()):
            logger.warning(
                "images without mask: %d | masks without image: %d | "
                "stem conflicts (img/mask): %d/%d | copy failures: %d",
                s["images_without_mask"], s["masks_without_image"],
                s["stem_conflicts_images"], s["stem_conflicts_masks"],
                s["copy_failures"],
            )

    _check_failure(counts, "build_dataset")
    return counts


def run_split_dataset(step_cfg: dict, paths: dict) -> None:
    dataset_dir = Path(paths["output_dataset"])
    counts = sds.run(
        dataset_dir=dataset_dir,
        split_ratio=step_cfg.get("split_ratio", 0.8),
        seed=step_cfg.get("seed", 42),
    )
    logger.info("Train: %d  Val: %d", counts["train"], counts["val"])
    return counts


def run_convert_to_sa1b(step_cfg: dict, global_cfg: dict, paths: dict) -> None:
    overwrite = resolve_flag(step_cfg.get("overwrite"), global_cfg["overwrite"])
    dry_run = resolve_flag(step_cfg.get("dry_run"), global_cfg["dry_run"])
    dataset_dir = Path(paths["output_dataset"])
    output_dir = Path(paths["output_sa1b"])

    entries = conv_sa1b.load_manifest(dataset_dir)
    entries = conv_sa1b.filter_entries(entries, stem_filter=None)

    counts = conv_sa1b.run_conversion(
        entries=entries,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        fg_threshold=step_cfg.get("fg_threshold", 0),
        overwrite=overwrite,
        dry_run=dry_run,
    )

    if step_cfg.get("copy_splits", True) and not dry_run:
        conv_sa1b.copy_split_files(dataset_dir, output_dir)

    _check_failure(counts, "convert_to_sa1b")
    return counts


def run_rename_to_sa1b(step_cfg: dict, global_cfg: dict, paths: dict) -> None:
    dry_run = resolve_flag(step_cfg.get("dry_run"), global_cfg["dry_run"])
    # SA-1B images live inside the output_sa1b/images/ subdirectory
    img_dir = Path(paths["output_sa1b"]) / "images"
    counts = rename_sa1b.run(
        img_dir=img_dir,
        prefix=step_cfg.get("prefix", "sa_"),
        dry_run=dry_run,
    )
    return counts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

STEP_REGISTRY = [
    ("png_to_mask",     run_png_to_mask),
    ("png_to_jpg",      run_png_to_jpg),
    ("build_dataset",   run_build_dataset),
    ("split_dataset",   run_split_dataset),
    ("convert_to_sa1b", run_convert_to_sa1b),
    ("rename_to_sa1b",  run_rename_to_sa1b),
]


def _setup_logging(log_level: str) -> None:
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(cfg: dict) -> None:
    _setup_logging(cfg["global"].get("log_level", "INFO"))
    validate_config(cfg)

    for name, runner in STEP_REGISTRY:
        step_cfg = cfg["steps"][name]
        if not step_cfg["enabled"]:
            logger.info("[SKIP] %s — disabled in config", name)
            continue

        _step_banner(name)
        t0 = time.monotonic()
        try:
            if name == "split_dataset":
                counts = runner(step_cfg, cfg["paths"])
            else:
                counts = runner(step_cfg, cfg["global"], cfg["paths"])
        except StepError:
            raise
        except Exception as exc:
            raise StepError(name, exc) from exc

        if counts:
            _step_summary(counts, time.monotonic() - t0)

    logger.info("")
    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch preprocessing orchestrator for the QuaggAI dataset pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, type=Path, metavar="PATH",
        help="Path to the YAML batch config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        cfg = load_config(args.config)
        run_pipeline(cfg)
    except (ConfigError, StepError) as exc:
        # Logger may not be set up yet for ConfigError, fall back to stderr
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
