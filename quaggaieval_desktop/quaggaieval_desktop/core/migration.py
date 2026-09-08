"""
core/migration.py — Data migration from old file layout to new layout.

Old layout (any version before the Phase 1 refactor):
  {folder}/
  ├── calibration.json
  └── evaluation_results/
      ├── {stem}_eval_area_polygon.json   (per-image, pixel-coord vertices)
      ├── {stem}_eval_area_mask.png
      ├── {stem}_segmentation_mask.png
      ├── {stem}_segmentation_overlay.png
      ├── {stem}_segmentation_meta.json
      ├── {stem}_coverage.json
      ├── {stem}_coverage.png
      └── {stem}_coverage_overlay.png

New layout:
  {folder}/
  └── evaluation_results/
      ├── .internal/
      │   ├── calibration.json
      │   ├── eval_areas.json
      │   └── {stem}_segmentation_mask.png  (max-compressed)
      ├── cropped/{stem}_cropped.png
      ├── area/{stem}_area.png
      └── coverage_data.csv
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .calibration import ScaleCalibration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "filename",
    "area_px", "covered_px", "uncovered_px", "outside_px",
    "coverage_pct", "outside_pct",
    "area_cm2", "covered_cm2", "uncovered_cm2", "outside_cm2",
]

# All old-format file suffixes that should be archived
_ARCHIVE_SUFFIXES = [
    "_eval_area_polygon.json",
    "_eval_area_mask.png",
    "_segmentation_mask.png",
    "_segmentation_overlay.png",
    "_segmentation_meta.json",
    "_coverage.json",
    "_coverage.png",
    "_coverage_overlay.png",
]

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_old_format(folder_path: Path) -> bool:
    """
    Return True if the folder contains old-format output files that require
    migration.  Returns False if the folder is already in the new format, or
    if no output directory exists yet.
    """
    out = folder_path / "evaluation_results"
    if not out.exists():
        return False
    # Already migrated if the consolidated polygon JSON is present
    if (out / ".internal" / "eval_areas.json").exists():
        return False
    # Old format is present if any per-image polygon JSON file exists
    return any(out.glob("*_eval_area_polygon.json"))


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_folder(
    folder_path: Path,
    calibration: ScaleCalibration | None,
    on_progress: Callable[[int, str], None],
    on_done: Callable[[list[str]], None],
    on_error: Callable[[str], None],
) -> None:
    """
    Migrate a folder from the old file layout to the new layout.

    Must be called from a background thread; all progress/completion
    callbacks are called directly (wire them through wx.CallAfter at the
    call site if UI updates are needed).

    Progress is reported as an integer percentage (0–100) plus a short
    message string.  Individual per-image errors are collected and passed
    to on_done(); critical errors that prevent migration abort early via
    on_error().
    """
    errors: list[str] = []

    try:
        out = folder_path / "evaluation_results"
        internal_dir = out / ".internal"

        # ── Step 1: Create .internal/ ─────────────────────────────────
        on_progress(2, "Creating internal directory…")
        try:
            internal_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            on_error(f"Cannot create .internal/ directory: {exc}")
            return

        # ── Step 2: Migrate calibration ───────────────────────────────
        on_progress(5, "Migrating calibration…")
        old_cal_path = folder_path / "calibration.json"
        new_cal_path = internal_dir / "calibration.json"
        if old_cal_path.exists() and not new_cal_path.exists():
            try:
                shutil.copy2(str(old_cal_path), str(new_cal_path))
            except Exception as exc:
                errors.append(f"calibration.json: {exc}")

        # ── Step 3: Migrate eval area polygons → eval_areas.json ──────
        on_progress(10, "Migrating evaluation area polygons…")
        polygon_files = sorted(out.glob("*_eval_area_polygon.json"))

        # Load existing eval_areas.json so interrupted migrations are additive
        eval_areas_path = internal_dir / "eval_areas.json"
        all_areas: dict = {}
        if eval_areas_path.exists():
            try:
                with open(eval_areas_path) as f:
                    all_areas = json.load(f)
            except Exception:
                all_areas = {}

        n_poly = max(len(polygon_files), 1)
        for i, poly_file in enumerate(polygon_files):
            pct = 10 + int(18 * i / n_poly)
            stem = poly_file.name.replace("_eval_area_polygon.json", "")
            on_progress(pct, f"Converting polygon: {stem}")
            try:
                with open(poly_file) as f:
                    data = json.load(f)

                img_size = data.get("image_size", {})
                if isinstance(img_size, dict):
                    img_w = int(img_size.get("width",  1))
                    img_h = int(img_size.get("height", 1))
                else:
                    img_w, img_h = int(img_size[0]), int(img_size[1])

                raw_verts = data.get("vertices", [])
                norm_verts = [
                    [px / img_w, py / img_h]
                    for px, py in raw_verts
                ] if img_w > 0 and img_h > 0 else []

                image_filename = data.get("image", f"{stem}.jpg")
                all_areas[image_filename] = {
                    "image_size": [img_w, img_h],
                    "vertices":   norm_verts,
                    "skipped":    data.get("skipped", False),
                    "timestamp":  data.get("timestamp", ""),
                }
            except Exception as exc:
                errors.append(f"{poly_file.name}: {exc}")

        try:
            with open(eval_areas_path, "w") as f:
                json.dump(all_areas, f, indent=2)
        except Exception as exc:
            on_error(f"Cannot write eval_areas.json: {exc}")
            return

        # ── Step 4: Migrate segmentation masks ────────────────────────
        on_progress(30, "Migrating segmentation masks…")
        old_masks = sorted(out.glob("*_segmentation_mask.png"))
        n_masks = max(len(old_masks), 1)
        for i, mask_file in enumerate(old_masks):
            pct = 30 + int(18 * i / n_masks)
            on_progress(pct, f"Recompressing mask: {mask_file.stem}")
            new_mask_path = internal_dir / mask_file.name
            if new_mask_path.exists():
                continue  # already migrated (interrupted migration)
            try:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    errors.append(f"{mask_file.name}: could not read image")
                    continue
                cv2.imwrite(
                    str(new_mask_path), mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
            except Exception as exc:
                errors.append(f"{mask_file.name}: {exc}")

        # ── Step 5: Migrate coverage data → coverage_data.csv ─────────
        on_progress(50, "Migrating coverage data to CSV…")
        coverage_files = sorted(out.glob("*_coverage.json"))
        csv_rows: list[dict] = []

        for coverage_file in coverage_files:
            stem = coverage_file.name.replace("_coverage.json", "")
            try:
                with open(coverage_file) as f:
                    cdata = json.load(f)

                area_cm2      = cdata.get("area_cm2")
                covered_cm2   = cdata.get("covered_cm2")
                uncovered_cm2 = cdata.get("uncovered_cm2")
                outside_cm2   = cdata.get("outside_cm2")

                # Recompute cm² if old data lacked it but calibration is now available
                if area_cm2 is None and calibration is not None:
                    area_cm2      = calibration.px_to_cm2(cdata["area_pixels"])
                    covered_cm2   = calibration.px_to_cm2(cdata["covered_pixels"])
                    uncovered_cm2 = calibration.px_to_cm2(cdata["uncovered_pixels"])
                    outside_cm2   = calibration.px_to_cm2(cdata["outside_pixels"])

                csv_rows.append({
                    "filename":      stem,
                    "area_px":       cdata["area_pixels"],
                    "covered_px":    cdata["covered_pixels"],
                    "uncovered_px":  cdata["uncovered_pixels"],
                    "outside_px":    cdata["outside_pixels"],
                    "coverage_pct":  cdata["coverage_pct"],
                    "outside_pct":   cdata["outside_pct"],
                    "area_cm2":      area_cm2      if area_cm2      is not None else "",
                    "covered_cm2":   covered_cm2   if covered_cm2   is not None else "",
                    "uncovered_cm2": uncovered_cm2 if uncovered_cm2 is not None else "",
                    "outside_cm2":   outside_cm2   if outside_cm2   is not None else "",
                })
            except Exception as exc:
                errors.append(f"{coverage_file.name}: {exc}")

        if csv_rows:
            csv_path = out / "coverage_data.csv"
            try:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(csv_rows)
            except Exception as exc:
                errors.append(f"coverage_data.csv: {exc}")

        # ── Step 6: Regenerate user outputs (cropped + area PNGs) ─────
        on_progress(55, "Regenerating user output images…")
        covered_stems = {row["filename"] for row in csv_rows}
        image_filenames = list(all_areas.keys())
        n_imgs = max(len(image_filenames), 1)

        for i, image_filename in enumerate(image_filenames):
            stem = Path(image_filename).stem
            if stem not in covered_stems:
                continue  # no coverage data → Tab 4 was never run

            mask_path = internal_dir / f"{stem}_segmentation_mask.png"
            if not mask_path.exists():
                continue

            pct = 55 + int(28 * i / n_imgs)
            on_progress(pct, f"Generating outputs: {stem}")

            try:
                source_bgr = cv2.imread(str(folder_path / image_filename))
                if source_bgr is None:
                    errors.append(f"{image_filename}: original image not found")
                    continue

                img_h, img_w = source_bgr.shape[:2]

                mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_raw is None:
                    errors.append(f"{stem}: cannot load segmentation mask")
                    continue
                obj_mask = mask_raw > 0

                entry    = all_areas[image_filename]
                ew, eh   = entry["image_size"]
                verts    = [(v[0], v[1]) for v in entry.get("vertices", [])]
                skipped  = entry.get("skipped", False)

                area_mask_raw = _rasterise_polygon(verts, ew, eh, skipped)
                area_mask = area_mask_raw > 0

                # Resize masks if they differ from the source image dimensions
                if area_mask.shape != (img_h, img_w):
                    area_mask = cv2.resize(
                        area_mask.astype(np.uint8), (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                if obj_mask.shape != (img_h, img_w):
                    obj_mask = cv2.resize(
                        mask_raw, (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0

                # Bounding box of eval polygon
                vertices_norm = entry.get("vertices", [])
                if skipped or len(vertices_norm) < 3:
                    x0, y0, x1, y1 = 0, 0, img_w, img_h
                else:
                    xs = [int(v[0] * img_w) for v in vertices_norm]
                    ys = [int(v[1] * img_h) for v in vertices_norm]
                    x0 = max(0, min(xs))
                    y0 = max(0, min(ys))
                    x1 = min(img_w, max(xs) + 1)
                    y1 = min(img_h, max(ys) + 1)

                crop_bgr        = source_bgr[y0:y1, x0:x1]
                crop_area_alpha = area_mask[y0:y1, x0:x1].astype(np.uint8) * 255
                crop_seg_alpha  = (
                    (obj_mask[y0:y1, x0:x1] & area_mask[y0:y1, x0:x1]).astype(np.uint8)
                ) * 255

                # cropped/{stem}_cropped.png — original BGRA, alpha = eval area
                bgra_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
                bgra_crop[:, :, 3] = crop_area_alpha
                cropped_dir = out / "cropped"
                cropped_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(cropped_dir / f"{stem}_cropped.png"), bgra_crop)

                # area/{stem}_area.png — original BGRA, alpha = seg ∩ eval area
                bgra_area = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
                bgra_area[:, :, 3] = crop_seg_alpha
                area_dir = out / "area"
                area_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(area_dir / f"{stem}_area.png"), bgra_area)

            except Exception as exc:
                errors.append(f"{stem} (output generation): {exc}")

        # ── Step 7: Archive old files ──────────────────────────────────
        on_progress(88, "Archiving old files…")
        backup_dir = out / "_backup"
        try:
            backup_dir.mkdir(exist_ok=True)

            for suffix in _ARCHIVE_SUFFIXES:
                for old_file in out.glob(f"*{suffix}"):
                    shutil.move(str(old_file), str(backup_dir / old_file.name))

            # Archive folder-root calibration.json
            old_cal = folder_path / "calibration.json"
            if old_cal.exists():
                target = backup_dir / "calibration.json"
                if not target.exists():
                    shutil.move(str(old_cal), str(target))

        except Exception as exc:
            errors.append(f"Archive step: {exc}")

        on_progress(100, "Migration complete.")
        on_done(errors)

    except Exception as exc:
        on_error(f"Unexpected error during migration: {exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rasterise_polygon(
    vertices_norm: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    skipped: bool = False,
) -> np.ndarray:
    """Convert normalised polygon vertices to a binary uint8 mask (0/255)."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if skipped or len(vertices_norm) < 3:
        mask[:] = 255
    else:
        pts = np.array(
            [(int(nx * img_w), int(ny * img_h)) for nx, ny in vertices_norm],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
    return mask
