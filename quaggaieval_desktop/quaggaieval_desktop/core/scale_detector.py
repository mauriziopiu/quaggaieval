"""
scale_detector — automatic detection of the B/W reference scale bar.

The scale bar is a vertical strip of alternating black and white rectangular
segments located on the right edge of every image (part of the metal frame).

Detection approach:
  1. Crop the rightmost ~30 % of image width as a vertical strip
  2. Grayscale + horizontal average → 1-D signal (one value per row)
  3. Gaussian smooth to reduce noise
  4. Threshold at (min + max) / 2 → binary profile
  5. Run-length encode → list of (value, length) runs
  6. Filter runs shorter than a minimum size (remove noise)
  7. Median run length = segment_px; count valid runs
  8. Locate the contiguous region of most-consistent runs
  9. Confidence = 1 - CoV(run lengths), clamped to [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Fraction of image width used as the right-edge strip (also imported by UI for overlay)
STRIP_FRACTION = 0.30

# Gaussian kernel size for smoothing (pixels)
_SMOOTH_KERNEL = 5

# Minimum run length as a fraction of the expected segment height.
# Runs shorter than this are discarded as noise.
_MIN_RUN_FRACTION = 0.3

# Confidence thresholds
CONF_GOOD = 0.80
CONF_WARN = 0.50


@dataclass
class DetectionResult:
    """Output from detect_scale_bar()."""

    segment_px: float       # median segment height in pixels
    segment_count: int      # number of detected segments
    scale_top_y: int        # first row of the detected scale region
    scale_bottom_y: int     # last row of the detected scale region
    confidence: float       # 0..1 — higher is more reliable
    strip_bgr: np.ndarray   # annotated strip image (H × strip_w × 3) for preview


def detect_scale_bar(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> DetectionResult:
    """
    Detect the alternating B/W scale bar within `image_bgr`.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full image in BGR format.
    roi : (x1, y1, x2, y2) in pixel coords, optional
        If provided, detection runs only within this crop.
        All coordinates in the returned DetectionResult are in full-image space.
        If None, the rightmost STRIP_FRACTION of the image is used (legacy fallback).

    Returns a DetectionResult.  Never raises; on total failure returns a
    result with confidence = 0.
    """
    img_h, img_w = image_bgr.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, x2 = max(0, min(x1, x2)), min(img_w - 1, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(img_h - 1, max(y1, y2))
        strip = image_bgr[y1:y2, x1:x2, :]
        y_offset = y1
    else:
        strip_w = max(1, int(img_w * STRIP_FRACTION))
        strip = image_bgr[:, img_w - strip_w:, :]
        y_offset = 0

    # ── 2. Grayscale + horizontal average → 1-D profile ─────────────────
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray.mean(axis=1)  # shape (img_h,)

    # ── 3. Gaussian smooth ───────────────────────────────────────────────
    k = _SMOOTH_KERNEL if _SMOOTH_KERNEL % 2 == 1 else _SMOOTH_KERNEL + 1
    smoothed = cv2.GaussianBlur(
        profile.reshape(-1, 1), (1, k), 0
    ).ravel()

    # ── 4. Threshold → binary profile ───────────────────────────────────
    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo < 10:
        # No meaningful contrast — detection fails
        return _failed_result(strip, y_offset)

    threshold = (lo + hi) / 2.0
    binary = (smoothed > threshold).astype(np.uint8)

    # ── 5. Run-length encode ─────────────────────────────────────────────
    runs = _rle(binary)  # list of [value, length, start_row]

    if len(runs) < 3:
        return _failed_result(strip, y_offset)

    # ── 6. Filter short runs (noise) ────────────────────────────────────
    lengths = np.array([r[1] for r in runs], dtype=np.float32)
    median_len = float(np.median(lengths))
    min_len = max(1.0, median_len * _MIN_RUN_FRACTION)
    valid_runs = [r for r in runs if r[1] >= min_len]

    if len(valid_runs) < 3:
        return _failed_result(strip, y_offset)

    # ── 7. Median segment size & count ──────────────────────────────────
    valid_lengths = np.array([r[1] for r in valid_runs], dtype=np.float32)
    segment_px = float(np.median(valid_lengths))
    segment_count = len(valid_runs)

    # ── 8. Bounding rows of scale region (in full-image coords) ─────────
    scale_top_y    = valid_runs[0][2] + y_offset
    scale_bottom_y = valid_runs[-1][2] + valid_runs[-1][1] - 1 + y_offset

    # ── 9. Confidence = 1 - CoV (coefficient of variation) ──────────────
    mean_len = float(valid_lengths.mean())
    std_len  = float(valid_lengths.std())
    cov = (std_len / mean_len) if mean_len > 0 else 1.0
    confidence = float(np.clip(1.0 - cov, 0.0, 1.0))

    # ── Annotated strip preview ──────────────────────────────────────────
    strip_bgr = _annotate_strip(strip, scale_top_y - y_offset, scale_bottom_y - y_offset, confidence)

    return DetectionResult(
        segment_px     = segment_px,
        segment_count  = segment_count,
        scale_top_y    = int(scale_top_y),
        scale_bottom_y = int(scale_bottom_y),
        confidence     = confidence,
        strip_bgr      = strip_bgr,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rle(binary: np.ndarray) -> list[list]:
    """
    Run-length encode a 1-D binary array.
    Returns list of [value, length, start_index].
    """
    runs = []
    if len(binary) == 0:
        return runs
    current_val = binary[0]
    start = 0
    for i in range(1, len(binary)):
        if binary[i] != current_val:
            runs.append([int(current_val), i - start, start])
            current_val = binary[i]
            start = i
    runs.append([int(current_val), len(binary) - start, start])
    return runs


def _failed_result(strip: np.ndarray, y_offset: int = 0) -> DetectionResult:
    h = strip.shape[0]
    return DetectionResult(
        segment_px     = 0.0,
        segment_count  = 0,
        scale_top_y    = y_offset,
        scale_bottom_y = y_offset + h - 1,
        confidence     = 0.0,
        strip_bgr      = strip.copy(),
    )


def _annotate_strip(
    strip: np.ndarray,
    top_y: int,
    bottom_y: int,
    confidence: float,
) -> np.ndarray:
    """Draw a coloured rectangle on the strip to indicate the detected region."""
    annotated = strip.copy()
    if confidence >= CONF_GOOD:
        colour = (0, 200, 0)      # green
    elif confidence >= CONF_WARN:
        colour = (0, 165, 255)    # orange
    else:
        colour = (0, 0, 200)      # red
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, top_y), (w - 1, bottom_y), colour, 2)
    return annotated
