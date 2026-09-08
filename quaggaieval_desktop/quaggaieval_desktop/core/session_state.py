"""
SessionState — mutable state for one image segmentation session.

Lifted directly from sam2_segmentation.py with zero logic changes.
No UI or OpenCV dependencies so it can be unit-tested independently.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SessionState:
    """Mutable state for one image labelling session."""

    # Points for the *current stroke* (not yet merged into accumulated mask)
    # Stored as image-pixel coordinates (int, int) matching SAM2's expected input.
    stroke_points: list[tuple[int, int]] = field(default_factory=list)
    stroke_labels: list[int]             = field(default_factory=list)  # 1=pos, 0=neg

    # Mask from the current (unmerged) stroke prediction — shape (H, W), uint8 0/1
    stroke_mask: Optional[np.ndarray] = None

    # Union of all accepted strokes — shape (H, W), uint8 0/1
    accumulated_mask: Optional[np.ndarray] = None

    def clear_stroke(self) -> None:
        self.stroke_points.clear()
        self.stroke_labels.clear()
        self.stroke_mask = None

    def clear_all(self) -> None:
        self.clear_stroke()
        self.accumulated_mask = None

    def undo_last_point(self) -> bool:
        """Remove the last point. Returns True if a point was removed."""
        if not self.stroke_points:
            return False
        self.stroke_points.pop()
        self.stroke_labels.pop()
        return True

    def accept_stroke(self) -> None:
        """OR-merge the current stroke mask into the accumulated mask."""
        if self.stroke_mask is None:
            return
        if self.accumulated_mask is None:
            self.accumulated_mask = self.stroke_mask.copy()
        else:
            self.accumulated_mask = np.logical_or(
                self.accumulated_mask, self.stroke_mask
            ).astype(np.uint8)
        self.clear_stroke()

    @property
    def display_mask(self) -> Optional[np.ndarray]:
        """Combined view: accumulated OR current stroke, for live display."""
        masks = [m for m in (self.accumulated_mask, self.stroke_mask) if m is not None]
        if not masks:
            return None
        result = masks[0].copy()
        for m in masks[1:]:
            result = np.logical_or(result, m).astype(np.uint8)
        return result

    @property
    def has_any_mask(self) -> bool:
        return self.accumulated_mask is not None or self.stroke_mask is not None

    @property
    def stroke_point_count(self) -> int:
        return len(self.stroke_points)