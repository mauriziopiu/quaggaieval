"""
ScaleCalibration — px/cm ratio derived from the reference scale bar.

Persisted as `evaluation_results/.internal/calibration.json`.
Loaded automatically when a folder is opened; triggers the calibration modal
if absent. Legacy path `{folder}/calibration.json` is checked as a fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScaleCalibration:
    """Immutable calibration result for one image folder."""

    px_per_cm: float
    segment_px: float                     # median pixels per scale segment
    segment_cm: float                     # user-supplied cm per segment
    segment_count: int                    # number of segments used
    default_area_cm: tuple[float, float]  # (width_cm, height_cm) for default square
    source_image: str                     # filename of the image used for calibration

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------

    def cm_to_px(self, cm: float) -> float:
        return cm * self.px_per_cm

    def px_to_cm(self, px: float) -> float:
        return px / self.px_per_cm

    # Area of Pixels to Area of cm2
    def px_to_cm2(self, px_area: float) -> float:
        return px_area / (self.px_per_cm ** 2)

    # Area of cm2 to Area of Pixels
    def cm2_to_px(self, cm2_area: float) -> float:
        return cm2_area * (self.px_per_cm ** 2)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "px_per_cm":       self.px_per_cm,
            "segment_px":      self.segment_px,
            "segment_cm":      self.segment_cm,
            "segment_count":   self.segment_count,
            "default_area_cm": list(self.default_area_cm),
            "source_image":    self.source_image,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScaleCalibration:
        return cls(
            px_per_cm       = float(d["px_per_cm"]),
            segment_px      = float(d["segment_px"]),
            segment_cm      = float(d["segment_cm"]),
            segment_count   = int(d["segment_count"]),
            default_area_cm = tuple(d["default_area_cm"]),  # type: ignore[arg-type]
            source_image    = str(d["source_image"]),
        )

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ScaleCalibration | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return None
