from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
import json
import numpy as np

from .calibration import ScaleCalibration
from .migration import detect_old_format
from .model_config import ModelConfig
from .sam2_worker import SAM2Worker
from .session_state import SessionState


class StepStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()


@dataclass
class ImageRecord:
    """Tracks processing state for a single image."""
    path: Path
    segmentation_status: StepStatus = StepStatus.PENDING
    eval_area_status: StepStatus = StepStatus.PENDING
    mask_correction_status: StepStatus = StepStatus.PENDING
    evaluation_status: StepStatus = StepStatus.PENDING

    @property
    def first_incomplete_step(self) -> int:
        """Returns 0-indexed tab number of the first incomplete step."""
        if self.eval_area_status != StepStatus.COMPLETE:
            return 0
        if self.segmentation_status != StepStatus.COMPLETE:
            return 1
        if self.mask_correction_status != StepStatus.COMPLETE:
            return 2
        return 3

    @property
    def is_fully_complete(self) -> bool:
        return (
            self.segmentation_status == StepStatus.COMPLETE
            and self.eval_area_status == StepStatus.COMPLETE
            and self.mask_correction_status == StepStatus.COMPLETE
            and self.evaluation_status == StepStatus.COMPLETE
        )


@dataclass
class AppState:
    """Central shared state passed to all UI components."""
    folder_path: Path | None = None
    image_records: list[ImageRecord] = field(default_factory=list)
    current_index: int = -1

    # Current image data (loaded on demand)
    current_image: np.ndarray | None = None

    # Masks — populated as steps complete
    segmentation_mask: np.ndarray | None = None
    eval_area_mask: np.ndarray | None = None

    # SAM2 — shared across all images
    model_config: ModelConfig = field(default_factory=ModelConfig)
    sam2_worker: SAM2Worker = field(default_factory=SAM2Worker)

    # Per-image segmentation session state
    session: SessionState = field(default_factory=SessionState)

    # Scale calibration — loaded from .internal/calibration.json on folder open
    calibration: ScaleCalibration | None = None

    # Set by load_folder(); consumed by main_frame to trigger migration flow
    needs_migration: bool = False

    IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    @property
    def current_record(self) -> ImageRecord | None:
        if 0 <= self.current_index < len(self.image_records):
            return self.image_records[self.current_index]
        return None

    @property
    def has_images(self) -> bool:
        return len(self.image_records) > 0

    def load_folder(self, folder_path: Path) -> None:
        self.folder_path = folder_path
        self.image_records = [
            ImageRecord(path=p)
            for p in sorted(folder_path.iterdir())
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        self.current_index = 0 if self.image_records else -1
        self.current_image = None
        self.segmentation_mask = None
        self.eval_area_mask = None
        self.session = SessionState()
        # Try new internal path first; fall back to legacy folder-root path
        internal = folder_path / "evaluation_results" / ".internal"
        self.calibration = ScaleCalibration.load(internal / "calibration.json")
        if self.calibration is None:
            self.calibration = ScaleCalibration.load(folder_path / "calibration.json")
        self._restore_statuses()
        self.needs_migration = detect_old_format(folder_path)

    def _restore_statuses(self) -> None:
        """
        Scan the output directory for previously saved files and restore each
        ImageRecord's step statuses, so resuming a session picks up where it
        left off without re-processing already-completed images.

        The mask correction step overwrites the segmentation mask in place, so
        its completion is inferred from the segmentation mask's presence (the
        user must pass through the correction tab to reach evaluation).
        """
        out = self.folder_path / "evaluation_results"
        if not out.exists():
            return
        internal = out / ".internal"

        # Load consolidated eval areas JSON once to check per-image completions
        eval_areas: dict = {}
        eval_areas_path = internal / "eval_areas.json"
        if eval_areas_path.exists():
            try:
                with open(eval_areas_path) as f:
                    eval_areas = json.load(f)
            except Exception:
                pass

        cropped_dir = out / "cropped"

        for record in self.image_records:
            stem = record.path.stem
            # Tab 1: entry keyed by full filename in eval_areas.json
            if record.path.name in eval_areas:
                record.eval_area_status = StepStatus.COMPLETE
            # Tab 2 & 3: internal mask PNG
            if (internal / f"{stem}_segmentation_mask.png").exists():
                record.segmentation_status = StepStatus.COMPLETE
                record.mask_correction_status = StepStatus.COMPLETE
            # Tab 4: cropped output PNG
            if (cropped_dir / f"{stem}_cropped.png").exists():
                record.evaluation_status = StepStatus.COMPLETE

    def select_image(self, index: int) -> None:
        if 0 <= index < len(self.image_records):
            self.current_index = index
            self.current_image = None
            self.segmentation_mask = None
            self.eval_area_mask = None
            self.session = SessionState()

    def next_unprocessed_index(self) -> int | None:
        """Returns index of next image that is not fully complete, or None."""
        for i, record in enumerate(self.image_records):
            if i > self.current_index and not record.is_fully_complete:
                return i
        return None

    def output_dir(self) -> Path | None:
        if self.folder_path is None:
            return None
        path = self.folder_path / "evaluation_results"
        path.mkdir(exist_ok=True)
        return path

    def internal_dir(self) -> Path | None:
        """Returns evaluation_results/.internal/, creating it if needed."""
        out = self.output_dir()
        if out is None:
            return None
        path = out / ".internal"
        path.mkdir(exist_ok=True)
        return path