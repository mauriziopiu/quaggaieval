"""
SAM2Worker — thin wrapper around SAM2ImagePredictor.

Responsibilities:
  • Load the model once on a background thread
  • Encode a new image (also on a background thread)
  • Run point-based prediction synchronously (fast enough, ~50ms on MPS/CUDA)
  • Guard against concurrent calls with a simple lock

All expensive operations post their results back to the wx main thread
via wx.CallAfter(callback, result).
"""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .model_config import ModelConfig
from .session_state import SessionState


class SAM2Worker:
    """
    Thread-safe SAM2 model wrapper.

    Usage pattern:
        worker = SAM2Worker()
        worker.load_model(config, on_ready=..., on_error=...)
        # later, once model is ready:
        worker.encode_image(cv2_bgr_image, on_ready=..., on_error=...)
        # after encoding:
        worker.predict(session_state)   # synchronous, call from main thread
    """

    def __init__(self) -> None:
        self._predictor = None
        self._model_ready   = False
        self._image_encoded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public state queries
    # ------------------------------------------------------------------

    @property
    def model_ready(self) -> bool:
        return self._model_ready

    @property
    def image_encoded(self) -> bool:
        return self._image_encoded

    # ------------------------------------------------------------------
    # Model loading (background thread)
    # ------------------------------------------------------------------

    def load_model(
        self,
        config: ModelConfig,
        on_ready: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None:
        """Start model loading on a daemon thread. Calls on_ready or on_error
        on the wx main thread via wx.CallAfter."""
        self._model_ready   = False
        self._image_encoded = False
        t = threading.Thread(
            target=self._load_model_thread,
            args=(config, on_ready, on_error),
            daemon=True,
        )
        t.start()

    def _load_model_thread(
        self,
        config: ModelConfig,
        on_ready: Callable,
        on_error: Callable,
    ) -> None:
        import wx
        try:
            # SAM2 imports are deferred so the app can start without them
            # installed — the error only surfaces when loading is attempted.
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except ImportError as exc:
            wx.CallAfter(on_error, (
                "SAM2 not installed.\n\n"
                "Clone https://github.com/facebookresearch/sam2\n"
                "and run:  pip install -e '.[demo]'\n\n"
                f"Details: {exc}"
            ))
            return

        try:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps"  if torch.backends.mps.is_available()
                else "cpu"
            )
            ckpt_path = str(config.checkpoint_path)

            # SAM2's build_sam2() resolves config names via Hydra's pkg://sam2
            # search path and ignores absolute filesystem paths. We bypass this
            # by loading the YAML ourselves with OmegaConf and calling the
            # internal builder directly.
            from omegaconf import OmegaConf
            from sam2.modeling.sam2_base import SAM2Base
            from hydra.utils import instantiate

            cfg = OmegaConf.load(str(config.config_path))
            OmegaConf.resolve(cfg)
            sam2_model = instantiate(cfg.model, _recursive_=True)
            sam2_model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=False)["model"],
                strict=False,
            )
            sam2_model = sam2_model.to(device).eval()
            with self._lock:
                self._predictor  = SAM2ImagePredictor(sam2_model)
                self._model_ready = True
            wx.CallAfter(on_ready)
        except Exception as exc:
            wx.CallAfter(on_error, str(exc))

    # ------------------------------------------------------------------
    # Image encoding (background thread)
    # ------------------------------------------------------------------

    def encode_image(
        self,
        image_bgr: np.ndarray,
        on_ready: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None:
        """Encode a new image into SAM2's image embedding space.
        Must be called after load_model() completes."""
        if not self._model_ready:
            on_error("Model not loaded yet.")
            return
        self._image_encoded = False
        t = threading.Thread(
            target=self._encode_image_thread,
            args=(image_bgr, on_ready, on_error),
            daemon=True,
        )
        t.start()

    def _encode_image_thread(
        self,
        image_bgr: np.ndarray,
        on_ready: Callable,
        on_error: Callable,
    ) -> None:
        import wx
        import cv2
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                with torch.inference_mode():
                    self._predictor.set_image(image_rgb)
                self._image_encoded = True
            wx.CallAfter(on_ready)
        except Exception as exc:
            wx.CallAfter(on_error, str(exc))

    # ------------------------------------------------------------------
    # Prediction (synchronous — fast enough for interactive use)
    # ------------------------------------------------------------------

    def predict(self, session: SessionState) -> bool:
        """
        Run SAM2 on session.stroke_points/labels. Updates session.stroke_mask.
        Returns True on success, False if preconditions not met.
        Call from the main thread — takes ~50ms on GPU/MPS.
        """
        if not self._model_ready or not self._image_encoded:
            return False
        if not session.stroke_points:
            session.stroke_mask = None
            return True

        points = np.array(session.stroke_points, dtype=np.float32)
        labels = np.array(session.stroke_labels, dtype=np.int32)

        try:
            with self._lock:
                with torch.inference_mode():
                    masks, scores, _ = self._predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=False,
                    )
            best_idx = int(np.argmax(scores))
            session.stroke_mask = masks[best_idx].astype(np.uint8)
            return True
        except Exception as exc:
            print(f"[SAM2Worker] prediction error: {exc}")
            session.stroke_mask = None
            return False