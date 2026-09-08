"""
SAM2 model configuration — defaults and registry.

Bundling layout expected under your project root (development) or inside
sys._MEIPASS (frozen PyInstaller bundle):

    models/
    └── sam2/
        ├── sam2.1_hiera_base_plus.pt   (or whichever checkpoint you ship)
        └── configs/
            └── sam2.1/
                ├── sam2.1_hiera_b+.yaml
                ├── sam2.1_hiera_t.yaml
                ├── sam2.1_hiera_s.yaml
                └── sam2.1_hiera_l.yaml

Development setup:
    Copy the configs from the sam2 repo:
        cp -r <sam2_repo>/sam2/configs/sam2.1  models/sam2/configs/
    The sam2 Python package must be installed separately:
        pip install -e <sam2_repo>

Frozen bundle:
    PyInstaller places everything under sys._MEIPASS.
    _resource_root() detects this and returns the correct base directory
    automatically — no user configuration required.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path


def _resource_root() -> Path:
    """
    Return the base directory for bundled resources.

    - Inside a PyInstaller frozen bundle: sys._MEIPASS
      (the temp directory where the bundle is extracted at launch)
    - During normal development: the project root
      (two levels up from this file: quaggaieval_desktop/core/model_config.py
       -> quaggaieval_desktop/ -> project_root/)
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    # Development: go up two directories from this file
    return Path(__file__).parent.parent.parent.resolve()


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR:  str = "models/sam2"
DEFAULT_MODEL_SIZE: str = "quaggai_v2"

# Maps model-size key -> (config filename, checkpoint filename)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # "tiny":      ("sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    # "small":     ("sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base_plus": ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    # "large":     ("sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
    "quaggai_v1": ("sam2.1_hiera_b+.yaml", "quaggai_v1.pt"),
    "quaggai_v2": ("sam2.1_hiera_b+.yaml", "quaggai_v2.pt"),
}

MODEL_SIZE_LABELS: dict[str, str] = {
    # "tiny":      "Tiny   (fastest, lowest quality)",
    # "small":     "Small",
    "base_plus": "Base+  (default)",
    # "large":     "Large  (slowest, highest quality)",
    "quaggai_v1": "Fine-Tuned QuaggAI Model v1",
    "quaggai_v2": "Fine-Tuned QuaggAI Model v2 (recommended)",
}


@dataclass
class ModelConfig:
    """User-editable model settings, persisted across the session in AppState."""
    model_size:          str = DEFAULT_MODEL_SIZE
    model_dir:           str = DEFAULT_MODEL_DIR
    # If set, overrides the default checkpoint filename derived from model_size.
    checkpoint_override: str = ""

    def _resolved_model_dir(self) -> Path:
        """
        Resolve model_dir to an absolute path.

        model_dir is stored as a relative string (e.g. "models/sam2") so it
        works in both development and inside the frozen bundle — _resource_root()
        provides the correct base in each context.

        If the user has set an absolute path (e.g. via the Settings panel), it
        is returned as-is.
        """
        p = Path(self.model_dir)
        if p.is_absolute():
            return p
        return _resource_root() / p

    @property
    def config_path(self) -> Path:
        """Absolute path to the Hydra config YAML."""
        cfg_filename, _ = MODEL_REGISTRY[self.model_size]
        return self._resolved_model_dir() / "configs" / "sam2.1" / cfg_filename

    @property
    def checkpoint_path(self) -> Path:
        """Absolute path to the model checkpoint."""
        if self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        _, ckpt_name = MODEL_REGISTRY[self.model_size]
        return self._resolved_model_dir() / ckpt_name

    def is_valid(self) -> tuple[bool, str]:
        """Returns (ok, error_message). ok=True means ready to load."""
        if self.model_size not in MODEL_REGISTRY:
            return False, f"Unknown model size: {self.model_size}"
        ckpt = self.checkpoint_path
        if not ckpt.exists():
            return False, f"Checkpoint not found:\n{ckpt}"
        cfg = self.config_path
        if not cfg.exists():
            return False, (
                f"Config YAML not found:\n{cfg}\n\n"
                f"Copy configs from the sam2 repo:\n"
                f"  cp -r <sam2_repo>/sam2/configs/sam2.1  {self.model_dir}/configs/"
            )
        return True, ""