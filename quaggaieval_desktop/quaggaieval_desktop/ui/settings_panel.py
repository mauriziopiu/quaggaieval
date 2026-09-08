"""
SettingsPanel — full-area subpage for SAM2 model configuration.

Swapped in/out by MainFrame over the notebook. Never shown simultaneously
with the main content — think of it as a settings "page" on a website.
"""
from __future__ import annotations
from pathlib import Path
import wx

from ..core.model_config import MODEL_REGISTRY, MODEL_SIZE_LABELS, ModelConfig


class SettingsPanel(wx.Panel):

    def __init__(self, parent: wx.Window, config: ModelConfig, **kwargs):
        super().__init__(parent, **kwargs)
        self._config = config
        self._on_back_callback = None
        self._on_reload_callback = None
        self.SetBackgroundColour(wx.Colour(40, 40, 40))
        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_back(self, callback) -> None:
        """Called when the user clicks ← Back."""
        self._on_back_callback = callback

    def set_on_reload_model(self, callback) -> None:
        """Called when the user saves settings and wants to reload the model."""
        self._on_reload_callback = callback

    def refresh_from_config(self) -> None:
        """Sync UI fields from the current ModelConfig (call before showing)."""
        self._populate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = wx.BoxSizer(wx.VERTICAL)

        # ── Top bar ───────────────────────────────────────────────────
        topbar = wx.BoxSizer(wx.HORIZONTAL)
        back_btn = wx.Button(self, label="← Back")
        back_btn.Bind(wx.EVT_BUTTON, self._on_back)
        topbar.Add(back_btn, flag=wx.ALL, border=8)

        title = wx.StaticText(self, label="Settings")
        title.SetForegroundColour(wx.Colour(210, 210, 210))
        title.SetFont(title.GetFont().Scaled(1.4).Bold())
        topbar.Add(title, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=8)
        outer.Add(topbar, flag=wx.EXPAND)
        outer.Add(wx.StaticLine(self), flag=wx.EXPAND)

        # ── Settings form (centred) ───────────────────────────────────
        form_wrapper = wx.BoxSizer(wx.HORIZONTAL)
        form_wrapper.AddStretchSpacer()

        form_panel = wx.Panel(self)
        form_panel.SetMinSize((520, -1))
        form_panel.SetBackgroundColour(wx.Colour(40, 40, 40))
        form = wx.BoxSizer(wx.VERTICAL)

        # Section: SAM2 Model
        form.Add(self._section_label(form_panel, "SAM2 Model"), flag=wx.TOP | wx.BOTTOM, border=12)

        # Model size
        form.Add(self._field_label(form_panel, "Model size"), flag=wx.BOTTOM, border=4)
        self._size_choice = wx.Choice(
            form_panel,
            choices=[MODEL_SIZE_LABELS[k] for k in MODEL_REGISTRY],
        )
        self._size_keys = list(MODEL_REGISTRY.keys())
        form.Add(self._size_choice, flag=wx.EXPAND | wx.BOTTOM, border=12)

        # Model directory
        form.Add(self._field_label(form_panel, "Model directory"), flag=wx.BOTTOM, border=4)
        dir_row = wx.BoxSizer(wx.HORIZONTAL)
        self._dir_text = wx.TextCtrl(form_panel)
        dir_row.Add(self._dir_text, proportion=1, flag=wx.ALIGN_CENTER_VERTICAL)
        browse_dir_btn = wx.Button(form_panel, label="Browse…")
        browse_dir_btn.Bind(wx.EVT_BUTTON, self._on_browse_dir)
        dir_row.Add(browse_dir_btn, flag=wx.LEFT, border=6)
        form.Add(dir_row, flag=wx.EXPAND | wx.BOTTOM, border=4)

        hint = wx.StaticText(
            form_panel,
            label="The checkpoint file will be resolved automatically from the\n"
                  "model size and this directory, unless overridden below.",
        )
        hint.SetForegroundColour(wx.Colour(120, 120, 120))
        form.Add(hint, flag=wx.BOTTOM, border=12)

        # Checkpoint override
        form.Add(self._field_label(form_panel, "Checkpoint override (optional)"), flag=wx.BOTTOM, border=4)
        ckpt_row = wx.BoxSizer(wx.HORIZONTAL)
        self._ckpt_text = wx.TextCtrl(form_panel)
        ckpt_row.Add(self._ckpt_text, proportion=1, flag=wx.ALIGN_CENTER_VERTICAL)
        browse_ckpt_btn = wx.Button(form_panel, label="Browse…")
        browse_ckpt_btn.Bind(wx.EVT_BUTTON, self._on_browse_ckpt)
        ckpt_row.Add(browse_ckpt_btn, flag=wx.LEFT, border=6)
        form.Add(ckpt_row, flag=wx.EXPAND | wx.BOTTOM, border=4)

        hint2 = wx.StaticText(
            form_panel,
            label="Leave blank to use the default filename for the selected model size.",
        )
        hint2.SetForegroundColour(wx.Colour(120, 120, 120))
        form.Add(hint2, flag=wx.BOTTOM, border=20)

        # Resolved path display
        self._resolved_label = wx.StaticText(form_panel, label="")
        self._resolved_label.SetForegroundColour(wx.Colour(140, 200, 140))
        form.Add(self._resolved_label, flag=wx.BOTTOM, border=16)

        # Bind live preview of resolved path
        self._size_choice.Bind(wx.EVT_CHOICE, self._on_field_change)
        self._dir_text.Bind(wx.EVT_TEXT, self._on_field_change)
        self._ckpt_text.Bind(wx.EVT_TEXT, self._on_field_change)

        # ── Action buttons ────────────────────────────────────────────
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        save_btn = wx.Button(form_panel, label="Save & reload model")
        save_btn.Bind(wx.EVT_BUTTON, self._on_save_reload)
        btn_row.Add(save_btn)

        save_only_btn = wx.Button(form_panel, label="Save only")
        save_only_btn.Bind(wx.EVT_BUTTON, self._on_save_only)
        btn_row.Add(save_only_btn, flag=wx.LEFT, border=8)
        form.Add(btn_row)

        form_panel.SetSizer(form)
        form_wrapper.Add(form_panel, flag=wx.ALL, border=24)
        form_wrapper.AddStretchSpacer()

        outer.Add(form_wrapper, proportion=1, flag=wx.EXPAND)
        self.SetSizer(outer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_label(parent, text: str) -> wx.StaticText:
        lbl = wx.StaticText(parent, label=text)
        lbl.SetForegroundColour(wx.Colour(210, 210, 210))
        lbl.SetFont(lbl.GetFont().Bold())
        return lbl

    @staticmethod
    def _field_label(parent, text: str) -> wx.StaticText:
        lbl = wx.StaticText(parent, label=text)
        lbl.SetForegroundColour(wx.Colour(170, 170, 170))
        return lbl

    def _populate(self) -> None:
        idx = self._size_keys.index(self._config.model_size) if self._config.model_size in self._size_keys else 0
        self._size_choice.SetSelection(idx)
        self._dir_text.SetValue(self._config.model_dir)
        self._ckpt_text.SetValue(self._config.checkpoint_override)
        self._update_resolved_label()

    def _read_fields_into_config(self) -> None:
        idx = self._size_choice.GetSelection()
        self._config.model_size         = self._size_keys[idx] if idx >= 0 else "base_plus"
        self._config.model_dir          = self._dir_text.GetValue().strip()
        self._config.checkpoint_override = self._ckpt_text.GetValue().strip()

    def _update_resolved_label(self) -> None:
        from ..core.model_config import ModelConfig, MODEL_REGISTRY
        idx = self._size_choice.GetSelection()
        size_key = self._size_keys[idx] if idx >= 0 else "base_plus"

        # Build a temporary config from current field values to reuse its path logic
        tmp = ModelConfig(
            model_size=size_key,
            model_dir=self._dir_text.GetValue().strip(),
            checkpoint_override=self._ckpt_text.GetValue().strip(),
        )
        ckpt_path = tmp.checkpoint_path
        cfg_path  = tmp.config_path

        ckpt_ok = ckpt_path.exists()
        cfg_ok  = cfg_path.exists()

        lines = [
            f"Checkpoint: {ckpt_path}",
            "  ✓ found" if ckpt_ok else "  ✗ not found",
            f"Config:     {cfg_path}",
            "  ✓ found" if cfg_ok  else "  ✗ not found  (copy configs/ from sam2 repo)",
        ]
        colour = wx.Colour(140, 200, 140) if (ckpt_ok and cfg_ok) else wx.Colour(200, 100, 100)
        self._resolved_label.SetLabel("\n".join(lines))
        self._resolved_label.SetForegroundColour(colour)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_field_change(self, _event) -> None:
        self._update_resolved_label()

    def _on_browse_dir(self, _event) -> None:
        with wx.DirDialog(self, "Select SAM2 model directory",
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self._dir_text.SetValue(dlg.GetPath())

    def _on_browse_ckpt(self, _event) -> None:
        with wx.FileDialog(
            self, "Select SAM2 checkpoint",
            wildcard="PyTorch checkpoints (*.pt;*.pth)|*.pt;*.pth|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self._ckpt_text.SetValue(dlg.GetPath())

    def _on_save_only(self, _event) -> None:
        self._read_fields_into_config()
        wx.MessageBox("Settings saved. Model will be reloaded next time you click\n"
                      "'Save & reload model'.", "Saved", wx.OK | wx.ICON_INFORMATION)

    def _on_save_reload(self, _event) -> None:
        self._read_fields_into_config()
        ok, err = self._config.is_valid()
        if not ok:
            wx.MessageBox(f"Cannot load model:\n\n{err}", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self._on_reload_callback:
            self._on_reload_callback()
        if self._on_back_callback:
            self._on_back_callback()

    def _on_back(self, _event) -> None:
        if self._on_back_callback:
            self._on_back_callback()