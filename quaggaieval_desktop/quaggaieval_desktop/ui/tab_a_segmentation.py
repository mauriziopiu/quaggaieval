"""
Tab 2 — SAM2-assisted Segmentation

Interaction:
  Left-click            → positive (foreground) point
  Shift + Left-click    → negative (background) point
  Middle-drag           → pan  (handled by ImageCanvas)
  Scroll                → zoom (handled by ImageCanvas)
  Double-click          → reset zoom (handled by ImageCanvas)

  Keyboard (canvas must have focus):
    A  — accept stroke (merge into accumulated mask)
    C  — clear all (stroke + accumulated mask)
    L  — undo last point
    S  — confirm / save & advance

After each point placement SAM2 runs a prediction synchronously and the
result is painted as a semi-transparent red overlay on the canvas.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import wx

from .canvas import ImageCanvas
from ..core.state import AppState, StepStatus
from ..core.session_state import SessionState

# Overlay appearance (matches original script)
OVERLAY_COLOUR_RGB = (255, 0, 0)   # red in RGB
OVERLAY_ALPHA      = 0.35


class SegmentationPanel(wx.Panel):

    # Splitter constants (shared across all tabs)
    _SASH_DEFAULT = 300   # right-panel default width (px)
    _SASH_MIN     = 240   # minimum right-panel width  (px)

    @staticmethod
    def _make_grid(parent, rows):
        """
        Build a 2-column FlexGridSizer from a list of (key, value) string pairs.
        Returns the sizer.
        """
        grid = wx.FlexGridSizer(cols=2, vgap=3, hgap=12)
        grid.AddGrowableCol(1, 1)
        dim = wx.Colour(160, 160, 160)
        for key, val in rows:
            lk = wx.StaticText(parent, label=key)
            lk.SetForegroundColour(dim)
            lv = wx.StaticText(parent, label=val)
            lv.SetForegroundColour(dim)
            grid.Add(lk, flag=wx.ALIGN_LEFT)
            grid.Add(lv, flag=wx.ALIGN_LEFT)
        return grid

    def __init__(self, parent: wx.Window, state: AppState, **kwargs):
        super().__init__(parent, **kwargs)
        self._state = state
        self._on_complete_callback = None

        # Prediction guard — prevents queuing a second prediction while one runs
        self._predicting = False

        self._overlay_alpha: float = 0.4

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_complete(self, callback) -> None:
        """callback() called when the user confirms this step."""
        self._on_complete_callback = callback

    def load_image(self) -> None:
        """
        Called by MainFrame whenever a new image is selected.
        Resets session state, draws the image, then triggers SAM2 encoding.
        Existing mask is restored if the image was previously processed.
        """
        session = self._state.session
        session.clear_all()

        # Restore accumulated mask if this image was already processed
        record = self._state.current_record
        if record and record.segmentation_status == StepStatus.COMPLETE:
            self._try_restore_mask(record)

        self._canvas.set_image(self._state.current_image)
        self._refresh_overlays()
        self._update_controls()
        self._set_status("Encoding image…")
        self._canvas.SetFocus()

        # Kick off SAM2 image encoding in the background
        self._state.sam2_worker.encode_image(
            self._state.current_image,
            on_ready=self._on_encoding_ready,
            on_error=self._on_encoding_error,
        )

    def reset(self) -> None:
        self._state.session.clear_all()
        self._canvas.clear()
        self._update_controls()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Splitter: canvas left, controls right ─────────────────────
        self._splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self._splitter.SetMinimumPaneSize(self._SASH_MIN)

        self._canvas = ImageCanvas(self._splitter)
        self._canvas.Bind(wx.EVT_LEFT_DOWN, self._on_left_click)
        self._canvas.Bind(wx.EVT_KEY_DOWN,  self._on_key)
        self._canvas.SetWindowStyle(self._canvas.GetWindowStyle() | wx.WANTS_CHARS)

        ctrl_panel = wx.Panel(self._splitter)
        ctrl = wx.BoxSizer(wx.VERTICAL)

        # ── Interaction box (2-column grid) ───────────────────────────
        instr_box   = wx.StaticBox(ctrl_panel, label="Interaction")
        instr_sizer = wx.StaticBoxSizer(instr_box, wx.VERTICAL)
        mouse_rows = [
            ("Left-click",       "+ point"),
            ("Shift + click",    "− point"),
            ("Middle-drag",      "pan"),
            ("Scroll",           "zoom"),
            ("Dbl-click",        "reset zoom"),
        ]
        key_rows = [
            ("A",  "accept stroke"),
            ("C",  "clear all"),
            ("L",  "undo last point"),
            ("S",  "confirm & save"),
        ]
        instr_sizer.Add(self._make_grid(ctrl_panel, mouse_rows),
                        flag=wx.EXPAND | wx.ALL, border=6)
        instr_sizer.Add(wx.StaticLine(ctrl_panel), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=4)
        instr_sizer.Add(self._make_grid(ctrl_panel, key_rows),
                        flag=wx.EXPAND | wx.ALL, border=6)
        ctrl.Add(instr_sizer, flag=wx.EXPAND | wx.ALL, border=8)

        # ── Actions box ───────────────────────────────────────────────
        act_box   = wx.StaticBox(ctrl_panel, label="Actions")
        act_sizer = wx.StaticBoxSizer(act_box, wx.VERTICAL)

        self._undo_btn = wx.Button(ctrl_panel, label="Undo last point  [L]")
        self._undo_btn.Bind(wx.EVT_BUTTON, self._on_undo)
        act_sizer.Add(self._undo_btn, flag=wx.EXPAND | wx.ALL, border=3)

        self._accept_btn = wx.Button(ctrl_panel, label="Accept stroke  [A]")
        self._accept_btn.Bind(wx.EVT_BUTTON, self._on_accept_stroke)
        act_sizer.Add(self._accept_btn, flag=wx.EXPAND | wx.ALL, border=3)

        self._clear_btn = wx.Button(ctrl_panel, label="Clear all  [C]")
        self._clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_all)
        act_sizer.Add(self._clear_btn, flag=wx.EXPAND | wx.ALL, border=3)

        ctrl.Add(act_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        # ── Overlay opacity ───────────────────────────────────────────
        ov_box   = wx.StaticBox(ctrl_panel, label="Overlay")
        ov_sizer = wx.StaticBoxSizer(ov_box, wx.VERTICAL)

        ov_row = wx.BoxSizer(wx.HORIZONTAL)
        ov_lbl = wx.StaticText(ctrl_panel, label="Opacity")
        ov_lbl.SetForegroundColour(wx.Colour(160, 160, 160))
        self._opacity_slider = wx.Slider(
            ctrl_panel, minValue=0, maxValue=10, value=4,
            style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
        )
        self._opacity_slider.SetTickFreq(1)
        self._opacity_label = wx.StaticText(ctrl_panel, label="0.4")
        self._opacity_label.SetMinSize((30, -1))
        ov_row.Add(ov_lbl,                proportion=0, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        ov_row.Add(self._opacity_slider,  proportion=1, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=4)
        ov_row.Add(self._opacity_label,   proportion=0, flag=wx.ALIGN_CENTER_VERTICAL)
        ov_sizer.Add(ov_row, flag=wx.EXPAND | wx.ALL, border=6)

        self._opacity_slider.Bind(wx.EVT_SLIDER, self._on_opacity_slider)
        ctrl.Add(ov_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        # ── Status + separator ────────────────────────────────────────
        self._status_label = wx.StaticText(ctrl_panel, label="No image loaded.")
        self._status_label.SetForegroundColour(wx.Colour(140, 140, 140))
        ctrl.Add(self._status_label, flag=wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=10)

        ctrl.Add(wx.StaticLine(ctrl_panel), flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=16)
        ctrl.AddStretchSpacer()

        # ── Confirm ───────────────────────────────────────────────────
        self._confirm_btn = wx.Button(ctrl_panel, label="Confirm & next step  [S]")
        self._confirm_btn.Bind(wx.EVT_BUTTON, self._on_confirm)
        self._confirm_btn.Disable()
        ctrl.Add(self._confirm_btn, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        ctrl_panel.SetSizer(ctrl)

        self._splitter.SplitVertically(self._canvas, ctrl_panel)
        self._splitter.SetSashGravity(1.0)   # canvas absorbs all resize

        # Defer sash placement until the window has been laid out
        def _set_sash(_evt=None):
            w = self._splitter.GetClientSize().width
            self._splitter.SetSashPosition(w - self._SASH_DEFAULT)
        self._splitter.Bind(wx.EVT_SIZE, _set_sash)

        root = wx.BoxSizer(wx.VERTICAL)
        root.Add(self._splitter, proportion=1, flag=wx.EXPAND)
        self.SetSizer(root)

    # ------------------------------------------------------------------
    # SAM2 encoding callbacks (called via wx.CallAfter from worker thread)
    # ------------------------------------------------------------------

    def _on_encoding_ready(self) -> None:
        self._set_status("Ready — click to add points.")
        self._update_controls()

    def _on_encoding_error(self, msg: str) -> None:
        self._set_status(f"Encoding error:\n{msg}")

    # ------------------------------------------------------------------
    # Point interaction
    # ------------------------------------------------------------------

    def _on_left_click(self, event: wx.MouseEvent) -> None:
        if not self._state.sam2_worker.image_encoded or self._predicting:
            event.Skip()
            return

        coords = self._canvas.canvas_to_image_coords(*event.GetPosition())
        if coords is None:
            event.Skip()
            return

        # Convert normalised → pixel coords for SAM2
        h, w = self._state.current_image.shape[:2]
        px = int(coords[0] * w)
        py = int(coords[1] * h)

        is_negative = event.ShiftDown()
        label = 0 if is_negative else 1

        self._state.session.stroke_points.append((px, py))
        self._state.session.stroke_labels.append(label)

        self._run_prediction()
        event.Skip()

    def _run_prediction(self) -> None:
        """Run SAM2 prediction and refresh display."""
        if self._predicting:
            return
        self._predicting = True
        self._set_status("Predicting…")

        success = self._state.sam2_worker.predict(self._state.session)

        self._predicting = False
        self._refresh_overlays()
        self._update_controls()

        if success and self._state.session.stroke_mask is not None:
            h, w = self._state.current_image.shape[:2]
            px = self._state.session.stroke_mask.sum()
            coverage = 100.0 * px / (h * w)
            self._set_status(
                f"{self._state.session.stroke_point_count} point(s)\n"
                f"Stroke coverage: {coverage:.1f}%"
            )
        else:
            self._set_status(f"{self._state.session.stroke_point_count} point(s)")

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _on_key(self, event: wx.KeyEvent) -> None:
        key = event.GetKeyCode()
        if key == ord('A'):
            self._on_accept_stroke(None)
        elif key == ord('C'):
            self._on_clear_all(None)
        elif key == ord('L'):
            self._on_undo(None)
        elif key == ord('S'):
            if self._confirm_btn.IsEnabled():
                self._on_confirm(None)
        else:
            event.Skip()

    # ------------------------------------------------------------------
    # Stroke / mask controls
    # ------------------------------------------------------------------

    def _on_accept_stroke(self, _event) -> None:
        self._state.session.accept_stroke()
        self._refresh_overlays()
        self._update_controls()
        h, w = self._state.current_image.shape[:2]
        if self._state.session.accumulated_mask is not None:
            px = self._state.session.accumulated_mask.sum()
            coverage = 100.0 * px / (h * w)
            self._set_status(f"Stroke accepted.\nTotal coverage: {coverage:.1f}%")
        self._canvas.SetFocus()

    def _on_undo(self, _event) -> None:
        removed = self._state.session.undo_last_point()
        if removed:
            if self._state.session.stroke_points:
                self._run_prediction()
            else:
                self._state.session.stroke_mask = None
                self._refresh_overlays()
            self._update_controls()
        self._canvas.SetFocus()

    def _on_clear_all(self, _event) -> None:
        self._state.session.clear_all()
        self._refresh_overlays()
        self._update_controls()
        self._set_status("Cleared.")
        self._canvas.SetFocus()

    def _on_confirm(self, _event) -> None:
        """Auto-accept any open stroke, save results, advance."""
        session = self._state.session
        if session.stroke_mask is not None:
            session.accept_stroke()

        if not session.has_any_mask:
            wx.MessageBox(
                "No mask to save.\nAdd points and accept at least one stroke first.",
                "Nothing to save",
                wx.OK | wx.ICON_WARNING,
            )
            return

        self._save_results()

        if self._state.current_record:
            self._state.current_record.segmentation_status = StepStatus.COMPLETE
        if self._on_complete_callback:
            self._on_complete_callback()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        internal_dir = self._state.internal_dir()
        if internal_dir is None:
            return

        record = self._state.current_record
        if record is None:
            return

        session = self._state.session
        base = record.path.stem

        # Binary mask PNG (0 or 255) with max compression
        mask_uint8 = (session.accumulated_mask > 0).astype(np.uint8) * 255
        mask_path = internal_dir / f"{base}_segmentation_mask.png"
        cv2.imwrite(str(mask_path), mask_uint8,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        # Store mask in AppState for downstream tabs
        self._state.segmentation_mask = mask_uint8

    # ------------------------------------------------------------------
    # Mask restoration (resume support)
    # ------------------------------------------------------------------

    def _try_restore_mask(self, record) -> None:
        internal_dir = self._state.internal_dir()
        if internal_dir is None:
            return
        mask_path = internal_dir / f"{record.path.stem}_segmentation_mask.png"
        if not mask_path.exists():
            return
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        binary = (mask > 0).astype(np.uint8)
        self._state.session.accumulated_mask = binary
        # Also populate AppState so Tab 3 can access the mask without re-saving
        self._state.segmentation_mask = mask
        # Enable confirm immediately — encoding still runs so the user can
        # refine the mask by adding more points.
        self._confirm_btn.Enable()

    # ------------------------------------------------------------------
    # Overlay opacity
    # ------------------------------------------------------------------

    def _on_opacity_slider(self, _evt) -> None:
        self._overlay_alpha = self._opacity_slider.GetValue() / 10.0
        self._opacity_label.SetLabel(f"{self._overlay_alpha:.1f}")
        self._refresh_overlays()

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _refresh_overlays(self) -> None:
        self._canvas.clear_overlays()
        session = self._state.session
        image = self._state.current_image

        if image is None:
            return

        # Build a snapshot so the closure captures fixed data
        display_mask = session.display_mask
        points_snap  = list(zip(session.stroke_points, session.stroke_labels))
        img_h, img_w = image.shape[:2]

        def painter(dc: wx.DC, _cw, _ch):
            # ── Mask overlay ──────────────────────────────────────────
            if display_mask is not None:
                # Build RGBA bitmap: red channel with alpha
                rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
                where = display_mask > 0
                rgba[where] = [255, 0, 0, int(self._overlay_alpha * 255)]

                # Compose onto a copy of the displayed bitmap region
                # We draw the overlay as a scaled bitmap matching the canvas transform
                cw_canvas, ch_canvas = self._canvas.GetClientSize()
                scaled_w = int(img_w * self._canvas._zoom)
                scaled_h = int(img_h * self._canvas._zoom)
                ox = int(self._canvas._offset[0])
                oy = int(self._canvas._offset[1])

                # Scale mask to display size
                mask_small = (display_mask * 255).astype(np.uint8)
                mask_scaled = cv2.resize(
                    mask_small, (scaled_w, scaled_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Build wx.Bitmap for the red overlay
                red_layer = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
                red_layer[mask_scaled > 0] = [255, 0, 0]
                alpha_layer = np.zeros((scaled_h, scaled_w), dtype=np.uint8)
                alpha_layer[mask_scaled > 0] = int(self._overlay_alpha * 255)

                overlay_bmp = wx.Bitmap.FromBufferRGBA(
                    scaled_w, scaled_h,
                    np.dstack([red_layer, alpha_layer]).astype(np.uint8).tobytes()
                )
                dc.DrawBitmap(overlay_bmp, ox, oy, useMask=True)

            # ── Points ───────────────────────────────────────────────
            for (px, py), label in points_snap:
                nx, ny = px / img_w, py / img_h
                cx, cy = self._canvas.image_to_canvas_coords(nx, ny)
                colour = wx.Colour(0, 230, 0) if label == 1 else wx.Colour(230, 0, 0)
                dc.SetPen(wx.Pen(wx.Colour(255, 255, 255), 1))
                dc.SetBrush(wx.Brush(colour))
                dc.DrawCircle(cx, cy, 6)

        self._canvas.add_overlay(painter)
        self._canvas.Refresh()

    # ------------------------------------------------------------------
    # Point list
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Control state
    # ------------------------------------------------------------------

    def _update_controls(self) -> None:
        session = self._state.session
        encoded = self._state.sam2_worker.image_encoded

        has_stroke_pts  = session.stroke_point_count > 0
        has_stroke_mask = session.stroke_mask is not None
        has_accumulated = session.accumulated_mask is not None

        self._undo_btn.Enable(has_stroke_pts)
        self._accept_btn.Enable(has_stroke_mask)
        self._clear_btn.Enable(has_stroke_pts or has_accumulated)
        self._confirm_btn.Enable(
            encoded and (has_accumulated or has_stroke_mask)
        )

    def _set_status(self, text: str) -> None:
        self._status_label.SetLabel(text)
        self._status_label.Wrap(190)