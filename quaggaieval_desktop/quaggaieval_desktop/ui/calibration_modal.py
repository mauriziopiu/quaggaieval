"""
CalibrationModal — blocking wx.Dialog for scale calibration.

Flow:
  Page 0 — Measure:
    - Pick reference image from dropdown
    - ImageCanvas with zoom/pan (scroll wheel, middle-drag, double-click to reset)
    - Click to place two endpoint vertices on the scale bar
    - Both vertices are draggable once placed (same interaction as tab_b_eval_area)
    - Right panel: segment count, cm/segment, live scale readout
    - [Reset  ESC]  [Confirm scale →]

  Page 1 — Default Area:
    - Enter default evaluation area size in cm × cm (defaults: 3.5 × 3.5)
    - Shows pixel equivalent live
    - [Save & close]

Vertices are stored as normalised (0..1) image coordinates, matching canvas.py
and tab_b_eval_area.py conventions.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import wx

from ..core.calibration import ScaleCalibration
from .canvas import ImageCanvas

# ── Layout constants ──────────────────────────────────────────────────────────
_DLG_MIN_W = 900
_DLG_MIN_H = 600

_DEFAULT_AREA_CM = 50.0
_DEFAULT_SEG_CM  = 3.5
_DEFAULT_SEG_N   = 11

# Vertex hit-testing threshold (canvas pixels)
_HOVER_THRESHOLD = 12

# Visual constants
_VERTEX_RADIUS  = 7
_VERTEX_COLOUR  = wx.Colour(255, 200, 0)
_VERTEX_HOVER   = wx.Colour(0, 255, 255)
_LINE_COLOUR    = wx.Colour(255, 200, 0)


class CalibrationModal(wx.Dialog):

    def __init__(self, parent: wx.Window, state):
        super().__init__(
            parent,
            title="Scale Calibration",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self._state  = state
        self._folder = state.folder_path

        # Current image
        self._image:      np.ndarray | None = None
        self._image_name: str               = ""

        # Line vertices — normalised (0..1) image coords; at most 2
        self._vertices:        list[tuple[float, float]] = []
        self._hover_vertex:    int | None                = None
        self._dragging_vertex: int | None                = None

        # Confirmed scale
        self._confirmed_px_per_cm:     float | None = None
        self._confirmed_segment_px:    float | None = None
        self._confirmed_segment_count: int   | None = None
        self._confirmed_seg_cm:        float | None = None

        self._build_ui()
        self._apply_dark_theme()
        self.SetMinSize((_DLG_MIN_W, _DLG_MIN_H))
        self.Fit()
        self.Centre()

        wx.CallAfter(self._load_first_image)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(self, label="Scale Calibration")
        title.SetFont(title.GetFont().Bold().Scaled(1.3))
        title.SetForegroundColour(wx.Colour(220, 220, 220))
        root.Add(title, flag=wx.ALL, border=12)
        root.Add(wx.StaticLine(self), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=8)

        self._pages       = wx.Simplebook(self)
        self._page_measure = wx.Panel(self._pages)
        self._page_area    = wx.Panel(self._pages)
        self._pages.AddPage(self._page_measure, "Measure")
        self._pages.AddPage(self._page_area,    "Area")

        self._build_measure_page()
        self._build_area_page()

        root.Add(self._pages, proportion=1, flag=wx.EXPAND | wx.ALL, border=8)
        self.SetSizer(root)

    # ── Page 0: Measure ───────────────────────────────────────────────

    def _build_measure_page(self) -> None:
        p = self._page_measure
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left: image picker + canvas
        left = wx.BoxSizer(wx.VERTICAL)

        sel_row = wx.BoxSizer(wx.HORIZONTAL)
        sel_lbl = wx.StaticText(p, label="Reference image:")
        sel_lbl.SetForegroundColour(wx.Colour(160, 160, 160))
        image_names = [r.path.name for r in self._state.image_records]
        self._image_choice = wx.Choice(p, choices=image_names)
        self._image_choice.SetSelection(0)
        self._image_choice.Bind(wx.EVT_CHOICE, self._on_image_choice)
        sel_row.Add(sel_lbl, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        sel_row.Add(self._image_choice, proportion=1, flag=wx.ALIGN_CENTER_VERTICAL)
        left.Add(sel_row, flag=wx.EXPAND | wx.BOTTOM, border=4)

        instr = wx.StaticText(
            p,
            label="Click to place two points on the scale bar.\n"
                  "Drag either point to reposition it.\n"
                  "Scroll to zoom · Middle-drag to pan · Double-click to reset zoom.",
        )
        instr.SetForegroundColour(wx.Colour(120, 120, 120))
        left.Add(instr, flag=wx.BOTTOM, border=6)

        self._canvas = ImageCanvas(p)
        self._canvas.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self._canvas.Bind(wx.EVT_LEFT_UP,   self._on_left_up)
        self._canvas.Bind(wx.EVT_MOTION,    self._on_motion)
        self._canvas.Bind(wx.EVT_KEY_DOWN,  self._on_key)
        left.Add(self._canvas, proportion=1, flag=wx.EXPAND)

        sizer.Add(left, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=12)

        # Right: controls
        right = wx.BoxSizer(wx.VERTICAL)
        right.SetMinSize((210, -1))

        meas_box   = wx.StaticBox(p, label="Measurement")
        meas_sizer = wx.StaticBoxSizer(meas_box, wx.VERTICAL)

        self._line_px_label = wx.StaticText(p, label="Line: — px")
        self._line_px_label.SetForegroundColour(wx.Colour(160, 160, 160))
        meas_sizer.Add(self._line_px_label, flag=wx.ALL, border=6)

        sc_row = wx.BoxSizer(wx.HORIZONTAL)
        sc_lbl = wx.StaticText(p, label="Segments:")
        sc_lbl.SetForegroundColour(wx.Colour(180, 180, 180))
        self._seg_spin = wx.SpinCtrl(
            p, min=1, max=500, initial=_DEFAULT_SEG_N, size=(70, -1)
        )
        self._seg_spin.Bind(wx.EVT_SPINCTRL, self._on_params_changed)
        sc_row.Add(sc_lbl, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        sc_row.Add(self._seg_spin, flag=wx.ALIGN_CENTER_VERTICAL)
        meas_sizer.Add(sc_row, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=6)

        cm_row = wx.BoxSizer(wx.HORIZONTAL)
        cm_lbl = wx.StaticText(p, label="cm / segment:")
        cm_lbl.SetForegroundColour(wx.Colour(180, 180, 180))
        self._seg_cm_ctrl = wx.TextCtrl(p, value=str(_DEFAULT_SEG_CM), size=(70, -1))
        self._seg_cm_ctrl.Bind(wx.EVT_TEXT, self._on_params_changed)
        cm_row.Add(cm_lbl, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        cm_row.Add(self._seg_cm_ctrl, flag=wx.ALIGN_CENTER_VERTICAL)
        meas_sizer.Add(cm_row, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=6)

        self._scale_label = wx.StaticText(p, label="Scale: —")
        self._scale_label.SetForegroundColour(wx.Colour(100, 220, 100))
        meas_sizer.Add(self._scale_label, flag=wx.LEFT | wx.BOTTOM, border=8)

        right.Add(meas_sizer, flag=wx.EXPAND | wx.BOTTOM, border=12)

        # Vertex status
        self._vertex_label = wx.StaticText(p, label="Points: 0 / 2")
        self._vertex_label.SetForegroundColour(wx.Colour(140, 140, 140))
        right.Add(self._vertex_label, flag=wx.BOTTOM, border=6)

        reset_btn = wx.Button(p, label="Reset  [ESC]")
        reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
        right.Add(reset_btn, flag=wx.EXPAND | wx.BOTTOM, border=16)

        right.AddStretchSpacer()

        self._confirm_btn = wx.Button(p, label="Confirm scale  →")
        self._confirm_btn.Bind(wx.EVT_BUTTON, self._on_confirm)
        self._confirm_btn.Disable()
        right.Add(self._confirm_btn, flag=wx.EXPAND)

        sizer.Add(right, flag=wx.EXPAND, proportion=0)
        p.SetSizer(sizer)

    # ── Page 1: Default Area ──────────────────────────────────────────

    def _build_area_page(self) -> None:
        p = self._page_area
        sizer = wx.BoxSizer(wx.VERTICAL)

        hdr = wx.StaticText(p, label="Default Evaluation Area")
        hdr.SetFont(hdr.GetFont().Bold().Scaled(1.1))
        hdr.SetForegroundColour(wx.Colour(210, 210, 210))
        sizer.Add(hdr, flag=wx.ALL, border=8)

        desc = wx.StaticText(
            p,
            label="Set the default square that will be pre-drawn on each new image.\n"
                  "You can always adjust or redraw it in the Evaluation Area tab.",
        )
        desc.SetForegroundColour(wx.Colour(150, 150, 150))
        desc.Wrap(500)
        sizer.Add(desc, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        self._area_scale_summary = wx.StaticText(p, label="")
        self._area_scale_summary.SetForegroundColour(wx.Colour(100, 200, 100))
        sizer.Add(self._area_scale_summary, flag=wx.LEFT | wx.BOTTOM, border=10)

        sizer.Add(wx.StaticLine(p), flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
                  border=8)

        grid = wx.FlexGridSizer(cols=3, vgap=8, hgap=10)
        dim_col = wx.Colour(180, 180, 180)

        w_lbl = wx.StaticText(p, label="Width (cm):")
        w_lbl.SetForegroundColour(dim_col)
        self._area_w_ctrl = wx.TextCtrl(p, value=str(_DEFAULT_AREA_CM), size=(80, -1))
        self._area_w_ctrl.Bind(wx.EVT_TEXT, self._on_area_changed)
        grid.Add(w_lbl,             flag=wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self._area_w_ctrl, flag=wx.ALIGN_CENTER_VERTICAL)
        grid.AddSpacer(0)

        h_lbl = wx.StaticText(p, label="Height (cm):")
        h_lbl.SetForegroundColour(dim_col)
        self._area_h_ctrl = wx.TextCtrl(p, value=str(_DEFAULT_AREA_CM), size=(80, -1))
        self._area_h_ctrl.Bind(wx.EVT_TEXT, self._on_area_changed)
        grid.Add(h_lbl,             flag=wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self._area_h_ctrl, flag=wx.ALIGN_CENTER_VERTICAL)
        grid.AddSpacer(0)

        sizer.Add(grid, flag=wx.LEFT | wx.BOTTOM, border=10)

        self._area_px_label = wx.StaticText(p, label="→ — px × — px")
        self._area_px_label.SetForegroundColour(wx.Colour(130, 130, 130))
        sizer.Add(self._area_px_label, flag=wx.LEFT | wx.BOTTOM, border=10)

        sizer.AddStretchSpacer()

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        back_btn = wx.Button(p, label="← Back")
        back_btn.Bind(wx.EVT_BUTTON, lambda _: self._pages.SetSelection(0))
        save_btn = wx.Button(p, label="Save & close")
        save_btn.Bind(wx.EVT_BUTTON, self._on_save)
        btn_row.Add(back_btn, flag=wx.RIGHT, border=8)
        btn_row.Add(save_btn)
        sizer.Add(btn_row, flag=wx.ALIGN_RIGHT | wx.ALL, border=10)

        p.SetSizer(sizer)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_first_image(self) -> None:
        records = self._state.image_records
        if records:
            self._load_image(records[0])

    def _load_image(self, record) -> None:
        img = cv2.imread(str(record.path))
        if img is None:
            return
        self._image      = img
        self._image_name = record.path.name
        self._vertices.clear()
        self._hover_vertex    = None
        self._dragging_vertex = None
        self._canvas.set_image(img)
        self._refresh_overlay()
        self._update_side_panel()
        self._canvas.SetFocus()

    def _on_image_choice(self, _event) -> None:
        idx = self._image_choice.GetSelection()
        records = self._state.image_records
        if 0 <= idx < len(records):
            self._load_image(records[idx])

    # ------------------------------------------------------------------
    # Canvas mouse interaction (mirrors tab_b_eval_area.py)
    # ------------------------------------------------------------------

    def _on_left_down(self, event: wx.MouseEvent) -> None:
        cx, cy = event.GetPosition()

        # If a vertex is near the cursor → start dragging it
        vi = self._vertex_near_canvas(cx, cy)
        if vi is not None:
            self._dragging_vertex = vi
            event.Skip()
            return

        # Otherwise place a new vertex (only if fewer than 2 exist)
        if len(self._vertices) < 2:
            coords = self._canvas.canvas_to_image_coords(cx, cy)
            if coords is not None:
                self._vertices.append(coords)
                self._refresh_overlay()
                self._update_side_panel()

        event.Skip()

    def _on_left_up(self, event: wx.MouseEvent) -> None:
        self._dragging_vertex = None
        event.Skip()

    def _on_motion(self, event: wx.MouseEvent) -> None:
        cx, cy = event.GetPosition()

        if self._dragging_vertex is not None and event.LeftIsDown():
            coords = self._canvas.canvas_to_image_coords(cx, cy)
            if coords is not None:
                self._vertices[self._dragging_vertex] = coords
                self._refresh_overlay()
                self._update_side_panel()
            event.Skip()
            return

        # Update hover highlight
        prev = self._hover_vertex
        self._hover_vertex = self._vertex_near_canvas(cx, cy)
        if self._hover_vertex != prev:
            self._refresh_overlay()

        event.Skip()

    def _on_key(self, event: wx.KeyEvent) -> None:
        key = event.GetKeyCode()
        if key == wx.WXK_ESCAPE:
            self._on_reset(None)
        elif key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER, ord('S'), ord('s')):
            if self._confirm_btn.IsEnabled():
                self._on_confirm(None)
        else:
            event.Skip()

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _vertex_near_canvas(self, cx: int, cy: int) -> int | None:
        best_idx  = None
        best_dist = float(_HOVER_THRESHOLD)
        for i, (nx, ny) in enumerate(self._vertices):
            vx, vy = self._canvas.image_to_canvas_coords(nx, ny)
            d = math.hypot(cx - vx, cy - vy)
            if d < best_dist:
                best_dist = d
                best_idx  = i
        return best_idx

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def _refresh_overlay(self) -> None:
        self._canvas.clear_overlays()

        vertices   = list(self._vertices)
        hover_vi   = self._hover_vertex

        def painter(dc: wx.DC, _cw, _ch) -> None:
            if not vertices:
                return

            canvas_pts = [
                self._canvas.image_to_canvas_coords(nx, ny)
                for nx, ny in vertices
            ]

            # Line between the two vertices
            if len(canvas_pts) == 2:
                dc.SetPen(wx.Pen(_LINE_COLOUR, 2))
                dc.DrawLine(canvas_pts[0][0], canvas_pts[0][1],
                            canvas_pts[1][0], canvas_pts[1][1])

            # Endpoint dots
            for i, (vx, vy) in enumerate(canvas_pts):
                colour = _VERTEX_HOVER if i == hover_vi else _VERTEX_COLOUR
                dc.SetBrush(wx.Brush(colour))
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
                dc.DrawCircle(vx, vy, _VERTEX_RADIUS)

        self._canvas.add_overlay(painter)
        self._canvas.Refresh()

    # ------------------------------------------------------------------
    # Side-panel state
    # ------------------------------------------------------------------

    def _on_reset(self, _event) -> None:
        self._vertices.clear()
        self._hover_vertex    = None
        self._dragging_vertex = None
        self._refresh_overlay()
        self._update_side_panel()
        self._canvas.SetFocus()

    def _on_params_changed(self, _event=None) -> None:
        self._update_side_panel()

    def _update_side_panel(self) -> None:
        n = len(self._vertices)
        self._vertex_label.SetLabel(f"Points: {n} / 2")

        line_px = self._line_px_length()
        if line_px is None:
            self._line_px_label.SetLabel("Line: — px")
            self._scale_label.SetLabel("Scale: —")
            self._confirm_btn.Disable()
            return

        self._line_px_label.SetLabel(f"Line: {line_px:.0f} px")

        seg_count = self._seg_spin.GetValue()
        try:
            seg_cm = float(self._seg_cm_ctrl.GetValue().replace(",", "."))
        except ValueError:
            self._scale_label.SetLabel("Scale: invalid input")
            self._confirm_btn.Disable()
            return

        if seg_cm <= 0 or seg_count <= 0:
            self._scale_label.SetLabel("Scale: —")
            self._confirm_btn.Disable()
            return

        px_per_cm = line_px / (seg_cm * seg_count)
        self._scale_label.SetLabel(f"Scale: {px_per_cm:.1f} px/cm")
        self._confirm_btn.Enable()

    def _line_px_length(self) -> float | None:
        if len(self._vertices) < 2 or self._image is None:
            return None
        img_h, img_w = self._image.shape[:2]
        (nx1, ny1), (nx2, ny2) = self._vertices
        dx = (nx2 - nx1) * img_w
        dy = (ny2 - ny1) * img_h
        return math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Confirm scale → advance to area page
    # ------------------------------------------------------------------

    def _on_confirm(self, _event) -> None:
        line_px = self._line_px_length()
        if line_px is None:
            return
        seg_count = self._seg_spin.GetValue()
        try:
            seg_cm = float(self._seg_cm_ctrl.GetValue().replace(",", "."))
        except ValueError:
            return

        self._confirmed_px_per_cm     = line_px / (seg_cm * seg_count)
        self._confirmed_segment_px    = line_px / seg_count
        self._confirmed_segment_count = seg_count
        self._confirmed_seg_cm        = seg_cm

        px_per_cm = self._confirmed_px_per_cm
        self._area_scale_summary.SetLabel(
            f"Confirmed scale: {px_per_cm:.1f} px/cm  ({seg_cm:.2f} cm/segment)"
        )
        self._update_area_px_label()
        self._pages.SetSelection(1)

    # ------------------------------------------------------------------
    # Area page
    # ------------------------------------------------------------------

    def _on_area_changed(self, _event=None) -> None:
        self._update_area_px_label()

    def _update_area_px_label(self) -> None:
        px_per_cm = self._confirmed_px_per_cm
        if px_per_cm is None:
            self._area_px_label.SetLabel("→ — px × — px")
            return
        try:
            w_cm = float(self._area_w_ctrl.GetValue().replace(",", "."))
            h_cm = float(self._area_h_ctrl.GetValue().replace(",", "."))
        except ValueError:
            self._area_px_label.SetLabel("→ invalid input")
            return
        w_px = int(round(w_cm * px_per_cm))
        h_px = int(round(h_cm * px_per_cm))
        self._area_px_label.SetLabel(f"→ {w_px} px × {h_px} px")

    def _on_save(self, _event) -> None:
        px_per_cm = self._confirmed_px_per_cm
        if px_per_cm is None:
            wx.MessageBox(
                "No scale confirmed. Please place two points and confirm first.",
                "Missing calibration",
                wx.OK | wx.ICON_WARNING,
            )
            return
        try:
            w_cm = float(self._area_w_ctrl.GetValue().replace(",", "."))
            h_cm = float(self._area_h_ctrl.GetValue().replace(",", "."))
        except ValueError:
            wx.MessageBox("Invalid area dimensions.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if w_cm <= 0 or h_cm <= 0:
            wx.MessageBox("Area dimensions must be positive.", "Error", wx.OK | wx.ICON_ERROR)
            return

        cal = ScaleCalibration(
            px_per_cm       = px_per_cm,
            segment_px      = self._confirmed_segment_px  or 0.0,
            segment_cm      = self._confirmed_seg_cm      or _DEFAULT_SEG_CM,
            segment_count   = self._confirmed_segment_count or 0,
            default_area_cm = (w_cm, h_cm),
            source_image    = self._image_name,
        )
        cal.save(self._state.internal_dir() / "calibration.json")
        self._state.calibration = cal
        self.EndModal(wx.ID_OK)

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        bg = wx.Colour(45, 45, 45)
        self.SetBackgroundColour(bg)
        for panel in (self._page_measure, self._page_area):
            panel.SetBackgroundColour(bg)
