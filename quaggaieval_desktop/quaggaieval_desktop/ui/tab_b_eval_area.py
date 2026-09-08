"""
Tab 1 — Evaluation Area Definition

The user draws a polygon on the image to define the area within which
object area is evaluated. Skipping produces a full-white (unrestricted) mask.

Interaction (drawing mode — polygon not yet closed):
  Left-click              Add vertex
  Click near first vertex Close polygon (snap threshold)
  Z / Undo button         Remove last vertex
  ESC / Clear button      Reset polygon entirely

Interaction (editing mode — polygon closed):
  Left-drag on vertex     Reposition vertex
  Shift+Left-click edge   Insert new vertex on that edge
  Ctrl+Left-click vertex  Delete vertex (minimum 3 enforced)

Both modes:
  Middle-drag             Pan  (ImageCanvas built-in)
  Scroll wheel            Zoom (ImageCanvas built-in)
  Enter / S               Confirm & advance

Polygon vertices are stored as normalised (0..1) image coordinates and
converted to pixel coordinates only when creating/saving the mask.
"""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import wx

from .canvas import ImageCanvas
from ..core.state import AppState, StepStatus

# ── Visual constants ──────────────────────────────────────────────────────────
FILL_COLOUR_RGBA  = (0, 200, 80, 60)     # green fill, semi-transparent
EDGE_COLOUR       = wx.Colour(0, 220, 80)
VERTEX_COLOUR     = wx.Colour(0, 220, 80)
VERTEX_HOVER      = wx.Colour(0, 255, 255)
VERTEX_FIRST      = wx.Colour(255, 80, 255)   # magenta — close-target hint
PREVIEW_COLOUR    = wx.Colour(180, 180, 180)  # ghost line to cursor

VERTEX_RADIUS     = 6
CLOSE_THRESHOLD   = 14   # canvas-px distance to snap-close on first vertex
HOVER_THRESHOLD   = 12   # canvas-px distance to detect vertex hover/drag
EDGE_THRESHOLD    = 10   # canvas-px distance to detect edge for insertion


def rasterise_eval_polygon(
    vertices_norm: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    skipped: bool = False,
) -> np.ndarray:
    """
    Convert normalised polygon vertices → binary uint8 mask (0/255).

    If *skipped* is True or fewer than 3 vertices are provided, returns a
    full-white (unrestricted) mask covering the entire image.
    """
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


class EvalAreaPanel(wx.Panel):

    # Splitter constants (shared across all tabs)
    _SASH_DEFAULT = 240   # right-panel default width (px)
    _SASH_MIN     = 160   # minimum right-panel width  (px)

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

        self._overlay_alpha: float = 0.4

        # Polygon: normalised (nx, ny) tuples
        self._vertices: list[tuple[float, float]] = []
        self._closed   = False

        # Editing state
        self._hover_vertex:   int | None = None   # index of vertex under cursor
        self._dragging_vertex: int | None = None  # index being dragged
        self._cursor_canvas: tuple[int, int] | None = None  # last known canvas pos

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_complete(self, callback) -> None:
        self._on_complete_callback = callback

    def load_image(self) -> None:
        """Called by MainFrame when a new image is selected."""
        self._reset_polygon_state()
        self._canvas.set_image(self._state.current_image)
        self._try_restore_polygon()
        if not self._closed and self._state.calibration is not None:
            self._apply_default_square()
        self._refresh_overlay()
        self._update_controls()
        self._update_status()
        self._canvas.SetFocus()

    def reset(self) -> None:
        self._reset_polygon_state()
        self._canvas.clear()
        self._update_controls()
        self._update_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Splitter ──────────────────────────────────────────────────
        self._splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self._splitter.SetMinimumPaneSize(self._SASH_MIN)

        self._canvas = ImageCanvas(self._splitter)
        self._canvas.Bind(wx.EVT_LEFT_DOWN,   self._on_left_down)
        self._canvas.Bind(wx.EVT_LEFT_UP,     self._on_left_up)
        self._canvas.Bind(wx.EVT_MOTION,      self._on_motion)
        self._canvas.Bind(wx.EVT_KEY_DOWN,    self._on_key)
        self._canvas.Bind(wx.EVT_LEFT_DCLICK, self._on_double_click)

        ctrl_panel = wx.Panel(self._splitter)
        ctrl = wx.BoxSizer(wx.VERTICAL)

        # ── Interaction box (2-column grid, split by mode) ────────────
        instr_box   = wx.StaticBox(ctrl_panel, label="Interaction")
        instr_sizer = wx.StaticBoxSizer(instr_box, wx.VERTICAL)

        draw_rows = [
            ("Left-click",       "add vertex"),
            ("Click 1st vertex", "close polygon"),
            ("Dbl-click",        "close polygon"),
        ]
        edit_rows = [
            ("Drag vertex",       "move"),
            ("Shift+click edge",  "insert vertex"),
            ("Ctrl+click vertex", "delete vertex"),
        ]
        nav_rows = [
            ("Middle-drag", "pan"),
            ("Scroll",      "zoom"),
        ]
        key_rows = [
            ("Z",       "undo last vertex"),
            ("ESC",     "reset polygon"),
            ("Enter/S", "confirm"),
        ]

        dim = wx.Colour(160, 160, 160)

        def section(label):
            lbl = wx.StaticText(ctrl_panel, label=label)
            lbl.SetForegroundColour(wx.Colour(120, 120, 120))
            return lbl

        instr_sizer.Add(self._make_grid(ctrl_panel, draw_rows),
                        flag=wx.EXPAND | wx.ALL, border=6)
        instr_sizer.Add(section("— when closed —"),
                        flag=wx.LEFT | wx.BOTTOM, border=6)
        instr_sizer.Add(self._make_grid(ctrl_panel, edit_rows),
                        flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=6)
        instr_sizer.Add(wx.StaticLine(ctrl_panel),
                        flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=4)
        instr_sizer.Add(self._make_grid(ctrl_panel, nav_rows + key_rows),
                        flag=wx.EXPAND | wx.ALL, border=6)
        ctrl.Add(instr_sizer, flag=wx.EXPAND | wx.ALL, border=8)

        # ── Polygon actions box ───────────────────────────────────────
        poly_box   = wx.StaticBox(ctrl_panel, label="Polygon")
        poly_sizer = wx.StaticBoxSizer(poly_box, wx.VERTICAL)

        self._vertex_label = wx.StaticText(ctrl_panel, label="Vertices: 0")
        self._vertex_label.SetForegroundColour(dim)
        poly_sizer.Add(self._vertex_label, flag=wx.LEFT | wx.TOP | wx.BOTTOM, border=6)

        self._undo_btn = wx.Button(ctrl_panel, label="Undo last vertex  [Z]")
        self._undo_btn.Bind(wx.EVT_BUTTON, self._on_undo)
        poly_sizer.Add(self._undo_btn, flag=wx.EXPAND | wx.ALL, border=3)

        self._clear_btn = wx.Button(ctrl_panel, label="Clear polygon  [ESC]")
        self._clear_btn.Bind(wx.EVT_BUTTON, self._on_clear)
        poly_sizer.Add(self._clear_btn, flag=wx.EXPAND | wx.ALL, border=3)

        ctrl.Add(poly_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

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
        ov_row.Add(ov_lbl,               proportion=0, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        ov_row.Add(self._opacity_slider, proportion=1, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=4)
        ov_row.Add(self._opacity_label,  proportion=0, flag=wx.ALIGN_CENTER_VERTICAL)
        ov_sizer.Add(ov_row, flag=wx.EXPAND | wx.ALL, border=6)

        self._opacity_slider.Bind(wx.EVT_SLIDER, self._on_opacity_slider)
        ctrl.Add(ov_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        ctrl.AddStretchSpacer()

        # ── Status + separator ────────────────────────────────────────
        ctrl.Add(wx.StaticLine(ctrl_panel), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=8)
        self._status_label = wx.StaticText(ctrl_panel, label="")
        self._status_label.SetForegroundColour(wx.Colour(140, 140, 140))
        ctrl.Add(self._status_label, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # ── Skip + Confirm ────────────────────────────────────────────
        skip_btn = wx.Button(ctrl_panel, label="Skip — use full image")
        skip_btn.Bind(wx.EVT_BUTTON, self._on_skip)
        ctrl.Add(skip_btn, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=8)

        self._confirm_btn = wx.Button(ctrl_panel, label="Confirm & next step  [S]")
        self._confirm_btn.Bind(wx.EVT_BUTTON, self._on_confirm)
        self._confirm_btn.Disable()
        ctrl.Add(self._confirm_btn, flag=wx.EXPAND | wx.ALL, border=8)

        ctrl_panel.SetSizer(ctrl)

        self._splitter.SplitVertically(self._canvas, ctrl_panel)
        self._splitter.SetSashGravity(1.0)

        def _set_sash(_evt=None):
            w = self._splitter.GetClientSize().width
            self._splitter.SetSashPosition(w - self._SASH_DEFAULT)
        self._splitter.Bind(wx.EVT_SIZE, _set_sash)

        root = wx.BoxSizer(wx.VERTICAL)
        root.Add(self._splitter, proportion=1, flag=wx.EXPAND)
        self.SetSizer(root)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _on_left_down(self, event: wx.MouseEvent) -> None:
        cx, cy = event.GetPosition()

        # ── Editing mode (closed polygon) ─────────────────────────────
        if self._closed:
            vi = self._vertex_near_canvas(cx, cy)

            if vi is not None:
                if event.ControlDown():
                    # Ctrl+click — delete vertex (min 3)
                    if len(self._vertices) > 3:
                        del self._vertices[vi]
                        self._hover_vertex = None
                        self._refresh_overlay()
                        self._update_status()
                    else:
                        self._update_status("Need ≥ 3 vertices.")
                else:
                    # Start dragging
                    self._dragging_vertex = vi
            elif event.ShiftDown():
                # Shift+click — insert on nearest edge
                ei = self._edge_near_canvas(cx, cy)
                if ei is not None:
                    coords = self._canvas.canvas_to_image_coords(cx, cy)
                    if coords:
                        self._vertices.insert(ei + 1, coords)
                        self._refresh_overlay()
                        self._update_status()
            event.Skip()
            return

        # ── Drawing mode (polygon not yet closed) ─────────────────────
        coords = self._canvas.canvas_to_image_coords(cx, cy)
        if coords is None:
            event.Skip()
            return

        # Snap-close if clicking near first vertex
        if len(self._vertices) >= 3:
            fx, fy = self._canvas.image_to_canvas_coords(*self._vertices[0])
            if math.hypot(cx - fx, cy - fy) <= CLOSE_THRESHOLD:
                self._close_polygon()
                return

        self._vertices.append(coords)
        self._refresh_overlay()
        self._update_controls()
        self._update_status()
        event.Skip()

    def _on_left_up(self, event: wx.MouseEvent) -> None:
        self._dragging_vertex = None
        event.Skip()

    def _on_motion(self, event: wx.MouseEvent) -> None:
        cx, cy = event.GetPosition()
        self._cursor_canvas = (cx, cy)

        if self._dragging_vertex is not None and event.LeftIsDown():
            # Drag vertex to new position
            coords = self._canvas.canvas_to_image_coords(cx, cy)
            if coords:
                self._vertices[self._dragging_vertex] = coords
                self._refresh_overlay()
            event.Skip()
            return

        # Update hover highlight
        prev_hover = self._hover_vertex
        self._hover_vertex = self._vertex_near_canvas(cx, cy)
        if self._hover_vertex != prev_hover or not self._closed:
            self._refresh_overlay()

        event.Skip()

    def _on_double_click(self, event: wx.MouseEvent) -> None:
        """Double-click closes an open polygon (suppresses ImageCanvas zoom reset)."""
        if not self._closed and len(self._vertices) >= 3:
            self._close_polygon()
        # intentionally no event.Skip() — suppress canvas zoom-reset

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _on_key(self, event: wx.KeyEvent) -> None:
        key = event.GetKeyCode()
        if key == ord('Z') or key == ord('z'):
            self._on_undo(None)
        elif key == wx.WXK_ESCAPE:
            self._on_clear(None)
        elif key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER, ord('S'), ord('s')):
            if self._confirm_btn.IsEnabled():
                self._on_confirm(None)
        else:
            event.Skip()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_undo(self, _event) -> None:
        if self._closed:
            # Reopen polygon for further editing
            self._closed = False
            self._confirm_btn.Disable()
            self._update_status("Polygon reopened for editing.")
        elif self._vertices:
            self._vertices.pop()
        self._refresh_overlay()
        self._update_controls()
        self._update_status()
        self._canvas.SetFocus()

    def _on_clear(self, _event) -> None:
        self._reset_polygon_state()
        self._refresh_overlay()
        self._update_controls()
        self._update_status("Polygon cleared.")
        self._canvas.SetFocus()

    def _on_skip(self, _event) -> None:
        """Generate a full-white (unrestricted) mask without drawing."""
        self._reset_polygon_state()
        self._refresh_overlay()
        self._update_status("Full-image mask will be used.")
        self._confirm_btn.Enable()
        self._canvas.SetFocus()

    def _on_confirm(self, _event) -> None:
        self._save_results()
        if self._state.current_record:
            self._state.current_record.eval_area_status = StepStatus.COMPLETE
        if self._on_complete_callback:
            self._on_complete_callback()

    # ------------------------------------------------------------------
    # Polygon helpers
    # ------------------------------------------------------------------

    def _close_polygon(self) -> None:
        self._closed = True
        self._confirm_btn.Enable()
        self._refresh_overlay()
        self._update_controls()
        self._update_status("Polygon closed ✓  —  drag vertices to refine.")

    def _reset_polygon_state(self) -> None:
        self._vertices.clear()
        self._closed            = False
        self._hover_vertex      = None
        self._dragging_vertex   = None
        self._cursor_canvas     = None
        self._confirm_btn.Disable()

    def _apply_default_square(self) -> None:
        """Pre-populate a centered square from the folder's scale calibration."""
        cal = self._state.calibration
        img = self._state.current_image
        if cal is None or img is None:
            return
        img_h, img_w = img.shape[:2]
        w_px = min(cal.cm_to_px(cal.default_area_cm[0]), img_w * 0.95)
        h_px = min(cal.cm_to_px(cal.default_area_cm[1]), img_h * 0.95)
        cx, cy = img_w / 2.0, img_h / 2.0
        self._vertices = [
            ((cx - w_px / 2) / img_w, (cy - h_px / 2) / img_h),
            ((cx + w_px / 2) / img_w, (cy - h_px / 2) / img_h),
            ((cx + w_px / 2) / img_w, (cy + h_px / 2) / img_h),
            ((cx - w_px / 2) / img_w, (cy + h_px / 2) / img_h),
        ]
        self._closed = True
        self._confirm_btn.Enable()
        self._update_status("Default area applied — drag vertices to adjust.")

    # ------------------------------------------------------------------
    # Hit testing (canvas-pixel space)
    # ------------------------------------------------------------------

    def _vertex_near_canvas(self, cx: int, cy: int) -> int | None:
        """Return index of the nearest vertex within HOVER_THRESHOLD, or None."""
        best_idx  = None
        best_dist = HOVER_THRESHOLD
        for i, (nx, ny) in enumerate(self._vertices):
            vx, vy = self._canvas.image_to_canvas_coords(nx, ny)
            d = math.hypot(cx - vx, cy - vy)
            if d < best_dist:
                best_dist = d
                best_idx  = i
        return best_idx

    def _edge_near_canvas(self, cx: int, cy: int) -> int | None:
        """
        Return index i of the edge (v[i], v[i+1 % n]) closest to (cx, cy)
        within EDGE_THRESHOLD canvas pixels, or None.
        """
        n = len(self._vertices)
        num_edges = n if self._closed else n - 1
        best_idx  = None
        best_dist = EDGE_THRESHOLD

        for i in range(num_edges):
            ax, ay = self._canvas.image_to_canvas_coords(*self._vertices[i])
            bx, by = self._canvas.image_to_canvas_coords(*self._vertices[(i + 1) % n])

            p  = np.array([cx,  cy],  dtype=np.float32)
            a  = np.array([ax,  ay],  dtype=np.float32)
            b  = np.array([bx,  by],  dtype=np.float32)
            ab = b - a
            ab_len = np.linalg.norm(ab)
            if ab_len == 0:
                continue
            t = np.clip(np.dot(p - a, ab) / (ab_len ** 2), 0, 1)
            closest = a + t * ab
            dist = np.linalg.norm(p - closest)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        return best_idx

    # ------------------------------------------------------------------
    # Overlay opacity
    # ------------------------------------------------------------------

    def _on_opacity_slider(self, _evt) -> None:
        self._overlay_alpha = self._opacity_slider.GetValue() / 10.0
        self._opacity_label.SetLabel(f"{self._overlay_alpha:.1f}")
        self._refresh_overlay()

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _refresh_overlay(self) -> None:
        self._canvas.clear_overlays()

        # Capture mutable state for the closure
        verts      = list(self._vertices)
        closed     = self._closed
        hover_vi   = self._hover_vertex
        cursor_pos = self._cursor_canvas   # canvas (cx, cy) or None

        def painter(dc: wx.DC, _cw, _ch) -> None:
            if not verts:
                return

            canvas_pts = [
                self._canvas.image_to_canvas_coords(nx, ny)
                for nx, ny in verts
            ]

            # ── Filled polygon overlay (closed only) ──────────────────
            if closed and len(canvas_pts) >= 3:
                pts = [wx.Point(x, y) for x, y in canvas_pts]
                gc = wx.GraphicsContext.Create(dc)
                if gc:
                    gc.SetBrush(wx.Brush(wx.Colour(
                        FILL_COLOUR_RGBA[0], FILL_COLOUR_RGBA[1], FILL_COLOUR_RGBA[2],
                        int(self._overlay_alpha * 255),
                    )))
                    gc.SetPen(wx.NullPen)
                    path = gc.CreatePath()
                    path.MoveToPoint(*canvas_pts[0])
                    for px, py in canvas_pts[1:]:
                        path.AddLineToPoint(px, py)
                    path.CloseSubpath()
                    gc.DrawPath(path)

            # ── Edges ─────────────────────────────────────────────────
            dc.SetPen(wx.Pen(EDGE_COLOUR, 2))
            n = len(canvas_pts)
            num_edges = n if closed else n - 1
            for i in range(num_edges):
                ax, ay = canvas_pts[i]
                bx, by = canvas_pts[(i + 1) % n]
                dc.DrawLine(ax, ay, bx, by)

            # ── Preview line to cursor (drawing mode) ─────────────────
            if not closed and cursor_pos and canvas_pts:
                dc.SetPen(wx.Pen(PREVIEW_COLOUR, 1, wx.PENSTYLE_DOT))
                lx, ly = canvas_pts[-1]
                dc.DrawLine(lx, ly, *cursor_pos)

            # ── Vertices ──────────────────────────────────────────────
            for i, (cx, cy) in enumerate(canvas_pts):
                # Choose colour
                if i == hover_vi:
                    colour = VERTEX_HOVER
                elif i == 0 and not closed and len(canvas_pts) >= 3:
                    colour = VERTEX_FIRST   # snap-close target hint
                else:
                    colour = VERTEX_COLOUR

                dc.SetBrush(wx.Brush(colour))
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
                dc.DrawCircle(cx, cy, VERTEX_RADIUS)

                # Extra ring on first vertex when closeable
                if i == 0 and not closed and len(canvas_pts) >= 3:
                    dc.SetBrush(wx.TRANSPARENT_BRUSH)
                    dc.SetPen(wx.Pen(VERTEX_FIRST, 2))
                    dc.DrawCircle(cx, cy, VERTEX_RADIUS + 4)

        self._canvas.add_overlay(painter)
        self._canvas.Refresh()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        internal_dir = self._state.internal_dir()
        record = self._state.current_record
        if internal_dir is None or record is None:
            return

        image = self._state.current_image
        img_h, img_w = image.shape[:2]
        skipped = not (self._closed and len(self._vertices) >= 3)

        # ── Build normalised vertex list ──────────────────────────────
        if skipped:
            norm_verts: list[list[float]] = []
        else:
            norm_verts = [[float(nx), float(ny)] for nx, ny in self._vertices]

        entry = {
            "image_size": [img_w, img_h],
            "vertices":   norm_verts,
            "skipped":    skipped,
            "timestamp":  datetime.now().isoformat() + "Z",
        }

        # ── Read-modify-write eval_areas.json ─────────────────────────
        json_path = internal_dir / "eval_areas.json"
        try:
            with open(json_path) as f:
                all_areas: dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_areas = {}

        all_areas[record.path.name] = entry

        with open(json_path, "w") as f:
            json.dump(all_areas, f, indent=2)

        # ── Build mask and store in AppState for downstream tab ───────
        mask = rasterise_eval_polygon(
            [(v[0], v[1]) for v in norm_verts], img_w, img_h, skipped
        )
        self._state.eval_area_mask = mask

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _try_restore_polygon(self) -> None:
        """Load previously saved polygon from eval_areas.json if an entry exists."""
        internal_dir = self._state.internal_dir()
        record = self._state.current_record
        if internal_dir is None or record is None:
            return

        json_path = internal_dir / "eval_areas.json"
        if not json_path.exists():
            return

        try:
            with open(json_path) as f:
                all_areas: dict = json.load(f)

            entry = all_areas.get(record.path.name)
            if entry is None:
                return

            if entry.get("skipped"):
                self._confirm_btn.Enable()
                self._update_status("Previously skipped — full-image mask loaded.")
                return

            img_w, img_h = entry["image_size"]
            raw_verts = entry.get("vertices", [])

            if len(raw_verts) >= 3:
                # Vertices stored as normalised [nx, ny] pairs
                self._vertices = [(v[0], v[1]) for v in raw_verts]
                self._closed = True
                self._confirm_btn.Enable()
                self._update_status(
                    f"Restored {len(self._vertices)}-vertex polygon.\n"
                    "Drag vertices to edit."
                )
        except Exception as exc:
            self._update_status(f"Could not restore polygon:\n{exc}")

    # ------------------------------------------------------------------
    # Control / label state
    # ------------------------------------------------------------------

    def _update_controls(self) -> None:
        has_verts = len(self._vertices) > 0
        self._undo_btn.Enable(has_verts)
        self._clear_btn.Enable(has_verts)

    def _update_status(self, text: str = "") -> None:
        n = len(self._vertices)
        state_str = "closed" if self._closed else "open"
        self._vertex_label.SetLabel(f"Vertices: {n}  ({state_str})")
        self._status_label.SetLabel(text)
        self._status_label.Wrap(190)