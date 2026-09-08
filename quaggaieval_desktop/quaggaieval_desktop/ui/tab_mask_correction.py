"""
Tab 3 — Manual Mask Correction

Allows the user to refine the segmentation mask from the previous step by
drawing polygons that add to or subtract from the existing mask.

Interaction (drawing mode — polygon not yet closed):
  Left-click              Add vertex
  Click near first vertex Close polygon (snap threshold)
  Dbl-click               Close polygon
  Z / Clear button        Reset current polygon

Interaction (editing mode — polygon closed):
  Left-drag on vertex     Reposition vertex
  Shift+Left-click edge   Insert new vertex on that edge
  Ctrl+Left-click vertex  Delete vertex (minimum 3 enforced)

  [+ Add to mask]         OR-merge polygon area into mask
  [− Remove from mask]    AND-NOT polygon area from mask

Both modes:
  Middle-drag             Pan  (ImageCanvas built-in)
  Scroll wheel            Zoom (ImageCanvas built-in)

After each apply the polygon is cleared and the updated mask is displayed.
The user may repeat draw→apply as many times as needed, then confirm.

On confirm, the corrected mask overwrites .internal/{stem}_segmentation_mask.png.
"""
from __future__ import annotations
import math

import cv2
import numpy as np
import wx

from .canvas import ImageCanvas
from ..core.state import AppState, StepStatus

# ── Visual constants — polygon ────────────────────────────────────────────────
FILL_COLOUR_RGBA  = (0, 200, 80, 60)      # green fill, semi-transparent
EDGE_COLOUR       = wx.Colour(0, 220, 80)
VERTEX_COLOUR     = wx.Colour(0, 220, 80)
VERTEX_HOVER      = wx.Colour(0, 255, 255)
VERTEX_FIRST      = wx.Colour(255, 80, 255)  # magenta — close-target hint
PREVIEW_COLOUR    = wx.Colour(180, 180, 180)

VERTEX_RADIUS   = 6
CLOSE_THRESHOLD = 14   # canvas-px distance to snap-close on first vertex
HOVER_THRESHOLD = 12   # canvas-px distance for vertex hover/drag
EDGE_THRESHOLD  = 10   # canvas-px distance for edge insertion

# ── Visual constants — mask overlay ──────────────────────────────────────────
MASK_OVERLAY_ALPHA = 0.35
# Blue in RGBA (used when drawing the wx overlay bitmap)
MASK_OVERLAY_RGBA  = [0, 0, 255, int(MASK_OVERLAY_ALPHA * 255)]


class MaskCorrectionPanel(wx.Panel):

    _SASH_DEFAULT = 240
    _SASH_MIN     = 160

    @staticmethod
    def _make_grid(parent, rows):
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

        # Working copy of the mask being edited (H, W) uint8 0/255
        self._working_mask: np.ndarray | None = None
        # Undo stack — each entry is a full mask snapshot before an apply
        self._mask_history: list[np.ndarray] = []

        # Current polygon — normalised (nx, ny) tuples
        self._vertices: list[tuple[float, float]] = []
        self._closed = False

        # Editing state
        self._hover_vertex:    int | None = None
        self._dragging_vertex: int | None = None
        self._cursor_canvas:   tuple[int, int] | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_complete(self, callback) -> None:
        self._on_complete_callback = callback

    def load_image(self) -> None:
        """Called by MainFrame when this tab becomes active."""
        self._reset_polygon_state()
        self._mask_history.clear()
        self._working_mask = None

        self._canvas.set_image(self._state.current_image)
        self._try_load_mask()
        self._refresh_overlay()
        self._update_controls()
        self._update_status()
        self._canvas.SetFocus()

    def reset(self) -> None:
        self._reset_polygon_state()
        self._mask_history.clear()
        self._working_mask = None
        self._canvas.clear()
        self._update_controls()
        self._update_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
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

        # ── Interaction hints ─────────────────────────────────────────
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
            ("Z",   "clear polygon"),
            ("ESC", "clear polygon"),
        ]

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

        # ── Polygon actions ───────────────────────────────────────────
        poly_box   = wx.StaticBox(ctrl_panel, label="Polygon")
        poly_sizer = wx.StaticBoxSizer(poly_box, wx.VERTICAL)

        self._vertex_label = wx.StaticText(ctrl_panel, label="Vertices: 0")
        self._vertex_label.SetForegroundColour(wx.Colour(160, 160, 160))
        poly_sizer.Add(self._vertex_label, flag=wx.LEFT | wx.TOP | wx.BOTTOM, border=6)

        self._clear_poly_btn = wx.Button(ctrl_panel, label="Clear polygon  [Z/ESC]")
        self._clear_poly_btn.Bind(wx.EVT_BUTTON, self._on_clear_polygon)
        poly_sizer.Add(self._clear_poly_btn, flag=wx.EXPAND | wx.ALL, border=3)

        ctrl.Add(poly_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        # ── Apply actions (enabled only when polygon is closed) ───────
        apply_box   = wx.StaticBox(ctrl_panel, label="Apply selection")
        apply_sizer = wx.StaticBoxSizer(apply_box, wx.VERTICAL)

        self._add_btn = wx.Button(ctrl_panel, label="+ Add to mask")
        self._add_btn.Bind(wx.EVT_BUTTON, self._on_add)
        self._add_btn.Disable()
        apply_sizer.Add(self._add_btn, flag=wx.EXPAND | wx.ALL, border=3)

        self._remove_btn = wx.Button(ctrl_panel, label="− Remove from mask")
        self._remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self._remove_btn.Disable()
        apply_sizer.Add(self._remove_btn, flag=wx.EXPAND | wx.ALL, border=3)

        ctrl.Add(apply_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        # ── Undo edit ─────────────────────────────────────────────────
        self._undo_edit_btn = wx.Button(ctrl_panel, label="Undo last edit")
        self._undo_edit_btn.Bind(wx.EVT_BUTTON, self._on_undo_edit)
        self._undo_edit_btn.Disable()
        ctrl.Add(self._undo_edit_btn,
                 flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

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

        # ── Status ────────────────────────────────────────────────────
        ctrl.Add(wx.StaticLine(ctrl_panel), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=8)
        self._status_label = wx.StaticText(ctrl_panel, label="")
        self._status_label.SetForegroundColour(wx.Colour(140, 140, 140))
        ctrl.Add(self._status_label, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # ── Skip + Confirm ────────────────────────────────────────────
        self._skip_btn = wx.Button(ctrl_panel, label="Skip — no edits")
        self._skip_btn.Bind(wx.EVT_BUTTON, self._on_skip)
        ctrl.Add(self._skip_btn,
                 flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=8)

        self._confirm_btn = wx.Button(ctrl_panel, label="Confirm & next step")
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
            coords = self._canvas.canvas_to_image_coords(cx, cy)
            if coords:
                self._vertices[self._dragging_vertex] = coords
                self._refresh_overlay()
            event.Skip()
            return

        prev_hover = self._hover_vertex
        self._hover_vertex = self._vertex_near_canvas(cx, cy)
        if self._hover_vertex != prev_hover or not self._closed:
            self._refresh_overlay()

        event.Skip()

    def _on_double_click(self, event: wx.MouseEvent) -> None:
        if not self._closed and len(self._vertices) >= 3:
            self._close_polygon()
        # intentionally no event.Skip() — suppress canvas zoom-reset

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _on_key(self, event: wx.KeyEvent) -> None:
        key = event.GetKeyCode()
        if key == ord('Z') or key == ord('z') or key == wx.WXK_ESCAPE:
            self._on_clear_polygon(None)
        else:
            event.Skip()

    # ------------------------------------------------------------------
    # Polygon helpers
    # ------------------------------------------------------------------

    def _close_polygon(self) -> None:
        self._closed = True
        self._add_btn.Enable()
        self._remove_btn.Enable()
        self._refresh_overlay()
        self._update_controls()
        self._update_status("Polygon closed — apply or refine.")

    def _reset_polygon_state(self) -> None:
        self._vertices.clear()
        self._closed           = False
        self._hover_vertex     = None
        self._dragging_vertex  = None
        self._cursor_canvas    = None
        self._add_btn.Disable()
        self._remove_btn.Disable()

    # ------------------------------------------------------------------
    # Hit testing (canvas-pixel space) — same logic as EvalAreaPanel
    # ------------------------------------------------------------------

    def _vertex_near_canvas(self, cx: int, cy: int) -> int | None:
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
    # Button handlers
    # ------------------------------------------------------------------

    def _on_clear_polygon(self, _event) -> None:
        self._reset_polygon_state()
        self._refresh_overlay()
        self._update_controls()
        self._update_status("Polygon cleared.")
        self._canvas.SetFocus()

    def _on_add(self, _event) -> None:
        self._apply_polygon(add=True)
        self._canvas.SetFocus()

    def _on_remove(self, _event) -> None:
        self._apply_polygon(add=False)
        self._canvas.SetFocus()

    def _apply_polygon(self, add: bool) -> None:
        if self._working_mask is None or not self._closed or len(self._vertices) < 3:
            return

        img = self._state.current_image
        if img is None:
            return
        img_h, img_w = img.shape[:2]

        # Push undo snapshot
        self._mask_history.append(self._working_mask.copy())

        # Rasterise polygon
        pts = np.array(
            [(int(nx * img_w), int(ny * img_h)) for nx, ny in self._vertices],
            dtype=np.int32,
        )
        poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)

        if add:
            self._working_mask = cv2.bitwise_or(self._working_mask, poly_mask)
        else:
            self._working_mask = cv2.bitwise_and(
                self._working_mask, cv2.bitwise_not(poly_mask)
            )

        # Clear polygon and refresh
        self._reset_polygon_state()
        self._refresh_overlay()
        self._update_controls()
        action = "added" if add else "removed"
        self._update_status(f"Region {action}.\nDraw another polygon or confirm.")

    def _on_undo_edit(self, _event) -> None:
        if not self._mask_history:
            return
        self._working_mask = self._mask_history.pop()
        self._reset_polygon_state()
        self._refresh_overlay()
        self._update_controls()
        self._update_status("Last edit undone.")
        self._canvas.SetFocus()

    def _on_skip(self, _event) -> None:
        """Pass the original segmentation mask through unchanged."""
        self._save_results()
        record = self._state.current_record
        if record:
            record.mask_correction_status = StepStatus.COMPLETE
        if self._on_complete_callback:
            self._on_complete_callback()

    def _on_confirm(self, _event) -> None:
        self._save_results()
        record = self._state.current_record
        if record:
            record.mask_correction_status = StepStatus.COMPLETE
        if self._on_complete_callback:
            self._on_complete_callback()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        internal_dir = self._state.internal_dir()
        record = self._state.current_record
        if internal_dir is None or record is None or self._working_mask is None:
            return

        mask_path = internal_dir / f"{record.path.stem}_segmentation_mask.png"
        cv2.imwrite(str(mask_path), self._working_mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        self._state.segmentation_mask = self._working_mask

    # ------------------------------------------------------------------
    # Mask loading (from segmentation tab output)
    # ------------------------------------------------------------------

    def _try_load_mask(self) -> None:
        """Load the segmentation mask produced by the previous tab."""
        record = self._state.current_record
        if record is None:
            return

        # Prefer the in-memory mask if freshly produced in this session
        if self._state.segmentation_mask is not None:
            self._working_mask = self._state.segmentation_mask.copy()
            self._confirm_btn.Enable()
            self._skip_btn.Enable()
            self._update_status("Mask loaded — draw polygons to correct.")
            return

        # Fall back to reading from disk
        internal_dir = self._state.internal_dir()
        if internal_dir is None:
            self._update_status("No segmentation mask found.")
            return

        mask_path = internal_dir / f"{record.path.stem}_segmentation_mask.png"
        if not mask_path.exists():
            self._update_status("No segmentation mask found.")
            return

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self._update_status("Failed to load mask.")
            return

        # Normalise to 0/255
        self._working_mask = (mask > 0).astype(np.uint8) * 255
        self._confirm_btn.Enable()
        self._skip_btn.Enable()
        self._update_status("Mask loaded — draw polygons to correct.")

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

        # Capture immutable snapshots for the closures
        working_mask = self._working_mask
        verts        = list(self._vertices)
        closed       = self._closed
        hover_vi     = self._hover_vertex
        cursor_pos   = self._cursor_canvas

        image = self._state.current_image
        if image is None:
            return
        img_h, img_w = image.shape[:2]

        def painter(dc: wx.DC, _cw, _ch) -> None:
            # ── Mask overlay (blue tint) ──────────────────────────────
            if working_mask is not None:
                scaled_w = int(img_w * self._canvas._zoom)
                scaled_h = int(img_h * self._canvas._zoom)
                if scaled_w <= 0 or scaled_h <= 0:
                    return
                ox = int(self._canvas._offset[0])
                oy = int(self._canvas._offset[1])

                mask_scaled = cv2.resize(
                    working_mask, (scaled_w, scaled_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                blue_layer = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
                blue_layer[mask_scaled > 0] = [0, 0, 255]   # RGB blue
                alpha_layer = np.zeros((scaled_h, scaled_w), dtype=np.uint8)
                alpha_layer[mask_scaled > 0] = int(self._overlay_alpha * 255)

                overlay_bmp = wx.Bitmap.FromBufferRGBA(
                    scaled_w, scaled_h,
                    np.dstack([blue_layer, alpha_layer]).astype(np.uint8).tobytes()
                )
                dc.DrawBitmap(overlay_bmp, ox, oy, useMask=True)

            # ── Polygon ───────────────────────────────────────────────
            if not verts:
                return

            canvas_pts = [
                self._canvas.image_to_canvas_coords(nx, ny)
                for nx, ny in verts
            ]

            # Filled polygon overlay (closed only)
            if closed and len(canvas_pts) >= 3:
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

            # Edges
            dc.SetPen(wx.Pen(EDGE_COLOUR, 2))
            n = len(canvas_pts)
            num_edges = n if closed else n - 1
            for i in range(num_edges):
                ax, ay = canvas_pts[i]
                bx, by = canvas_pts[(i + 1) % n]
                dc.DrawLine(ax, ay, bx, by)

            # Preview line to cursor (drawing mode)
            if not closed and cursor_pos and canvas_pts:
                dc.SetPen(wx.Pen(PREVIEW_COLOUR, 1, wx.PENSTYLE_DOT))
                lx, ly = canvas_pts[-1]
                dc.DrawLine(lx, ly, *cursor_pos)

            # Vertices
            for i, (cx, cy) in enumerate(canvas_pts):
                if i == hover_vi:
                    colour = VERTEX_HOVER
                elif i == 0 and not closed and len(canvas_pts) >= 3:
                    colour = VERTEX_FIRST
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
    # Control / label state
    # ------------------------------------------------------------------

    def _update_controls(self) -> None:
        has_verts = len(self._vertices) > 0
        self._clear_poly_btn.Enable(has_verts)
        self._undo_edit_btn.Enable(len(self._mask_history) > 0)
        # Apply buttons are toggled by _close_polygon / _reset_polygon_state

    def _update_status(self, text: str = "") -> None:
        n = len(self._vertices)
        state_str = "closed" if self._closed else "open"
        label = f"Vertices: {n}  ({state_str})"
        if self._mask_history:
            label += f"  |  {len(self._mask_history)} edit(s)"
        self._vertex_label.SetLabel(label)
        self._status_label.SetLabel(text)
        self._status_label.Wrap(190)
