"""
ImageCanvas — a reusable wx.Panel subclass for displaying and interacting
with images. Supports pan, zoom, and emits normalised (0..1) coordinates
so tab panels don't need to know about zoom/pan state.
"""
from __future__ import annotations
import wx
import numpy as np


class ImageCanvas(wx.Panel):
    """
    Displays a numpy (H, W, 3) BGR or RGB image with pan and scroll-to-zoom.
    Subclass or bind to EVT_* on this panel for step-specific interactions.

    Coordinate system
    -----------------
    All public callbacks receive image-space coordinates (float, 0..1 normalised)
    so callers are decoupled from display zoom/offset.
    """

    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, parent: wx.Window, **kwargs):
        super().__init__(parent, style=wx.WANTS_CHARS, **kwargs)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(wx.Colour(40, 40, 40))

        self._bitmap: wx.Bitmap | None = None
        self._img_w: int = 0
        self._img_h: int = 0

        # Pan / zoom state
        self._zoom: float = 1.0
        self._offset: list[float] = [0.0, 0.0]  # [x, y] in canvas pixels
        self._pan_start: tuple[int, int] | None = None
        self._pan_offset_start: list[float] = [0.0, 0.0]

        # Overlay drawables: list of callables (dc, canvas_w, canvas_h)
        self._overlay_painters: list = []

        self._bind_events()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray, bgr: bool = True) -> None:
        """Load a numpy image array (H, W, 3). Call Refresh() automatically."""
        if image is None:
            self._bitmap = None
            self.Refresh()
            return

        if bgr:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        self._img_w, self._img_h = w, h

        # Ensure contiguous memory for wx
        image = np.ascontiguousarray(image, dtype=np.uint8)
        self._bitmap = wx.Bitmap.FromBuffer(w, h, image)
        self._fit_to_canvas()
        self.Refresh()

    def clear(self) -> None:
        self._bitmap = None
        self._overlay_painters.clear()
        self.Refresh()

    def add_overlay(self, painter_fn) -> None:
        """
        Register a painter callback: fn(dc: wx.DC, canvas_w: int, canvas_h: int)
        called after the image is drawn each paint cycle.
        """
        self._overlay_painters.append(painter_fn)

    def clear_overlays(self) -> None:
        self._overlay_painters.clear()
        self.Refresh()

    def canvas_to_image_coords(self, cx: int, cy: int) -> tuple[float, float] | None:
        """
        Convert canvas pixel position to normalised image coords (0..1).
        Returns None if the point is outside the image.
        """
        if self._bitmap is None:
            return None
        ix = (cx - self._offset[0]) / (self._img_w * self._zoom)
        iy = (cy - self._offset[1]) / (self._img_h * self._zoom)
        if 0.0 <= ix <= 1.0 and 0.0 <= iy <= 1.0:
            return ix, iy
        return None

    def image_to_canvas_coords(self, nx: float, ny: float) -> tuple[int, int]:
        """Convert normalised image coords (0..1) back to canvas pixels."""
        cx = int(nx * self._img_w * self._zoom + self._offset[0])
        cy = int(ny * self._img_h * self._zoom + self._offset[1])
        return cx, cy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_to_canvas(self) -> None:
        """Reset zoom/offset so the image fits the current canvas size."""
        cw, ch = self.GetClientSize()
        if cw <= 0 or ch <= 0 or self._img_w == 0:
            return
        self._zoom = min(cw / self._img_w, ch / self._img_h)
        self._offset = [
            (cw - self._img_w * self._zoom) / 2,
            (ch - self._img_h * self._zoom) / 2,
        ]

    def _clamp_offset(self) -> None:
        cw, ch = self.GetClientSize()
        iw = self._img_w * self._zoom
        ih = self._img_h * self._zoom
        # Allow panning up to one canvas-width past the image edge
        margin_x = min(cw * 0.9, iw * 0.9)
        margin_y = min(ch * 0.9, ih * 0.9)
        self._offset[0] = max(margin_x - iw, min(self._offset[0], cw - margin_x))
        self._offset[1] = max(margin_y - ih, min(self._offset[1], ch - margin_y))

    # ------------------------------------------------------------------
    # Event bindings
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_scroll)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._on_pan_start)
        self.Bind(wx.EVT_MIDDLE_UP, self._on_pan_end)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_fit)  # double-click resets view

    def _on_paint(self, _event: wx.PaintEvent) -> None:
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()

        if self._bitmap is None:
            dc.SetTextForeground(wx.Colour(120, 120, 120))
            dc.DrawLabel("No image loaded", self.GetClientRect(), wx.ALIGN_CENTER)
            return

        scaled_w = int(self._img_w * self._zoom)
        scaled_h = int(self._img_h * self._zoom)
        img = self._bitmap.ConvertToImage().Scale(
            scaled_w, scaled_h, wx.IMAGE_QUALITY_BILINEAR
        )
        dc.DrawBitmap(wx.Bitmap(img), int(self._offset[0]), int(self._offset[1]))

        cw, ch = self.GetClientSize()
        for painter in self._overlay_painters:
            painter(dc, cw, ch)

    def _on_size(self, event: wx.SizeEvent) -> None:
        if self._bitmap is not None:
            self._fit_to_canvas()
        self.Refresh()
        event.Skip()

    def _on_scroll(self, event: wx.MouseEvent) -> None:
        if self._bitmap is None:
            return
        mx, my = event.GetPosition()
        factor = self.ZOOM_STEP if event.GetWheelRotation() > 0 else 1 / self.ZOOM_STEP
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))

        # Zoom toward mouse cursor
        self._offset[0] = mx - (mx - self._offset[0]) * (new_zoom / self._zoom)
        self._offset[1] = my - (my - self._offset[1]) * (new_zoom / self._zoom)
        self._zoom = new_zoom
        self._clamp_offset()
        self.Refresh()

    def _on_pan_start(self, event: wx.MouseEvent) -> None:
        self._pan_start = event.GetPosition()
        self._pan_offset_start = list(self._offset)
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))

    def _on_pan_end(self, _event: wx.MouseEvent) -> None:
        self._pan_start = None
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

    def _on_motion(self, event: wx.MouseEvent) -> None:
        if self._pan_start and event.MiddleIsDown():
            dx = event.GetX() - self._pan_start[0]
            dy = event.GetY() - self._pan_start[1]
            self._offset[0] = self._pan_offset_start[0] + dx
            self._offset[1] = self._pan_offset_start[1] + dy
            self._clamp_offset()
            self.Refresh()
        event.Skip()  # allow subclass / bound handlers to also receive motion

    def _on_fit(self, _event: wx.MouseEvent) -> None:
        """Double-click resets zoom to fit."""
        self._fit_to_canvas()
        self.Refresh()