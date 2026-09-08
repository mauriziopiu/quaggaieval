"""
Tab 3 — Coverage

Loads the segmentation mask and evaluation area mask produced by Tabs 1 & 2,
runs coverage computation, displays a colour-coded overlay on the canvas,
and writes results to the output directory.

No threading needed — compute_coverage() is pure numpy and runs in < 1s.
"""
from __future__ import annotations
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import wx

from .canvas import ImageCanvas
from ..core.state import AppState, StepStatus
from .tab_b_eval_area import rasterise_eval_polygon

# ---------------------------------------------------------------------------
# Inline coverage logic (lifted from sam2_eval_coverage.py — no import needed)
# ---------------------------------------------------------------------------

from dataclasses import dataclass

COLOR_COVERED    = (0,   200,   0)
COLOR_UNCOVERED  = (0,     0, 220)
COLOR_OUTSIDE    = (180, 180, 180)
COLOR_BACKGROUND = (30,   30,  30)
ALPHA            = 0.55


@dataclass
class CoverageResult:
    object_mask_file:  str
    area_mask_file:    str
    area_pixels:       int
    covered_pixels:    int
    uncovered_pixels:  int
    outside_pixels:    int
    coverage_pct:      float
    outside_pct:       float
    image_width:       int
    image_height:      int
    area_cm2:          float | None = None
    covered_cm2:       float | None = None
    uncovered_cm2:     float | None = None
    outside_cm2:       float | None = None

    def summary(self) -> str:
        lines = [
            f"Area pixels:      {self.area_pixels:,}",
            f"Covered:          {self.covered_pixels:,}  ({self.coverage_pct:.2f}%)",
            f"Uncovered:        {self.uncovered_pixels:,}  ({100 - self.coverage_pct:.2f}%)",
            f"Outside area:     {self.outside_pixels:,}  ({self.outside_pct:.2f}% of object)",
        ]
        return "\n".join(lines)


def _load_binary(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    return img > 0


def compute_coverage(
    obj_mask: np.ndarray,
    area_mask: np.ndarray,
    obj_name: str,
    area_name: str,
) -> CoverageResult:
    if obj_mask.shape != area_mask.shape:
        raise ValueError(
            f"Mask shapes differ: object={obj_mask.shape}, area={area_mask.shape}"
        )
    h, w = obj_mask.shape

    covered   =  obj_mask &  area_mask
    uncovered = ~obj_mask &  area_mask
    outside   =  obj_mask & ~area_mask

    area_px    = int(area_mask.sum())
    covered_px = int(covered.sum())
    obj_px     = int(obj_mask.sum())

    coverage_pct = (covered_px / area_px  * 100) if area_px  > 0 else 0.0
    outside_pct  = (int(outside.sum()) / obj_px * 100) if obj_px > 0 else 0.0

    return CoverageResult(
        object_mask_file  = obj_name,
        area_mask_file    = area_name,
        area_pixels       = area_px,
        covered_pixels    = covered_px,
        uncovered_pixels  = int(uncovered.sum()),
        outside_pixels    = int(outside.sum()),
        coverage_pct      = round(coverage_pct, 4),
        outside_pct       = round(outside_pct,  4),
        image_width       = w,
        image_height      = h,
    )


def render_visualisation(
    obj_mask: np.ndarray,
    area_mask: np.ndarray,
) -> np.ndarray:
    h, w = obj_mask.shape
    canvas = np.full((h, w, 3), COLOR_BACKGROUND, dtype=np.uint8)
    canvas[ obj_mask & ~area_mask] = COLOR_OUTSIDE
    canvas[~obj_mask &  area_mask] = COLOR_UNCOVERED
    canvas[ obj_mask &  area_mask] = COLOR_COVERED
    return canvas


def render_overlay(
    source_bgr: np.ndarray,
    obj_mask: np.ndarray,
    area_mask: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    img = source_bgr.copy()
    if img.shape[:2] != obj_mask.shape:
        img = cv2.resize(img, (obj_mask.shape[1], obj_mask.shape[0]))
    color_layer = render_visualisation(obj_mask, area_mask)
    return cv2.addWeighted(img, 1 - alpha, color_layer, alpha, 0)


_CSV_FIELDS = [
    "filename",
    "area_px", "covered_px", "uncovered_px", "outside_px",
    "coverage_pct", "outside_pct",
    "area_cm2", "covered_cm2", "uncovered_cm2", "outside_cm2",
]


def _write_coverage_csv_row(csv_path: Path, stem: str, result: "CoverageResult") -> None:
    """
    Upsert the coverage row for *stem* in *csv_path*.

    Reads existing rows keyed by 'filename', replaces the matching row (or
    appends a new one), then rewrites the whole file.
    """
    filename = stem  # key column — matches the image stem

    new_row = {
        "filename":     filename,
        "area_px":      result.area_pixels,
        "covered_px":   result.covered_pixels,
        "uncovered_px": result.uncovered_pixels,
        "outside_px":   result.outside_pixels,
        "coverage_pct": result.coverage_pct,
        "outside_pct":  result.outside_pct,
        "area_cm2":     result.area_cm2     if result.area_cm2     is not None else "",
        "covered_cm2":  result.covered_cm2  if result.covered_cm2  is not None else "",
        "uncovered_cm2":result.uncovered_cm2 if result.uncovered_cm2 is not None else "",
        "outside_cm2":  result.outside_cm2  if result.outside_cm2  is not None else "",
    }

    rows: list[dict] = []
    updated = False
    if csv_path.exists():
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("filename") == filename:
                        rows.append(new_row)
                        updated = True
                    else:
                        rows.append(row)
        except Exception:
            rows = []

    if not updated:
        rows.append(new_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_coverage_csv_row(csv_path: Path, stem: str) -> dict | None:
    """Return the CSV row dict for *stem*, or None if not found."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("filename") == stem:
                    return row
    except Exception:
        pass
    return None


def draw_legend(canvas: np.ndarray, result: CoverageResult) -> np.ndarray:
    canvas = canvas.copy()
    h, _ = canvas.shape[:2]
    entries = [
        (COLOR_COVERED,   f"Covered    {result.coverage_pct:.1f}%"),
        (COLOR_UNCOVERED, f"Uncovered  {100 - result.coverage_pct:.1f}%"),
        (COLOR_OUTSIDE,   f"Outside area  {result.outside_pct:.1f}% of object"),
    ]
    box_sz = 14
    pad    = 8
    line_h = box_sz + 6
    total_h = len(entries) * line_h + pad * 2
    total_w = box_sz + pad * 3 + max(len(e[1]) for e in entries) * 9

    x0, y0 = pad, h - total_h - pad
    roi = canvas[y0:y0 + total_h, x0:x0 + total_w]
    dark = np.full_like(roi, 20)
    canvas[y0:y0 + total_h, x0:x0 + total_w] = cv2.addWeighted(roi, 0.35, dark, 0.65, 0)

    for i, (colour, label) in enumerate(entries):
        by = y0 + pad + i * line_h
        bx = x0 + pad
        cv2.rectangle(canvas, (bx, by), (bx + box_sz, by + box_sz), colour, -1)
        cv2.putText(canvas, label, (bx + box_sz + 6, by + box_sz - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class AreaEvaluationPanel(wx.Panel):

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
        self._state  = state
        self._result: CoverageResult | None = None
        self._on_complete_callback = None

        self._overlay_alpha: float = 0.4
        self._last_obj_mask:    np.ndarray | None = None
        self._last_area_mask:   np.ndarray | None = None
        self._last_source_bgr:  np.ndarray | None = None
        self._last_area_entry:  dict | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_complete(self, callback) -> None:
        self._on_complete_callback = callback

    def load_image(self) -> None:
        """Called by MainFrame when a new image is selected / tab is shown."""
        self._result = None
        self._last_obj_mask   = None
        self._last_area_mask  = None
        self._last_source_bgr = None
        self._last_area_entry = None
        self._canvas.set_image(self._state.current_image)
        self._update_input_labels()
        self._result_label.SetLabel("Running coverage…")
        self._run_btn.Disable()
        self._confirm_btn.Disable()

        # Restore previous result if this image was already evaluated;
        # otherwise auto-run immediately.
        if not self._try_restore_result():
            self._on_run(None)

    def reset(self) -> None:
        self._result = None
        self._last_obj_mask   = None
        self._last_area_mask  = None
        self._last_source_bgr = None
        self._last_area_entry = None
        self._canvas.clear()
        self._result_label.SetLabel("")
        self._confirm_btn.Disable()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Splitter ──────────────────────────────────────────────────
        self._splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self._splitter.SetMinimumPaneSize(self._SASH_MIN)

        self._canvas = ImageCanvas(self._splitter)

        ctrl_panel = wx.Panel(self._splitter)
        self._ctrl_panel = ctrl_panel
        ctrl = wx.BoxSizer(wx.VERTICAL)
        dim = wx.Colour(160, 160, 160)

        # ── Inputs box ────────────────────────────────────────────────
        in_box   = wx.StaticBox(ctrl_panel, label="Inputs")
        in_sizer = wx.StaticBoxSizer(in_box, wx.VERTICAL)
        self._seg_label  = wx.StaticText(ctrl_panel, label="Seg mask:  —")
        self._area_label = wx.StaticText(ctrl_panel, label="Area mask: —")
        for lbl in (self._seg_label, self._area_label):
            lbl.SetForegroundColour(dim)
            in_sizer.Add(lbl, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=6)
        in_sizer.AddSpacer(4)
        ctrl.Add(in_sizer, flag=wx.EXPAND | wx.ALL, border=8)

        # ── Legend box ────────────────────────────────────────────────
        leg_box   = wx.StaticBox(ctrl_panel, label="Legend")
        leg_sizer = wx.StaticBoxSizer(leg_box, wx.VERTICAL)
        legend_entries = [
            (wx.Colour(0, 200, 0),   "Covered  (object ∩ area)"),
            (wx.Colour(220, 0, 0),   "Uncovered  (area ∖ object)"),
            (wx.Colour(180,180,180), "Outside area  (object ∖ area)"),
        ]
        for colour, label in legend_entries:
            row = wx.BoxSizer(wx.HORIZONTAL)
            swatch = wx.Panel(ctrl_panel, size=(12, 12))
            swatch.SetBackgroundColour(colour)
            lbl = wx.StaticText(ctrl_panel, label=label)
            lbl.SetForegroundColour(dim)
            row.Add(swatch, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
            row.Add(lbl,    flag=wx.ALIGN_CENTER_VERTICAL)
            leg_sizer.Add(row, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=6)
        leg_sizer.AddSpacer(4)
        ctrl.Add(leg_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        # ── Results box (3-column grid) ───────────────────────────────
        res_box   = wx.StaticBox(ctrl_panel, label="Results")
        res_sizer = wx.StaticBoxSizer(res_box, wx.VERTICAL)

        self._results_grid = wx.FlexGridSizer(cols=3, vgap=4, hgap=10)
        self._results_grid.AddGrowableCol(1, 1)

        # Placeholder rows — replaced by _populate_results_grid()
        for _ in range(4):
            for _ in range(3):
                lbl = wx.StaticText(ctrl_panel, label="—")
                lbl.SetForegroundColour(dim)
                self._results_grid.Add(lbl, flag=wx.ALIGN_LEFT)

        res_sizer.Add(self._results_grid, flag=wx.EXPAND | wx.ALL, border=6)

        # Footnote
        self._footnote = wx.StaticText(
            ctrl_panel,
            label="* relative to total object area"
        )
        self._footnote.SetForegroundColour(wx.Colour(110, 110, 110))
        res_sizer.Add(self._footnote, flag=wx.LEFT | wx.BOTTOM, border=6)

        ctrl.Add(res_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        # Status label (auto-run feedback)
        self._result_label = wx.StaticText(ctrl_panel, label="Not yet computed.")
        self._result_label.SetForegroundColour(wx.Colour(140, 140, 140))
        ctrl.Add(self._result_label, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)

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

        # ── Separator + buttons ───────────────────────────────────────
        ctrl.Add(wx.StaticLine(ctrl_panel), flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=8)

        self._run_btn = wx.Button(ctrl_panel, label="Run Coverage")
        self._run_btn.Bind(wx.EVT_BUTTON, self._on_run)
        ctrl.Add(self._run_btn, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=8)

        self._confirm_btn = wx.Button(ctrl_panel, label="Done — next image →")
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

    def _populate_results_grid(self, result) -> None:
        """Fill the 3-column results grid from a CoverageResult."""
        self._results_grid.Clear(delete_windows=True)
        dim = wx.Colour(160, 160, 160)
        ctrl_panel = self._ctrl_panel

        def _fmt(px: int, cm2: float | None) -> str:
            if cm2 is not None:
                return f"{cm2:.4f} cm\u00b2"
            return f"{px:,} px"

        rows = [
            ("Eval area",  _fmt(result.area_pixels,      result.area_cm2),      "—"),
            ("Covered",    _fmt(result.covered_pixels,    result.covered_cm2),   f"{result.coverage_pct:.2f}%"),
            ("Uncovered",  _fmt(result.uncovered_pixels,  result.uncovered_cm2), f"{100 - result.coverage_pct:.2f}%"),
            ("Outside",    _fmt(result.outside_pixels,    result.outside_cm2),   f"{result.outside_pct:.2f}% *"),
        ]
        for label, px, pct in rows:
            for text in (label, px, pct):
                lbl = wx.StaticText(ctrl_panel, label=text)
                lbl.SetForegroundColour(dim)
                self._results_grid.Add(lbl, flag=wx.ALIGN_LEFT)

        ctrl_panel.Layout()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_run(self, _event) -> None:
        record       = self._state.current_record
        output_dir   = self._state.output_dir()
        internal_dir = self._state.internal_dir()
        if record is None or output_dir is None or internal_dir is None:
            return

        self._run_btn.Disable()

        stem     = record.path.stem
        seg_path = internal_dir / f"{stem}_segmentation_mask.png"

        # Load eval area entry from consolidated JSON
        eval_areas_path = internal_dir / "eval_areas.json"
        eval_areas_entry: dict | None = None
        if eval_areas_path.exists():
            try:
                with open(eval_areas_path) as f:
                    all_areas = json.load(f)
                eval_areas_entry = all_areas.get(record.path.name)
            except Exception:
                pass

        if not seg_path.exists():
            wx.MessageBox(
                f"Missing segmentation mask:\n{seg_path}",
                "Cannot evaluate",
                wx.OK | wx.ICON_ERROR,
            )
            self._run_btn.Enable()
            return

        if eval_areas_entry is None:
            wx.MessageBox(
                "No evaluation area defined for this image.",
                "Cannot evaluate",
                wx.OK | wx.ICON_ERROR,
            )
            self._run_btn.Enable()
            return

        try:
            obj_mask = _load_binary(seg_path)

            # Rasterise eval polygon on the fly
            img_w, img_h = eval_areas_entry["image_size"]
            verts_norm = [(v[0], v[1]) for v in eval_areas_entry.get("vertices", [])]
            skipped    = eval_areas_entry.get("skipped", False)
            area_mask_raw = rasterise_eval_polygon(verts_norm, img_w, img_h, skipped)
            area_mask = area_mask_raw > 0

            self._result = compute_coverage(
                obj_mask, area_mask,
                seg_path.name, "eval_area",
            )

            # Populate physical area fields if calibration is available
            cal = self._state.calibration
            if cal is not None:
                self._result.area_cm2      = cal.px_to_cm2(self._result.area_pixels)
                self._result.covered_cm2   = cal.px_to_cm2(self._result.covered_pixels)
                self._result.uncovered_cm2 = cal.px_to_cm2(self._result.uncovered_pixels)
                self._result.outside_cm2   = cal.px_to_cm2(self._result.outside_pixels)

            # Store masks so the opacity slider can re-render without re-computing
            self._last_obj_mask      = obj_mask
            self._last_area_mask     = area_mask
            self._last_source_bgr    = self._state.current_image
            self._last_area_entry    = eval_areas_entry

            # Build and display overlay on canvas
            overlay = render_overlay(
                self._state.current_image, obj_mask, area_mask,
                alpha=self._overlay_alpha,
            )
            overlay = draw_legend(overlay, self._result)
            self._canvas.set_image(overlay, bgr=True)

            # Show numeric results in grid
            self._populate_results_grid(self._result)
            self._result_label.SetLabel("")

            # Save user outputs
            self._save_results(obj_mask, area_mask, eval_areas_entry, output_dir, stem)

            self._confirm_btn.Enable()
            self._run_btn.SetLabel("Re-run Coverage")
            self._run_btn.Enable()

        except Exception as exc:
            self._run_btn.Enable()
            wx.MessageBox(str(exc), "Coverage error", wx.OK | wx.ICON_ERROR)

    def _on_opacity_slider(self, _evt) -> None:
        self._overlay_alpha = self._opacity_slider.GetValue() / 10.0
        self._opacity_label.SetLabel(f"{self._overlay_alpha:.1f}")
        self._rerender_overlay()

    def _rerender_overlay(self) -> None:
        if self._last_obj_mask is None or self._result is None:
            return
        overlay = render_overlay(
            self._last_source_bgr, self._last_obj_mask, self._last_area_mask,
            alpha=self._overlay_alpha,
        )
        overlay = draw_legend(overlay, self._result)
        self._canvas.set_image(overlay, bgr=True)

    def _on_confirm(self, _event) -> None:
        if self._state.current_record:
            self._state.current_record.evaluation_status = StepStatus.COMPLETE
        if self._on_complete_callback:
            self._on_complete_callback()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_results(
        self,
        obj_mask: np.ndarray,
        area_mask: np.ndarray,
        eval_area_entry: dict,
        output_dir: Path,
        stem: str,
    ) -> None:
        source_bgr = self._state.current_image
        img_h, img_w = source_bgr.shape[:2]

        # ── Compute crop bounding box from eval polygon ───────────────
        vertices_norm = eval_area_entry.get("vertices", [])
        skipped       = eval_area_entry.get("skipped", False)

        if skipped or len(vertices_norm) < 3:
            x0, y0, x1, y1 = 0, 0, img_w, img_h
        else:
            xs = [int(v[0] * img_w) for v in vertices_norm]
            ys = [int(v[1] * img_h) for v in vertices_norm]
            x0 = max(0, min(xs))
            y0 = max(0, min(ys))
            x1 = min(img_w, max(xs) + 1)
            y1 = min(img_h, max(ys) + 1)

        # Cropped slices
        crop_bgr        = source_bgr[y0:y1, x0:x1]
        crop_area_alpha = (area_mask[y0:y1, x0:x1].astype(np.uint8)) * 255
        crop_seg_alpha  = (
            (obj_mask[y0:y1, x0:x1] & area_mask[y0:y1, x0:x1]).astype(np.uint8)
        ) * 255

        # ── {stem}_cropped.png: BGRA — original crop, alpha = eval area ──
        bgra_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        bgra_crop[:, :, 3] = crop_area_alpha

        cropped_dir = output_dir / "cropped"
        cropped_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(cropped_dir / f"{stem}_cropped.png"), bgra_crop)

        # ── {stem}_area.png: BGRA — original crop, alpha = seg ∩ eval area ──
        bgra_area = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        bgra_area[:, :, 3] = crop_seg_alpha

        area_dir = output_dir / "area"
        area_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(area_dir / f"{stem}_area.png"), bgra_area)

        # ── coverage_data.csv: append / update row for this image ────
        csv_path = output_dir / "coverage_data.csv"
        _write_coverage_csv_row(csv_path, stem, self._result)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _try_restore_result(self) -> bool:
        """
        If a coverage_data.csv row exists for the current image, restore the
        result.  Returns True if restoration succeeded (caller skips auto-run),
        False if no prior result was found.
        """
        record       = self._state.current_record
        output_dir   = self._state.output_dir()
        internal_dir = self._state.internal_dir()
        if record is None or output_dir is None or internal_dir is None:
            return False

        stem = record.path.stem
        csv_path = output_dir / "coverage_data.csv"

        row = _read_coverage_csv_row(csv_path, stem)
        if row is None:
            return False

        try:
            def _opt_float(val: str) -> float | None:
                return float(val) if val not in ("", "None", None) else None

            self._result = CoverageResult(
                object_mask_file  = row.get("filename", stem),
                area_mask_file    = "eval_area",
                area_pixels       = int(row["area_px"]),
                covered_pixels    = int(row["covered_px"]),
                uncovered_pixels  = int(row["uncovered_px"]),
                outside_pixels    = int(row["outside_px"]),
                coverage_pct      = float(row["coverage_pct"]),
                outside_pct       = float(row["outside_pct"]),
                image_width       = self._state.current_image.shape[1] if self._state.current_image is not None else 0,
                image_height      = self._state.current_image.shape[0] if self._state.current_image is not None else 0,
                area_cm2          = _opt_float(row.get("area_cm2", "")),
                covered_cm2       = _opt_float(row.get("covered_cm2", "")),
                uncovered_cm2     = _opt_float(row.get("uncovered_cm2", "")),
                outside_cm2       = _opt_float(row.get("outside_cm2", "")),
            )
            self._populate_results_grid(self._result)
            self._result_label.SetLabel("")
            self._run_btn.SetLabel("Re-run Coverage")
            self._run_btn.Enable()
            self._confirm_btn.Enable()

            # Try to load source masks for live opacity slider
            seg_path = internal_dir / f"{stem}_segmentation_mask.png"
            eval_areas_path = internal_dir / "eval_areas.json"
            masks_loaded = False

            if seg_path.exists() and eval_areas_path.exists():
                try:
                    with open(eval_areas_path) as f:
                        all_areas = json.load(f)
                    entry = all_areas.get(record.path.name)
                    if entry is not None:
                        img_w, img_h = entry["image_size"]
                        verts = [(v[0], v[1]) for v in entry.get("vertices", [])]
                        area_mask_raw = rasterise_eval_polygon(
                            verts, img_w, img_h, entry.get("skipped", False)
                        )
                        self._last_obj_mask   = _load_binary(seg_path)
                        self._last_area_mask  = area_mask_raw > 0
                        self._last_source_bgr = self._state.current_image
                        self._last_area_entry = entry
                        overlay = render_overlay(
                            self._last_source_bgr,
                            self._last_obj_mask,
                            self._last_area_mask,
                            alpha=self._overlay_alpha,
                        )
                        overlay = draw_legend(overlay, self._result)
                        self._canvas.set_image(overlay, bgr=True)
                        masks_loaded = True
                except Exception:
                    pass

            # Fallback: show the saved cropped image on the canvas
            if not masks_loaded:
                cropped_path = output_dir / "cropped" / f"{stem}_cropped.png"
                if cropped_path.exists():
                    saved = cv2.imread(str(cropped_path), cv2.IMREAD_UNCHANGED)
                    if saved is not None:
                        # Convert BGRA → BGR for display
                        if saved.shape[2] == 4:
                            saved = cv2.cvtColor(saved, cv2.COLOR_BGRA2BGR)
                        self._canvas.set_image(saved, bgr=True)

            return True

        except Exception:
            return False  # silently fall back — auto-run will fire

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_input_labels(self) -> None:
        record       = self._state.current_record
        internal_dir = self._state.internal_dir()
        if record is None or internal_dir is None:
            return

        stem = record.path.stem

        seg_path = internal_dir / f"{stem}_segmentation_mask.png"
        seg_ok   = seg_path.exists()

        # Eval area status: check consolidated JSON for entry
        eval_areas_path = internal_dir / "eval_areas.json"
        area_ok = False
        if eval_areas_path.exists():
            try:
                with open(eval_areas_path) as f:
                    area_ok = record.path.name in json.load(f)
            except Exception:
                pass

        seg_tick  = "✓" if seg_ok  else "✗"
        area_tick = "✓" if area_ok else "✗"
        seg_name  = seg_path.name if len(seg_path.name) <= 18 else seg_path.name[:15] + "…"

        self._seg_label.SetLabel(f"Seg mask:   {seg_tick} {seg_name}")
        self._seg_label.SetToolTip(str(seg_path))
        self._seg_label.SetForegroundColour(
            wx.Colour(140, 200, 140) if seg_ok else wx.Colour(200, 100, 100)
        )
        self._area_label.SetLabel(f"Area mask:  {area_tick} eval_areas.json")
        self._area_label.SetToolTip(str(eval_areas_path))
        self._area_label.SetForegroundColour(
            wx.Colour(140, 200, 140) if area_ok else wx.Colour(200, 100, 100)
        )