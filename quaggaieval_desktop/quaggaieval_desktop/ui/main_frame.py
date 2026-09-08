"""
MainFrame — top-level wx.Frame.

Layout (two swappable pages via a wx.Simplebook):
  Page 0 — main_page:  folder bar + image list sidebar + tabbed notebook (3 processing steps)
  Page 1 — settings_panel: full-area settings subpage

A gear button in the folder bar toggles between pages.
SAM2 model loading starts automatically on __init__ using default config.
"""
from __future__ import annotations
import threading
from pathlib import Path
import cv2
import wx

from ..core.migration import migrate_folder
from ..core.state import AppState, StepStatus
from .calibration_modal import CalibrationModal
from .image_list import ImageListPanel
from .settings_panel import SettingsPanel
from .tab_a_segmentation import SegmentationPanel
from .tab_b_eval_area import EvalAreaPanel
from .tab_c_evaluation import AreaEvaluationPanel
from .tab_mask_correction import MaskCorrectionPanel

TAB_EVAL_AREA       = 0
TAB_SEGMENTATION    = 1
TAB_MASK_CORRECTION = 2
TAB_COVERAGE        = 3
TAB_LABELS = [
    "1 · Evaluation Area",
    "2 · Segmentation",
    "3 · Mask Correction",
    "4 · Coverage",
]

PAGE_MAIN     = 0
PAGE_SETTINGS = 1


class MainFrame(wx.Frame):

    def __init__(self):
        super().__init__(
            None,
            title="Image Evaluation Pipeline",
            size=(1440, 900),
            style=wx.DEFAULT_FRAME_STYLE,
        )
        self._state = AppState()
        self._build_ui()
        self._apply_dark_theme()
        self.Centre()

        # Start model loading with default config immediately
        self._start_model_load()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root_panel = wx.Panel(self)
        root_sizer = wx.BoxSizer(wx.VERTICAL)

        # wx.Simplebook: page-flip container (no tabs shown)
        self._book = wx.Simplebook(root_panel)

        # ── Page 0: main working area ─────────────────────────────────
        main_page = wx.Panel(self._book)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        main_sizer.Add(self._build_folder_bar(main_page), flag=wx.EXPAND)
        main_sizer.Add(wx.StaticLine(main_page), flag=wx.EXPAND)

        content = wx.BoxSizer(wx.HORIZONTAL)

        self._image_list = ImageListPanel(main_page, self._state)
        self._image_list.set_on_select(self._on_image_selected)
        content.Add(self._image_list, flag=wx.EXPAND | wx.ALL, border=4)
        content.Add(wx.StaticLine(main_page, style=wx.LI_VERTICAL), flag=wx.EXPAND)

        self._notebook = wx.Notebook(main_page)
        self._tab_area       = EvalAreaPanel(self._notebook, self._state)
        self._tab_seg        = SegmentationPanel(self._notebook, self._state)
        self._tab_mask_corr  = MaskCorrectionPanel(self._notebook, self._state)
        self._tab_eval       = AreaEvaluationPanel(self._notebook, self._state)

        self._notebook.AddPage(self._tab_area,      TAB_LABELS[TAB_EVAL_AREA])
        self._notebook.AddPage(self._tab_seg,       TAB_LABELS[TAB_SEGMENTATION])
        self._notebook.AddPage(self._tab_mask_corr, TAB_LABELS[TAB_MASK_CORRECTION])
        self._notebook.AddPage(self._tab_eval,      TAB_LABELS[TAB_COVERAGE])

        self._tab_area.set_on_complete(self._on_eval_area_complete)
        self._tab_seg.set_on_complete(self._on_segmentation_complete)
        self._tab_mask_corr.set_on_complete(self._on_mask_correction_complete)
        self._tab_eval.set_on_complete(self._on_evaluation_complete)

        self._notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self._on_tab_changing)

        content.Add(self._notebook, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        main_sizer.Add(content, proportion=1, flag=wx.EXPAND)
        main_page.SetSizer(main_sizer)

        # ── Page 1: settings ─────────────────────────────────────────
        self._settings_panel = SettingsPanel(self._book, self._state.model_config)
        self._settings_panel.set_on_back(self._show_main)
        self._settings_panel.set_on_reload_model(self._start_model_load)

        self._book.AddPage(main_page, "Main")
        self._book.AddPage(self._settings_panel, "Settings")

        root_sizer.Add(self._book, proportion=1, flag=wx.EXPAND)

        # Status bar
        self._statusbar = self.CreateStatusBar(2)
        self._statusbar.SetStatusWidths([-1, 220])

        root_panel.SetSizer(root_sizer)
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(frame_sizer)

        self._set_all_tabs_enabled(False)

    def _build_folder_bar(self, parent: wx.Window) -> wx.Sizer:
        bar_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bar_panel = wx.Panel(parent)
        bar_panel.SetMinSize((-1, 44))

        inner = wx.BoxSizer(wx.HORIZONTAL)

        lbl = wx.StaticText(bar_panel, label="Image folder:")
        lbl.SetForegroundColour(wx.Colour(200, 200, 200))
        inner.Add(lbl, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        self._folder_text = wx.TextCtrl(
            bar_panel, value="No folder selected", style=wx.TE_READONLY
        )
        self._folder_text.SetForegroundColour(wx.Colour(160, 160, 160))
        inner.Add(self._folder_text, proportion=1,
                  flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, border=8)

        browse_btn = wx.Button(bar_panel, label="Browse…")
        browse_btn.Bind(wx.EVT_BUTTON, self._on_browse)
        inner.Add(browse_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)

        settings_btn = wx.Button(bar_panel, label="⚙ Settings")
        settings_btn.Bind(wx.EVT_BUTTON, self._show_settings)
        inner.Add(settings_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=10)

        bar_panel.SetSizer(inner)
        bar_sizer.Add(bar_panel, proportion=1, flag=wx.EXPAND)
        return bar_sizer

    # ------------------------------------------------------------------
    # Page switching
    # ------------------------------------------------------------------

    def _show_settings(self, _event=None) -> None:
        self._settings_panel.refresh_from_config()
        self._book.SetSelection(PAGE_SETTINGS)

    def _show_main(self) -> None:
        self._book.SetSelection(PAGE_MAIN)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _start_model_load(self) -> None:
        ok, err = self._state.model_config.is_valid()
        if not ok:
            self._set_model_status(f"Model not found — check Settings. ({err})", error=True)
            return

        self._set_model_status("Loading SAM2 model…")
        self._state.sam2_worker.load_model(
            self._state.model_config,
            on_ready=self._on_model_ready,
            on_error=self._on_model_error,
        )

    def _on_model_ready(self) -> None:
        size = self._state.model_config.model_size
        self._set_model_status(f"SAM2 ready  [{size}]")

    def _on_model_error(self, msg: str) -> None:
        self._set_model_status(f"Model error: {msg}", error=True)

    def _set_model_status(self, msg: str, error: bool = False) -> None:
        self._statusbar.SetStatusText(msg, 1)

    # ------------------------------------------------------------------
    # Folder / image loading
    # ------------------------------------------------------------------

    def _on_browse(self, _event) -> None:
        with wx.DirDialog(
            self, "Select image folder",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            folder = Path(dlg.GetPath())

        self._state.load_folder(folder)
        self._folder_text.SetValue(str(folder))

        if not self._state.has_images:
            wx.MessageBox(
                "No supported images found in this folder.\n"
                "Supported: jpg, jpeg, png, tif, tiff",
                "Empty folder",
                wx.OK | wx.ICON_WARNING,
            )
            return

        self._image_list.refresh_list()

        if self._state.needs_migration:
            self._handle_migration_flow()
            return  # _handle_migration_flow calls _load_current_image when done

        if self._state.calibration is None:
            dlg = CalibrationModal(self, self._state)
            dlg.ShowModal()
            dlg.Destroy()

        self._load_current_image()

    # ------------------------------------------------------------------
    # Migration flow
    # ------------------------------------------------------------------

    def _handle_migration_flow(self) -> None:
        """
        Called after load_folder() detects old-format data.
        Gates on calibration availability, prompts user, then runs migration.
        """
        folder = self._state.folder_path

        # Gate: calibration must exist before migration can fill in cm² values
        if self._state.calibration is None:
            wx.MessageBox(
                "This folder contains data from an older version of QuaggAI "
                "and needs to be migrated.\n\n"
                "Calibration is required before migration can proceed. "
                "Please create a calibration first.",
                "Migration — Calibration Required",
                wx.OK | wx.ICON_INFORMATION,
            )
            dlg = CalibrationModal(self, self._state)
            dlg.ShowModal()
            dlg.Destroy()
            # Reload state so calibration is picked up
            self._state.load_folder(folder)
            if self._state.calibration is None:
                # User cancelled calibration — open folder without migrating
                self._load_current_image()
                return

        # Prompt user
        answer = wx.MessageBox(
            "This folder contains data from an older version of QuaggAI.\n\n"
            "Would you like to migrate it to the new format now?\n"
            "Old files will be archived in evaluation_results/_backup/.",
            "Migrate Folder Data",
            wx.YES_NO | wx.ICON_QUESTION,
        )
        if answer != wx.YES:
            self._load_current_image()
            return

        self._run_migration()

    def _run_migration(self) -> None:
        """Start migration in a background thread with a progress dialog."""
        folder = self._state.folder_path

        progress_dlg = wx.ProgressDialog(
            "Migrating folder…",
            "Starting migration…",
            maximum=100,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME,
        )

        def on_progress(pct: int, msg: str) -> None:
            wx.CallAfter(progress_dlg.Update, pct, msg)

        def on_done(errors: list[str]) -> None:
            def _finish():
                progress_dlg.Destroy()
                if errors:
                    error_text = "\n".join(f"  • {e}" for e in errors)
                    wx.MessageBox(
                        f"Migration completed with {len(errors)} warning(s):\n\n"
                        f"{error_text}",
                        "Migration Complete (with warnings)",
                        wx.OK | wx.ICON_WARNING,
                    )
                else:
                    wx.MessageBox(
                        "Migration completed successfully.",
                        "Migration Complete",
                        wx.OK | wx.ICON_INFORMATION,
                    )
                # Reload folder state from newly migrated files
                self._state.load_folder(folder)
                self._image_list.refresh_list()
                self._load_current_image()
            wx.CallAfter(_finish)

        def on_error(msg: str) -> None:
            def _abort():
                progress_dlg.Destroy()
                wx.MessageBox(
                    f"Migration failed:\n\n{msg}",
                    "Migration Error",
                    wx.OK | wx.ICON_ERROR,
                )
                self._load_current_image()
            wx.CallAfter(_abort)

        def _worker():
            migrate_folder(folder, self._state.calibration, on_progress, on_done, on_error)

        threading.Thread(target=_worker, daemon=True).start()

    def _load_current_image(self) -> None:
        record = self._state.current_record
        if record is None:
            return

        img = cv2.imread(str(record.path))
        if img is None:
            self._set_status(f"Failed to load: {record.path.name}", error=True)
            return

        self._state.current_image = img
        self._set_status(
            f"Loaded: {record.path.name}  ({img.shape[1]}×{img.shape[0]})"
        )

        target_tab = record.first_incomplete_step
        self._update_tab_availability(record)
        self._notebook.SetSelection(target_tab)
        self._load_active_tab(target_tab)
        self._image_list.highlight_current()

    def _load_active_tab(self, tab_index: int) -> None:
        if tab_index == TAB_EVAL_AREA:
            self._tab_area.load_image()
        elif tab_index == TAB_SEGMENTATION:
            self._tab_seg.load_image()
        elif tab_index == TAB_MASK_CORRECTION:
            self._tab_mask_corr.load_image()
        elif tab_index == TAB_COVERAGE:
            self._tab_eval.load_image()

    # ------------------------------------------------------------------
    # Image list selection
    # ------------------------------------------------------------------

    def _on_image_selected(self, index: int) -> None:
        self._state.select_image(index)
        self._load_current_image()

    # ------------------------------------------------------------------
    # Step completion callbacks
    # ------------------------------------------------------------------

    def _on_eval_area_complete(self) -> None:
        self._image_list.update_record_status(self._state.current_index)
        self._update_tab_availability(self._state.current_record)
        self._notebook.SetSelection(TAB_SEGMENTATION)
        self._tab_seg.load_image()

    def _on_segmentation_complete(self) -> None:
        self._image_list.update_record_status(self._state.current_index)
        self._update_tab_availability(self._state.current_record)
        self._notebook.SetSelection(TAB_MASK_CORRECTION)
        self._tab_mask_corr.load_image()

    def _on_mask_correction_complete(self) -> None:
        self._image_list.update_record_status(self._state.current_index)
        self._update_tab_availability(self._state.current_record)
        self._notebook.SetSelection(TAB_COVERAGE)
        self._tab_eval.load_image()

    def _on_evaluation_complete(self) -> None:
        self._image_list.update_record_status(self._state.current_index)
        next_idx = self._state.next_unprocessed_index()
        if next_idx is not None:
            self._state.select_image(next_idx)
            self._load_current_image()
        else:
            wx.MessageBox(
                "All images have been processed!",
                "Complete",
                wx.OK | wx.ICON_INFORMATION,
            )

    # ------------------------------------------------------------------
    # Tab locking
    # ------------------------------------------------------------------

    def _on_tab_changing(self, event: wx.BookCtrlEvent) -> None:
        target = event.GetSelection()
        record = self._state.current_record
        if record is None:
            event.Veto()
            return

        if target == TAB_SEGMENTATION and record.eval_area_status != StepStatus.COMPLETE:
            self._set_status("Complete Evaluation Area first.", error=True)
            event.Veto()
        elif target == TAB_MASK_CORRECTION and record.segmentation_status != StepStatus.COMPLETE:
            self._set_status("Complete Segmentation first.", error=True)
            event.Veto()
        elif target == TAB_COVERAGE and record.mask_correction_status != StepStatus.COMPLETE:
            self._set_status("Complete Mask Correction first.", error=True)
            event.Veto()
        else:
            wx.CallAfter(self._load_active_tab, target)

    def _update_tab_availability(self, record) -> None:
        if record is None:
            self._set_all_tabs_enabled(False)
            return
        area_done = record.eval_area_status       == StepStatus.COMPLETE
        seg_done  = record.segmentation_status    == StepStatus.COMPLETE
        corr_done = record.mask_correction_status == StepStatus.COMPLETE
        self._notebook.SetPageText(
            TAB_SEGMENTATION,
            TAB_LABELS[TAB_SEGMENTATION] + ("" if area_done else "  🔒"),
        )
        self._notebook.SetPageText(
            TAB_MASK_CORRECTION,
            TAB_LABELS[TAB_MASK_CORRECTION] + ("" if seg_done else "  🔒"),
        )
        self._notebook.SetPageText(
            TAB_COVERAGE,
            TAB_LABELS[TAB_COVERAGE] + ("" if corr_done else "  🔒"),
        )

    def _set_all_tabs_enabled(self, enabled: bool) -> None:
        suffix = "" if enabled else "  🔒"
        for i, label in enumerate(TAB_LABELS):
            self._notebook.SetPageText(i, label + (suffix if i > 0 else ""))

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._statusbar.SetStatusText(msg, 0)

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        bg = wx.Colour(45, 45, 45)
        self.SetBackgroundColour(bg)
        self._notebook.SetBackgroundColour(bg)