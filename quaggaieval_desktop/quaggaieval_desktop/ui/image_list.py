"""
ImageListPanel — left sidebar showing all images in the folder
with per-image step status indicators.
"""
from __future__ import annotations
import wx
from ..core.state import AppState, ImageRecord, StepStatus


# Status symbol displayed next to each step
_STATUS_SYMBOL = {
    StepStatus.PENDING: "○",
    StepStatus.IN_PROGRESS: "◐",
    StepStatus.COMPLETE: "●",
}

_STATUS_COLOUR = {
    StepStatus.PENDING: wx.Colour(140, 140, 140),
    StepStatus.IN_PROGRESS: wx.Colour(255, 190, 50),
    StepStatus.COMPLETE: wx.Colour(80, 200, 100),
}


class ImageListPanel(wx.Panel):
    """
    Scrollable list of image filenames with step-status badges.
    Fires a custom EVT_IMAGE_SELECTED event when the user clicks an entry.
    """

    def __init__(self, parent: wx.Window, state: AppState, **kwargs):
        super().__init__(parent, **kwargs)
        self._state = state
        self._on_select_callback = None

        self.SetMinSize((220, -1))
        self.SetBackgroundColour(wx.Colour(30, 30, 30))

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_select(self, callback) -> None:
        """callback(index: int) called when user selects an image."""
        self._on_select_callback = callback

    def refresh_list(self) -> None:
        """Rebuild list contents from current AppState."""
        self._list.DeleteAllItems()
        for i, record in enumerate(self._state.image_records):
            self._list.InsertItem(i, record.path.name)
            self._list.SetItem(i, 1, self._status_str(record))

        if 0 <= self._state.current_index < self._list.GetItemCount():
            self._list.Select(self._state.current_index)
            self._list.EnsureVisible(self._state.current_index)

        self._update_counter()

    def highlight_current(self) -> None:
        """Update selection highlight without rebuilding the list."""
        for i in range(self._list.GetItemCount()):
            self._list.Select(i, i == self._state.current_index)
        if 0 <= self._state.current_index < self._list.GetItemCount():
            self._list.EnsureVisible(self._state.current_index)
        self._update_counter()

    def update_record_status(self, index: int) -> None:
        """Refresh the status column for a single record."""
        if 0 <= index < self._list.GetItemCount():
            record = self._state.image_records[index]
            self._list.SetItem(index, 1, self._status_str(record))
        self._update_counter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = wx.StaticText(self, label="Images")
        header.SetForegroundColour(wx.Colour(200, 200, 200))
        header.SetFont(header.GetFont().Bold())
        sizer.Add(header, flag=wx.ALL, border=8)

        # Counter label
        self._counter_label = wx.StaticText(self, label="")
        self._counter_label.SetForegroundColour(wx.Colour(140, 140, 140))
        sizer.Add(self._counter_label, flag=wx.LEFT | wx.BOTTOM, border=8)

        # List control
        self._list = wx.ListCtrl(
            self,
            style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_NO_HEADER | wx.BORDER_NONE,
        )
        self._list.SetBackgroundColour(wx.Colour(35, 35, 35))
        self._list.SetForegroundColour(wx.Colour(210, 210, 210))
        self._list.InsertColumn(0, "Name", width=140)
        self._list.InsertColumn(1, "Status", width=60)
        self._list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_item_selected)
        sizer.Add(self._list, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=4)

        self.SetSizer(sizer)

    @staticmethod
    def _status_str(record: ImageRecord) -> str:
        s = _STATUS_SYMBOL
        return (
            f"{s[record.segmentation_status]}"
            f"{s[record.eval_area_status]}"
            f"{s[record.evaluation_status]}"
        )

    def _update_counter(self) -> None:
        total = len(self._state.image_records)
        done = sum(1 for r in self._state.image_records if r.is_fully_complete)
        self._counter_label.SetLabel(f"{done}/{total} complete")

    def _on_item_selected(self, event: wx.ListEvent) -> None:
        index = event.GetIndex()
        if self._on_select_callback and index != self._state.current_index:
            self._on_select_callback(index)