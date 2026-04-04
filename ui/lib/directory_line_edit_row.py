"""Line edit plus directory browse button (shared by settings pages)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

from mb.utils.translations import _
from ui.lib.fast_directory_picker_qt import get_existing_directory


def make_directory_line_edit_row(
    owner: QWidget,
    line_edit: QLineEdit,
    *,
    dialog_title: str | None = None,
) -> QWidget:
    """
    Return a row widget containing *line_edit* and a right-aligned **Browse** button.

    *owner* is used as the modal parent for :func:`get_existing_directory` (typically
    the page widget).
    """
    row = QWidget(owner)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    title = dialog_title or _("Select directory")
    btn = QPushButton(_("Browse..."))

    def _pick() -> None:
        start = line_edit.text().strip() or str(Path.cwd())
        picked = get_existing_directory(owner, title, start)
        if picked:
            line_edit.setText(picked)

    btn.clicked.connect(_pick)
    layout.addWidget(line_edit, 1)
    layout.addWidget(btn, 0)
    return row
