"""Qt test helpers (avoid picking nested :class:`QStackedWidget` inside tab pages)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QHBoxLayout, QStackedWidget

if TYPE_CHECKING:
    from ui.main_window import MainWindow


def main_nav_stacked_widget(main_window: MainWindow) -> QStackedWidget:
    """The main window's nav :class:`QStackedWidget` (object name ``main_nav_stack`` or layout column 1)."""
    named = main_window.findChild(QStackedWidget, "main_nav_stack")
    if named is not None:
        return named
    central = main_window.centralWidget()
    assert central is not None
    lay = central.layout()
    assert isinstance(lay, QHBoxLayout)
    item = lay.itemAt(1)
    assert item is not None
    w = item.widget()
    assert isinstance(w, QStackedWidget), (w, type(w))
    return w
