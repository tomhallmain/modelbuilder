"""Qt test helpers (avoid picking nested :class:`QStackedWidget` inside tab pages)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QStackedWidget, QWidget

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


def stacked_inner_page(main_window: MainWindow, nav_index: int) -> QWidget:
    """Return the inner page widget at *nav_index* (not the wrapping :class:`QScrollArea`)."""
    stack = main_nav_stacked_widget(main_window)
    scroll = stack.widget(nav_index)
    assert scroll is not None
    assert isinstance(scroll, QScrollArea), type(scroll)
    inner = scroll.widget()
    assert inner is not None
    return inner
