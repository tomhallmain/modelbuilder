"""
Single headless UI E2E: open shell, walk all nav targets, About dialog, consistent state.

Marked ``slow`` + ``ui_e2e`` so CI can filter; uses ``QT_QPA_PLATFORM=offscreen`` from ``conftest``.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QStackedWidget

from mb import __version__ as MB_VERSION
from ui.main_window import MainWindow


def _stack(main_window: MainWindow) -> QStackedWidget:
    c = main_window.centralWidget()
    assert c is not None
    stacks = c.findChildren(QStackedWidget)
    assert len(stacks) == 1
    return stacks[0]


@pytest.mark.slow
@pytest.mark.ui
@pytest.mark.ui_e2e
def test_headless_ui_full_navigation_and_about(qtbot, main_window: MainWindow) -> None:
    nav = main_window.nav_widget
    stack = _stack(main_window)

    for row in range(nav.count()):
        nav.setCurrentRow(row)
        qtbot.wait(20)
        assert stack.currentIndex() == row
        assert nav.currentRow() == row

    closed: list[bool] = []

    def close_about() -> None:
        w = QApplication.activeModalWidget()
        if isinstance(w, QMessageBox):
            assert MB_VERSION in w.text()
            w.close()
            closed.append(True)

    QTimer.singleShot(0, close_about)
    main_window._show_about()
    qtbot.waitUntil(lambda: bool(closed), timeout=3000)

    nav.setCurrentRow(0)
    assert stack.currentIndex() == 0
