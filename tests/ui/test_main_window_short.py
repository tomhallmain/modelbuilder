"""
Short headless UI checks (mirror CLI / smoke themes from other test modules).

Requires PySide6 + pytest-qt; does not run training or touch GPU.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QStackedWidget

from mb import __version__ as MB_VERSION
from ui.main_window import MainWindow
from ui.pages import ConvertPage, DataPage, HomePage, InfoPage, TrainPage


def _stacked_widget(parent) -> QStackedWidget:
    stacks = parent.findChildren(QStackedWidget)
    assert len(stacks) == 1
    return stacks[0]


@pytest.mark.ui
def test_main_window_opens_and_title(qtbot, main_window: MainWindow) -> None:
    assert "Model Builder" in main_window.windowTitle()


@pytest.mark.ui
def test_nav_lists_same_sections_as_cli_data_train_convert_flow(
    qtbot, main_window: MainWindow
) -> None:
    """Sidebar order aligns with typical ``mb`` flow: shell areas for data → train → convert."""
    nav = main_window.nav_widget
    labels = [nav.item(i).text() for i in range(nav.count())]
    assert labels == ["Home", "Data", "Train", "Convert", "Info"]


@pytest.mark.ui
def test_navigation_switches_stacked_pages(qtbot, main_window: MainWindow) -> None:
    nav = main_window.nav_widget
    stack = _stacked_widget(main_window.centralWidget())
    for row in range(nav.count()):
        nav.setCurrentRow(row)
        assert stack.currentIndex() == row


@pytest.mark.ui
def test_page_widget_types_match_nav(qtbot, main_window: MainWindow) -> None:
    stack = _stacked_widget(main_window.centralWidget())
    expected = (HomePage, DataPage, TrainPage, ConvertPage, InfoPage)
    for i, cls in enumerate(expected):
        scroll = stack.widget(i)
        inner = scroll.widget()
        assert isinstance(inner, cls)


@pytest.mark.ui
def test_status_bar_no_workspace_message(qtbot, main_window: MainWindow) -> None:
    msg = main_window.statusBar().currentMessage()
    assert "workspace" in msg.lower()


@pytest.mark.ui
def test_about_dialog_contains_mb_version(qtbot, main_window: MainWindow) -> None:
    box_holder: list[QMessageBox | None] = [None]

    def grab_and_close() -> None:
        w = QApplication.activeModalWidget()
        if isinstance(w, QMessageBox):
            box_holder[0] = w
            w.close()

    QTimer.singleShot(0, grab_and_close)
    main_window._show_about()
    qtbot.waitUntil(lambda: box_holder[0] is not None, timeout=2000)
    assert box_holder[0] is not None
    assert MB_VERSION in box_holder[0].text()
