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

from tests.ui.qt_helpers import main_nav_stacked_widget


@pytest.mark.ui
def test_main_window_opens_and_title(qtbot, main_window: MainWindow) -> None:
    assert "Model Builder" in main_window.windowTitle()


@pytest.mark.ui
def test_main_nav_stack_object_name_matches_helper(qtbot, main_window: MainWindow) -> None:
    w = main_window.findChild(QStackedWidget, "main_nav_stack")
    assert w is not None
    assert w is main_nav_stacked_widget(main_window)


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
    stack = main_nav_stacked_widget(main_window)
    for row in range(nav.count()):
        nav.setCurrentRow(row)
        assert stack.currentIndex() == row


@pytest.mark.ui
def test_page_widget_types_match_nav(qtbot, main_window: MainWindow) -> None:
    stack = main_nav_stacked_widget(main_window)
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
    """Copy About text before closing — ``close()`` deletes the C++ side immediately."""
    texts: list[str] = []

    def grab_text_and_close() -> None:
        w = QApplication.activeModalWidget()
        if isinstance(w, QMessageBox):
            texts.append(w.text())
            w.close()

    QTimer.singleShot(0, grab_text_and_close)
    main_window._show_about()
    qtbot.waitUntil(lambda: len(texts) > 0, timeout=2000)
    assert MB_VERSION in texts[0]
