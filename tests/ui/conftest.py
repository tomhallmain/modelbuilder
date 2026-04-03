"""
Headless Qt: set platform before any PySide6 import.

Uses ``offscreen`` (no display). On unusual platforms you can override with
``QT_QPA_PLATFORM`` in the environment before invoking pytest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

# Must run before importing Qt or ui.*
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings

import ui.workspace as workspace_module


@pytest.fixture
def isolated_qsettings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], QSettings]:
    """Avoid touching real ``QSettings`` (registry / plist) during tests."""
    ini_path = tmp_path / "gui_test_settings.ini"
    store: dict[str, QSettings | None] = {"s": None}

    def get_settings() -> QSettings:
        if store["s"] is None:
            store["s"] = QSettings(str(ini_path), QSettings.Format.IniFormat)
        return store["s"]

    # :class:`MainWindow` and pages import ``default_settings`` from ``ui.workspace``.
    monkeypatch.setattr(workspace_module, "default_settings", get_settings)
    return get_settings


@pytest.fixture
def main_window(qtbot, isolated_qsettings: Callable[[], QSettings]):
    """A :class:`ui.main_window.MainWindow` using temp QSettings; registered with qtbot."""
    from ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win
