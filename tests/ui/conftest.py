"""
Headless Qt: set platform before any PySide6 import.

Uses ``offscreen`` (no display). On unusual platforms you can override with
``QT_QPA_PLATFORM`` in the environment before invoking pytest.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Callable

# Must run before importing Qt or ui.*
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings

import ui.workspace as workspace_module


@pytest.fixture
def english_gui_locale() -> Generator[None, None, None]:
    """
    Force :mod:`gettext` via :class:`mb.utils.translations.I18N` to English.

    Page log lines use ``_()``, so substring assertions on English fail when the
    OS locale or an earlier test left another language installed on ``I18N``.
    """
    from mb.utils.translations import I18N

    prev_lang = os.environ.get("LANG")
    prev_short = I18N.locale
    I18N.install_locale("en", verbose=False)
    os.environ["LANG"] = "en"
    try:
        yield
    finally:
        if prev_lang is None:
            os.environ.pop("LANG", None)
        else:
            os.environ["LANG"] = prev_lang
        I18N.install_locale(prev_short or "en", verbose=False)


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
