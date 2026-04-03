"""Main window: workspace menu with mocked file dialog (headless)."""

from __future__ import annotations

import pytest

from ui.main_window import MainWindow


@pytest.mark.ui
def test_choose_workspace_updates_status_bar(qtbot, main_window: MainWindow, tmp_path, monkeypatch) -> None:
    ws = tmp_path / "my_workspace"
    ws.mkdir()

    def fake_get_existing_directory(*_args, **_kwargs):
        return str(ws)

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getExistingDirectory",
        fake_get_existing_directory,
    )
    main_window._choose_workspace()
    msg = main_window.statusBar().currentMessage()
    assert str(ws) in msg
