"""Data page: validation and cache round-trip without running pipeline tasks."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from mb.data.class_layout import SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.models.types import ModelBuildStepCommand
from ui.pages.data_page import DataPage


@pytest.mark.ui
def test_data_page_validate_enables_run_on_create_dataset_tab(
    qtbot, english_gui_locale, synthetic_raw_data_dir, tmp_path: Path
) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    page.tabs.setCurrentIndex(4)
    out_data = tmp_path / "out_data"
    page.dataset_raw_data_dir.setText(str(synthetic_raw_data_dir))
    page.dataset_data_dir.setText(str(out_data))
    page.dataset_test_per_class.setValue(2)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_run.isEnabled()
    assert "look valid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_data_page_validate_disables_run_when_raw_data_missing(
    qtbot, english_gui_locale, tmp_path
) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    page.tabs.setCurrentIndex(4)
    page.dataset_raw_data_dir.setText(str(tmp_path / "missing_raw"))
    page.dataset_data_dir.setText(str(tmp_path / "data"))
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert not page.btn_run.isEnabled()
    assert "invalid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_data_page_validate_enables_run_on_gather_tab(qtbot, synthetic_raw_data_dir) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    page.tabs.setCurrentIndex(0)
    page.gather_source.setText(str(synthetic_raw_data_dir))
    page.gather_subdirs.setText(SYNTHETIC_DEFAULT_CLASS_NAMES[0])
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_run.isEnabled()


@pytest.mark.ui
def test_data_page_space_precheck_updates_status_and_log(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    """Convert / create-dataset space heuristics update the status line and output (headless)."""
    raw = tmp_path / "raw_space"
    raw.mkdir()
    out = tmp_path / "data_out"
    out.mkdir()
    page = DataPage()
    qtbot.addWidget(page)

    ok_c = page._space_precheck_ui(
        ModelBuildStepCommand.CONVERT,
        {"raw_data_dir": raw, "skip_space_check": False, "pipeline_config_path": None},
    )
    assert ok_c is True
    status_c = page._space_estimate_status.text()
    assert "Latest space check" in status_c
    assert "Convert" in status_c
    assert "[space]" in page.output.toPlainText()

    page.output.clear()
    ok_d = page._space_precheck_ui(
        ModelBuildStepCommand.CREATE_DATASET,
        {
            "raw_data_dir": raw,
            "data_dir": out,
            "skip_space_check": False,
            "pipeline_config_path": None,
        },
    )
    assert ok_d is True
    status_d = page._space_estimate_status.text()
    assert "Latest space check" in status_d
    assert "Create-dataset" in status_d or "dataset" in status_d.lower()
    assert "[space]" in page.output.toPlainText()


@pytest.mark.ui
def test_data_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    page.tabs.setCurrentIndex(2)
    page.dedup_raw_data_dir.setText(str(tmp_path / "raw"))
    blob = page.collect_gui_state()
    page2 = DataPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.tabs.currentIndex() == 2
    assert page2.dedup_raw_data_dir.text() == str(tmp_path / "raw")
