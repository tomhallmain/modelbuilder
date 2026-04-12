"""Data page: validation and cache round-trip without running pipeline tasks."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt

from mb.data.class_layout import SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.models.types import ModelBuildStepCommand
from ui.pages.data_page import DataPage
from ui.task_context import LongTaskContext


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
    assert blob.get("pipeline", {}).get("raw_data_dir") == str(tmp_path / "raw")
    page2 = DataPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.tabs.currentIndex() == 2
    assert page2.dedup_raw_data_dir.text() == str(tmp_path / "raw")
    assert page2.convert_raw_data_dir.text() == str(tmp_path / "raw")


@pytest.mark.ui
def test_data_page_wildcard_command_persists_in_gui_state_roundtrip(qtbot, english_gui_locale) -> None:
    page = DataPage()
    qtbot.addWidget(page)
    # Last entry in gui_wildcard_command_values() is fix-jpeg-extension-mismatch.
    page.wildcard_command_combo.setCurrentIndex(page.wildcard_command_combo.count() - 1)
    page.wildcard_extra_args.setPlainText("--dry-run\n")

    blob = page.collect_gui_state()
    assert blob.get("wildcard", {}).get("command") == "fix-jpeg-extension-mismatch"

    blob_json = json.loads(json.dumps(blob))
    page2 = DataPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob_json)

    assert page2.wildcard_command_combo.currentIndex() == page.wildcard_command_combo.count() - 1
    assert page2.collect_gui_state().get("wildcard", {}).get("command") == "fix-jpeg-extension-mismatch"
    assert page2.wildcard_extra_args.toPlainText() == "--dry-run\n"


@pytest.mark.ui
def test_wildcard_fix_jpeg_passes_config_before_data_to_mb_main(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    """GUI must invoke ``main([--config, PATH, data, <sub>, …])`` so argparse accepts ``--config``."""
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("model:\n  default_type: image_classification\n", encoding="utf-8")
    raw = tmp_path / "raw_data"
    raw.mkdir()
    (raw / "cls").mkdir()

    page = DataPage()
    qtbot.addWidget(page)

    ctx = LongTaskContext(threading.Event(), lambda *_a: None)
    payload = {
        "wildcard_cli": True,
        "data_subcommand": ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH,
        "extra_argv": ["--dry-run", "--raw-data-dir", str(raw)],
        "pipeline_config_path": cfg,
    }
    with patch("mb.cli.main", return_value=0) as mock_main:
        ok = page._execute_wildcard_mb_data_cli(
            ctx, ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH, payload
        )
    assert ok is True
    mock_main.assert_called_once()
    argv = mock_main.call_args[0][0]
    assert argv[:3] == ["--config", str(cfg), "data"]
    assert argv[3] == "fix-jpeg-extension-mismatch"


@pytest.mark.ui
def test_wildcard_without_pipeline_config_omits_config_prefix(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    raw = tmp_path / "raw_data"
    raw.mkdir()

    page = DataPage()
    qtbot.addWidget(page)
    ctx = LongTaskContext(threading.Event(), lambda *_a: None)
    payload = {
        "wildcard_cli": True,
        "data_subcommand": ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH,
        "extra_argv": ["--dry-run", "--raw-data-dir", str(raw)],
        "pipeline_config_path": None,
    }
    with patch("mb.cli.main", return_value=0) as mock_main:
        page._execute_wildcard_mb_data_cli(
            ctx, ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH, payload
        )
    argv = mock_main.call_args[0][0]
    assert argv[0] == "data"
    assert argv[1] == "fix-jpeg-extension-mismatch"
