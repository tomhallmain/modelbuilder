"""Info page: inspect helpers and cache round-trip (no full model training)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from ui.pages.info_page import InfoPage


@pytest.mark.ui
def test_info_page_inspect_model_populates_output_for_onnx_file(qtbot, tmp_path: Path) -> None:
    onnx = tmp_path / "minimal.onnx"
    onnx.write_bytes(b"")
    page = InfoPage()
    qtbot.addWidget(page)
    page.model_path.setText(str(onnx))
    qtbot.mouseClick(page.btn_model_info, Qt.MouseButton.LeftButton)
    text = page.model_output.toPlainText().lower()
    assert str(onnx) in page.model_output.toPlainText() or onnx.name in page.model_output.toPlainText()
    assert "onnx" in text


@pytest.mark.ui
def test_info_page_inspect_dataset_lists_class_counts(
    qtbot, two_class_classification_data_dir: Path
) -> None:
    page = InfoPage()
    qtbot.addWidget(page)
    page._tabs.setCurrentIndex(1)
    page.dataset_dir.setText(str(two_class_classification_data_dir))
    qtbot.mouseClick(page.btn_dataset_info, Qt.MouseButton.LeftButton)
    out = page.dataset_output.toPlainText()
    assert "train" in out.lower()
    assert "class_a" in out or "class_b" in out


@pytest.mark.ui
def test_info_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path: Path) -> None:
    page = InfoPage()
    qtbot.addWidget(page)
    page._tabs.setCurrentIndex(1)
    page.model_path.setText(str(tmp_path / "m.onnx"))
    page.dataset_dir.setText(str(tmp_path / "data"))
    blob = page.collect_gui_state()
    page2 = InfoPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2._tabs.currentIndex() == 1
    assert page2.model_path.text() == str(tmp_path / "m.onnx")
    assert page2.dataset_dir.text() == str(tmp_path / "data")
