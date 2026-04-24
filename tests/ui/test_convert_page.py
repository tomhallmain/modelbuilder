"""Convert page: validation and cache round-trip without running conversion."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from ui.pages.convert_page import ConvertPage
from mb.models.types import ArchitectureType


@pytest.mark.ui
def test_convert_page_validate_enables_convert_when_inputs_valid(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    input_pth = tmp_path / "stub.pth"
    input_pth.write_bytes(b"")
    output_onnx = tmp_path / "out.onnx"
    page = ConvertPage()
    qtbot.addWidget(page)
    page.framework.setCurrentIndex(1)
    page.target.setCurrentIndex(0)
    page.input_model.setText(str(input_pth))
    page.output_model.setText(str(output_onnx))
    page.architecture.setText(ArchitectureType.RESNET18.value)
    page.num_classes.setValue(3)
    page.image_size.setValue(224)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_convert.isEnabled()
    assert "valid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_convert_page_validate_disables_convert_when_input_missing(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    page = ConvertPage()
    qtbot.addWidget(page)
    page.framework.setCurrentIndex(1)
    page.target.setCurrentIndex(0)
    page.input_model.setText(str(tmp_path / "nope.pth"))
    page.output_model.setText(str(tmp_path / "out.onnx"))
    page.architecture.setText(ArchitectureType.RESNET18.value)
    page.num_classes.setValue(3)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert not page.btn_convert.isEnabled()
    assert "invalid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_convert_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path: Path) -> None:
    page = ConvertPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(tmp_path / "a.pth"))
    page.output_model.setText(str(tmp_path / "b.onnx"))
    page.framework.setCurrentIndex(2)
    page.target.setCurrentIndex(1)
    page.architecture.setText(ArchitectureType.EFFICIENTNET_B0.value)
    page.num_classes.setValue(5)
    page.image_size.setValue(299)
    blob = page.collect_gui_state()
    page2 = ConvertPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.input_model.text() == str(tmp_path / "a.pth")
    assert page2.output_model.text() == str(tmp_path / "b.onnx")
    assert page2.framework.currentIndex() == 2
    assert page2.target.currentIndex() == 1
    assert page2.architecture.text() == ArchitectureType.EFFICIENTNET_B0.value
    assert page2.num_classes.value() == 5
    assert page2.image_size.value() == 299


@pytest.mark.ui
def test_convert_page_validate_safetensors_pytorch_without_arch_num_classes(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    input_pth = tmp_path / "stub.pth"
    input_pth.write_bytes(b"x")
    output_st = tmp_path / "out.safetensors"
    page = ConvertPage()
    qtbot.addWidget(page)
    page.framework.setCurrentIndex(1)  # pytorch
    page.target.setCurrentIndex(1)  # safetensors
    page.input_model.setText(str(input_pth))
    page.output_model.setText(str(output_st))
    page.architecture.setText("")
    page.num_classes.setValue(0)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_convert.isEnabled()
    assert "valid" in page.output.toPlainText().lower()
