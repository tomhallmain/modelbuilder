"""Export page tests: validation, state round-trip, and run callback wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from ui.pages.export_page import ExportPage


@pytest.mark.ui
def test_export_page_validate_enables_export_when_inputs_valid(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    input_pth = tmp_path / "stub.pth"
    input_pth.write_bytes(b"x")
    out_dir = tmp_path / "bundle"
    page = ExportPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(input_pth))
    page.output_dir.setText(str(out_dir))
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_export.isEnabled()
    assert "valid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_export_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path: Path) -> None:
    page = ExportPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(tmp_path / "a.pth"))
    page.output_dir.setText(str(tmp_path / "bundle"))
    page.architecture.setText("resnet18")
    page.num_classes.setValue(3)
    page.class_names.setText("a, b, c")
    page.data_dir.setText(str(tmp_path / "data"))
    page.image_size.setValue(224)
    page.run_id.setText("rid123")
    page.snapshot_path.setText(str(tmp_path / "snapshot_rid123.json"))
    page.emit_arch_py.setChecked(False)
    blob = page.collect_gui_state()

    page2 = ExportPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.input_model.text() == str(tmp_path / "a.pth")
    assert page2.output_dir.text() == str(tmp_path / "bundle")
    assert page2.architecture.text() == "resnet18"
    assert page2.num_classes.value() == 3
    assert page2.class_names.text() == "a, b, c"
    assert page2.data_dir.text() == str(tmp_path / "data")
    assert page2.image_size.value() == 224
    assert page2.run_id.text() == "rid123"
    assert page2.snapshot_path.text() == str(tmp_path / "snapshot_rid123.json")
    assert page2.emit_arch_py.isChecked() is False


@pytest.mark.ui
def test_export_page_run_uses_export_bundle_result(monkeypatch: pytest.MonkeyPatch, qtbot, tmp_path: Path) -> None:
    input_pth = tmp_path / "stub.pth"
    input_pth.write_bytes(b"x")
    out_dir = tmp_path / "bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    page = ExportPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(input_pth))
    page.output_dir.setText(str(out_dir))

    def fake_export_bundle(**kwargs):
        return {
            "weights_path": str(Path(kwargs["output_dir"]) / "model.safetensors"),
            "manifest_path": str(Path(kwargs["output_dir"]) / "model_manifest.json"),
            "architecture_path": None,
        }

    monkeypatch.setattr("ui.pages.export_page.export_bundle", fake_export_bundle)
    payload = page._collect_inputs()
    result = page._execute_export(type("Ctx", (), {"cancel_event": type("E", (), {"is_set": lambda self: False})()})(), payload)
    page._on_success(result)
    text = page.output.toPlainText().lower()
    assert "export succeeded" in text
    assert "model.safetensors" in text
    assert "model_manifest.json" in text

