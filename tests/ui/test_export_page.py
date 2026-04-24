"""Export page tests: validation, state round-trip, and run callback wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

from ui.pages.export_page import ExportPage


@pytest.mark.ui
def test_export_page_validate_enables_export_when_inputs_valid(
    qtbot, english_gui_locale, tmp_path: Path
) -> None:
    input_pth = tmp_path / "stub.pth"
    input_pth.write_bytes(b"x")
    page = ExportPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(input_pth))
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_export.isEnabled()
    assert "valid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_export_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path: Path) -> None:
    page = ExportPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(tmp_path / "a.pth"))
    page.architecture.setText("resnet18")
    page.num_classes.setValue(3)
    page.data_dir.setText(str(tmp_path / "data"))
    page.image_size.setValue(224)
    page.run_id.setText("rid123")
    page.emit_arch_py.setChecked(False)
    blob = page.collect_gui_state()

    page2 = ExportPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.input_model.text() == str(tmp_path / "a.pth")
    assert page2.architecture.text() == "resnet18"
    assert page2.num_classes.value() == 3
    assert page2.data_dir.text() == str(tmp_path / "data")
    assert page2.image_size.value() == 224
    assert page2.run_id.text() == "rid123"
    assert page2.emit_arch_py.isChecked() is False


@pytest.mark.ui
def test_export_page_run_uses_export_bundle_result(monkeypatch: pytest.MonkeyPatch, qtbot, tmp_path: Path) -> None:
    input_pth = tmp_path / "stub.pth"
    input_pth.write_bytes(b"x")
    page = ExportPage()
    qtbot.addWidget(page)
    page.input_model.setText(str(input_pth))

    def fake_export_bundle(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        return {
            "weights_path": str(out_dir / "model.safetensors"),
            "manifest_path": str(out_dir / "model_manifest.json"),
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


@pytest.mark.ui
def test_export_page_prefills_from_pipeline_and_latest_run_id(
    monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    class _PC:
        def get(self, key: str, default=None):
            vals = {
                "model.default_architecture": "resnet34",
                "data.data_dir": "data",
                "data.raw_data_dir": "raw_data",
                "data.image_size": 320,
            }
            return vals.get(key, default)

        def to_dict(self):
            return {}

    monkeypatch.setattr("ui.pages.export_page.get_pipeline_config", lambda: _PC())
    monkeypatch.setattr(
        "ui.pages.export_page.run_id_from_latest_unified_snapshot",
        lambda paths, quiet=True: "rid_latest",
    )

    page = ExportPage()
    qtbot.addWidget(page)
    assert page.architecture.text() == "resnet34"
    assert page.data_dir.text() == "data"
    assert page.image_size.value() == 320
    assert page.run_id.text() == "rid_latest"


@pytest.mark.ui
def test_export_page_restore_empty_state_keeps_prefill(
    monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    class _PC:
        def get(self, key: str, default=None):
            vals = {
                "model.default_architecture": "resnet50",
                "data.data_dir": "data_prefill",
                "data.raw_data_dir": "raw_prefill",
                "data.image_size": 256,
            }
            return vals.get(key, default)

        def to_dict(self):
            return {}

    monkeypatch.setattr("ui.pages.export_page.get_pipeline_config", lambda: _PC())
    monkeypatch.setattr(
        "ui.pages.export_page.run_id_from_latest_unified_snapshot",
        lambda paths, quiet=True: "rid_prefill",
    )
    page = ExportPage()
    qtbot.addWidget(page)
    page.restore_gui_state(
        {
            "input_model": "",
            "architecture": "",
            "data_dir": "",
            "run_id": "",
            "num_classes": 0,
            "image_size": 0,
        }
    )
    assert page.architecture.text() == "resnet50"
    assert page.data_dir.text() == "data_prefill"
    assert page.run_id.text() == "rid_prefill"


@pytest.mark.ui
def test_export_page_input_model_syncs_from_convert_page_when_empty(
    monkeypatch: pytest.MonkeyPatch, qtbot, tmp_path: Path
) -> None:
    class _PC:
        def get(self, key: str, default=None):
            return default

        def to_dict(self):
            return {}

    monkeypatch.setattr("ui.pages.export_page.get_pipeline_config", lambda: _PC())
    monkeypatch.setattr(
        "ui.pages.export_page.run_id_from_latest_unified_snapshot",
        lambda paths, quiet=True: None,
    )
    page = ExportPage()
    qtbot.addWidget(page)
    model_path = str(tmp_path / "from_convert_input.pth")

    class _ConvertLike:
        def __init__(self, txt: str) -> None:
            self.input_model = QLineEdit(txt)

    class _WindowLike:
        def __init__(self, pages) -> None:
            self.page_widgets = pages

    fake_window = _WindowLike([_ConvertLike(model_path)])
    monkeypatch.setattr(page, "window", lambda: fake_window)
    page.input_model.setText("")
    page._sync_input_model_from_convert_page_if_empty()
    assert page.input_model.text() == model_path

