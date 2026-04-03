"""Train page: validation path without running training."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from ui.pages.train_page import TrainPage


@pytest.mark.ui
def test_train_page_widgets_expose_stable_object_names(qtbot) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    assert page.objectName() == "train_page"
    assert page.data_dir.objectName() == "train_data_dir_edit"
    assert page.output.objectName() == "train_output_log"
    assert page.findChild(QPushButton, "train_validate_btn") is page.btn_validate
    assert page.findChild(QPushButton, "train_start_btn") is page.btn_start


@pytest.mark.ui
def test_train_page_validate_enables_start_when_layout_valid(
    qtbot, english_gui_locale, two_class_classification_data_dir, tmp_path
) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.data_dir.setText(str(two_class_classification_data_dir))
    page.output_dir.setText(str(tmp_path / "models"))
    page.architecture.setText("resnet18")
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_start.isEnabled()
    assert "look valid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_train_page_validate_disables_start_when_data_dir_missing(
    qtbot, english_gui_locale, tmp_path
) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.data_dir.setText(str(tmp_path / "nonexistent_data"))
    page.architecture.setText("resnet18")
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert not page.btn_start.isEnabled()
    assert "invalid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_train_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.data_dir.setText(str(tmp_path / "d"))
    page.architecture.setText("efficientnet_b0")
    page.frozen_epochs.setValue(2)
    blob = page.collect_gui_state()
    page2 = TrainPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.data_dir.text() == str(tmp_path / "d")
    assert page2.architecture.text() == "efficientnet_b0"
    assert page2.frozen_epochs.value() == 2
