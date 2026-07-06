"""Train page: validation path without running training."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from mb.models.generation_architectures import BaseGenerationArchitecture
from mb.models.types import ArchitectureType, ModelType
from mb.training.lora_diffusion_trainer import LoraTrainingConfig
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
    page.architecture.setText(ArchitectureType.RESNET18.value)
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
    page.architecture.setText(ArchitectureType.RESNET18.value)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert not page.btn_start.isEnabled()
    assert "invalid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_train_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.data_dir.setText(str(tmp_path / "d"))
    page.architecture.setText(ArchitectureType.EFFICIENTNET_B0.value)
    page.frozen_epochs.setValue(2)
    blob = page.collect_gui_state()
    page2 = TrainPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.data_dir.text() == str(tmp_path / "d")
    assert page2.architecture.text() == ArchitectureType.EFFICIENTNET_B0.value
    assert page2.frozen_epochs.value() == 2


@pytest.mark.ui
def test_train_page_model_type_combo_includes_lora(qtbot) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    values = [page.model_type.itemText(i) for i in range(page.model_type.count())]
    assert ModelType.IMAGE_GENERATION_LORA.value in values


@pytest.mark.ui
def test_train_page_lora_model_type_toggles_field_visibility(qtbot, english_gui_locale) -> None:
    """Selecting the LoRA model type must actually hide the classification-only rows.

    This is the scenario that previously hit a ``QFormLayout.setRowVisible: Invalid
    widget`` warning because the row-visibility toggle was keyed off the wrong widget
    object — the warning is silent (no exception), so only checking the resulting
    visibility state catches a regression.
    """
    page = TrainPage()
    qtbot.addWidget(page)
    page.show()

    assert page._hp_group.isVisible()
    assert not page._lora_group.isVisible()
    assert page.resume_from.isVisible()
    assert page.run_id.isVisible()
    assert page.skip_snapshot.isVisible()
    assert page.train_subprocess.isVisible()

    lora_ix = page.model_type.findText(ModelType.IMAGE_GENERATION_LORA.value)
    assert lora_ix >= 0
    page.model_type.setCurrentIndex(lora_ix)

    assert not page._hp_group.isVisible()
    assert page._lora_group.isVisible()
    assert not page.resume_from.isVisible()
    assert not page.run_id.isVisible()
    assert not page.skip_snapshot.isVisible()
    assert not page.train_subprocess.isVisible()

    classification_ix = page.model_type.findText(ModelType.IMAGE_CLASSIFICATION.value)
    page.model_type.setCurrentIndex(classification_ix)

    assert page._hp_group.isVisible()
    assert not page._lora_group.isVisible()
    assert page.resume_from.isVisible()
    assert page.run_id.isVisible()
    assert page.skip_snapshot.isVisible()
    assert page.train_subprocess.isVisible()


@pytest.mark.ui
def test_train_page_lora_validate_enables_start_when_inputs_valid(
    qtbot, english_gui_locale, tmp_path
) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.model_type.setCurrentIndex(page.model_type.findText(ModelType.IMAGE_GENERATION_LORA.value))
    page.framework.setCurrentText("pytorch")
    page.architecture.setText("some-base-model")
    page.data_dir.setText(str(tmp_path))
    arch_ix = page.base_model_architecture.findData(BaseGenerationArchitecture.STABLE_DIFFUSION_1X.value)
    page.base_model_architecture.setCurrentIndex(arch_ix)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert page.btn_start.isEnabled()
    assert "look valid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_train_page_lora_validate_disables_start_when_data_dir_missing(
    qtbot, english_gui_locale, tmp_path
) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.model_type.setCurrentIndex(page.model_type.findText(ModelType.IMAGE_GENERATION_LORA.value))
    page.framework.setCurrentText("pytorch")
    page.architecture.setText("some-base-model")
    page.data_dir.setText(str(tmp_path / "missing_data"))
    arch_ix = page.base_model_architecture.findData(BaseGenerationArchitecture.STABLE_DIFFUSION_1X.value)
    page.base_model_architecture.setCurrentIndex(arch_ix)
    qtbot.mouseClick(page.btn_validate, Qt.MouseButton.LeftButton)
    assert not page.btn_start.isEnabled()
    assert "invalid" in page.output.toPlainText().lower()


@pytest.mark.ui
def test_train_page_lora_collect_inputs_builds_lora_config(qtbot, tmp_path) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.model_type.setCurrentIndex(page.model_type.findText(ModelType.IMAGE_GENERATION_LORA.value))
    page.framework.setCurrentText("pytorch")
    page.architecture.setText("some-base-model")
    page.data_dir.setText(str(tmp_path))
    arch_ix = page.base_model_architecture.findData(BaseGenerationArchitecture.STABLE_DIFFUSION_1X.value)
    page.base_model_architecture.setCurrentIndex(arch_ix)
    request = page._collect_inputs()
    assert isinstance(request, LoraTrainingConfig)
    assert request.base_model == "some-base-model"
    assert request.base_architecture == BaseGenerationArchitecture.STABLE_DIFFUSION_1X


@pytest.mark.ui
def test_train_page_lora_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path) -> None:
    page = TrainPage()
    qtbot.addWidget(page)
    page.model_type.setCurrentIndex(page.model_type.findText(ModelType.IMAGE_GENERATION_LORA.value))
    page.architecture.setText("some-base-model")
    arch_ix = page.base_model_architecture.findData(BaseGenerationArchitecture.FLUX.value)
    page.base_model_architecture.setCurrentIndex(arch_ix)
    page.lora_rank.setValue(8)
    page.lora_alpha.setValue(32)
    page.learning_rate.setValue(5e-5)
    page.max_train_steps.setValue(500)
    page.seed.setText("123")
    blob = page.collect_gui_state()

    page2 = TrainPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.model_type.currentText() == ModelType.IMAGE_GENERATION_LORA.value
    assert page2.base_model_architecture.currentData() == BaseGenerationArchitecture.FLUX.value
    assert page2.lora_rank.value() == 8
    assert page2.lora_alpha.value() == 32
    assert page2.learning_rate.value() == pytest.approx(5e-5)
    assert page2.max_train_steps.value() == 500
    assert page2.seed.text() == "123"
