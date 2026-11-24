"""Pipeline page: form labels line up with their rows, and data keys survive a save."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QFormLayout

from mb.models.types import LabelMode, MixedPrecisionMode, OptimizerType
from ui.pages.pipeline_config_page import PipelineConfigPage


def _row_labels(form: QFormLayout) -> list[str]:
    out = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        out.append(item.widget().text() if item is not None and item.widget() is not None else "")
    return out


@pytest.mark.ui
def test_data_form_labels_match_rows_after_retranslate(qtbot, english_gui_locale) -> None:
    page = PipelineConfigPage()
    qtbot.addWidget(page)
    page.retranslate_ui()
    assert _row_labels(page._data_form) == page._data_form_labels()
    assert page._data_form.labelForField(page._d_test_split_mode).text() == page._data_form_labels()[3]


@pytest.mark.ui
def test_label_mode_round_trips_through_form(qtbot) -> None:
    page = PipelineConfigPage()
    qtbot.addWidget(page)
    page._apply_dict_to_form({"data": {"label_mode": LabelMode.MULTI_LABEL.value}})
    assert page._full_pipeline_dict_from_form()["data"]["label_mode"] == LabelMode.MULTI_LABEL.value
    page._apply_dict_to_form({"data": {}})
    assert page._full_pipeline_dict_from_form()["data"]["label_mode"] == LabelMode.get_default().value


@pytest.mark.ui
def test_optimization_keys_round_trip_through_form(qtbot) -> None:
    """Every training key the trainer reads must survive a save from this page."""
    page = PipelineConfigPage()
    qtbot.addWidget(page)
    values = {
        "optimizer": OptimizerType.ADAMW.value,
        "weight_decay": 0.05,
        "layer_decay": 0.75,
        "grad_accum_steps": 4,
        "mixed_precision": MixedPrecisionMode.BF16.value,
        "gradient_checkpointing": True,
        "restore_best": True,
    }
    page._apply_dict_to_form({"training": dict(values)})
    saved = page._full_pipeline_dict_from_form()["training"]
    for key, value in values.items():
        if isinstance(value, float):
            assert saved[key] == pytest.approx(value), key
        else:
            assert saved[key] == value, key


@pytest.mark.ui
def test_optimization_defaults_when_keys_absent(qtbot) -> None:
    """An older pipeline file without these keys saves the original-behavior defaults."""
    page = PipelineConfigPage()
    qtbot.addWidget(page)
    page._apply_dict_to_form({"training": {}})
    saved = page._full_pipeline_dict_from_form()["training"]
    assert saved["optimizer"] == OptimizerType.get_default().value
    assert saved["weight_decay"] == 0.0
    assert saved["layer_decay"] == 1.0
    assert saved["grad_accum_steps"] == 1
    assert saved["mixed_precision"] == MixedPrecisionMode.get_default().value
    assert saved["gradient_checkpointing"] is False
    assert saved["restore_best"] is False


@pytest.mark.ui
def test_training_form_labels_match_rows_after_retranslate(qtbot, english_gui_locale) -> None:
    page = PipelineConfigPage()
    qtbot.addWidget(page)
    page.retranslate_ui()
    labels = _row_labels(page._training_form)
    assert page._training_form.labelForField(page._t_restore_best).text() == labels[-2]
    assert labels[-1] == ""
