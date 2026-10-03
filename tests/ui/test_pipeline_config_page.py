"""Pipeline page: form labels line up with their rows, and data keys survive a save."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QFormLayout

from mb.models.types import LabelMode
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
