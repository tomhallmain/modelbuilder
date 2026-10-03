"""Text page: construction and form-state round trip (no long-running work)."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton, QTabWidget

from mb.models.types import TextSubcommand
from ui.pages.text_page import TextPage


@pytest.mark.ui
def test_text_page_has_a_tab_per_subcommand(qtbot) -> None:
    page = TextPage()
    qtbot.addWidget(page)
    assert page.objectName() == "text_page"
    assert page.findChild(QTabWidget).count() == len(tuple(TextSubcommand))
    assert page.findChild(QPushButton, "text_verify_btn") is page.btn_verify


@pytest.mark.ui
def test_text_page_collect_and_restore_gui_state_roundtrip(qtbot, tmp_path) -> None:
    page = TextPage()
    qtbot.addWidget(page)
    page.verify_data_dir.setText(str(tmp_path / "ds"))
    page.score_input.setText(str(tmp_path / "in.txt"))
    page.compare_runs.setPlainText("runs/*")
    blob = page.collect_gui_state()

    page2 = TextPage()
    qtbot.addWidget(page2)
    page2.restore_gui_state(blob)
    assert page2.collect_gui_state() == blob


@pytest.mark.ui
def test_text_page_compare_reports_missing_runs(qtbot, english_gui_locale, tmp_path) -> None:
    from mb.utils.translations import _

    page = TextPage()
    qtbot.addWidget(page)
    page.compare_runs.setPlainText(str(tmp_path / "none*"))
    page.btn_compare.click()
    assert _("No run directories with metrics.json were found.") in page.compare_output.toPlainText()
