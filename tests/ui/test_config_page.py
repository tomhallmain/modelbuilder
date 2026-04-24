"""Headless tests for :class:`ui.pages.config_page.ConfigPage`."""

from __future__ import annotations

import yaml
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QPushButton

from ui.main_window import MainWindow
from ui.pages import ConfigPage


def _config_page(win: MainWindow) -> ConfigPage:
    for page in win.page_widgets:
        if isinstance(page, ConfigPage):
            return page
    raise AssertionError("ConfigPage not found in MainWindow.page_widgets")


@pytest.mark.ui
def test_config_page_widget_identity_and_object_name(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    assert page.objectName() == "config_page"


@pytest.mark.ui
def test_config_save_and_default_buttons_have_object_names(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    assert page.findChild(QPushButton, "config_save_btn") is page._btn_save
    assert page.findChild(QPushButton, "config_set_default_btn") is page._btn_default


@pytest.mark.ui
def test_config_locale_combo_only_supported_codes(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    combo = page.findChild(QComboBox, "config_locale_combo")
    assert combo is not None
    assert combo.count() == 3
    datas = [combo.itemData(i, Qt.ItemDataRole.UserRole) for i in range(combo.count())]
    assert datas == ["", "en", "de"]


@pytest.mark.ui
def test_retranslate_shell_ui_runs(qtbot, main_window: MainWindow) -> None:
    main_window.retranslate_shell_ui()


@pytest.mark.ui
def test_gather_from_form_locale_and_debug_rules(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    page._locale_combo.setCurrentIndex(1)
    page._debug.setChecked(False)
    page._debug2.setChecked(True)
    page._print_settings.setChecked(False)
    d = page._gather_from_form()
    assert d["gui"]["locale"] == "en"
    assert d["app"]["debug2"] is True
    assert d["app"]["debug"] is True
    assert d["app"]["print_settings"] is False


@pytest.mark.ui
def test_gather_from_form_locale_none_when_system_default(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    page._locale_combo.setCurrentIndex(0)
    d = page._gather_from_form()
    assert d["gui"]["locale"] is None


@pytest.mark.ui
def test_apply_dict_to_form_maps_supported_locale(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    page._apply_dict_to_form({"gui": {"locale": "de"}, "app": {}})
    assert page._locale_combo.currentIndex() == 2


@pytest.mark.ui
def test_apply_dict_to_form_unsupported_locale_maps_to_system_default(
    qtbot, main_window: MainWindow
) -> None:
    page = _config_page(main_window)
    page._locale_combo.setCurrentIndex(2)
    page._apply_dict_to_form({"gui": {"locale": "fr"}, "app": {}})
    assert page._locale_combo.currentIndex() == 0


@pytest.mark.ui
def test_apply_dict_to_form_de_de_maps_to_de(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    page._apply_dict_to_form({"gui": {"locale": "de_AT"}, "app": {}})
    assert page._locale_combo.currentIndex() == 2


@pytest.mark.ui
def test_save_rejects_invalid_main_window_size_without_writing_file(
    qtbot, main_window: MainWindow, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "should_not_exist.yaml"
    monkeypatch.setattr(
        "ui.pages.config_page.resolve_application_save_path",
        lambda: target,
    )
    alerts: list[tuple[object, ...]] = []

    def capture_alert(*args: object, **kwargs: object) -> None:
        alerts.append(args)

    monkeypatch.setattr("ui.pages.config_page.qt_alert", capture_alert)
    page = _config_page(main_window)
    page._main_size.setText("not-a-size")
    page._on_save()
    assert not target.exists()
    assert len(alerts) == 1


@pytest.mark.ui
def test_save_writes_yaml_with_expected_gui_keys(
    qtbot, main_window: MainWindow, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "application_saved.yaml"
    monkeypatch.setattr(
        "ui.pages.config_page.resolve_application_save_path",
        lambda: target,
    )
    monkeypatch.setattr("ui.pages.config_page.qt_alert", lambda *a, **k: None)
    page = _config_page(main_window)
    page._main_size.setText("900x700")
    page._locale_combo.setCurrentIndex(2)
    page._toast_sec.setValue(3)
    page._on_save()
    assert target.is_file()
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["gui"]["locale"] == "de"
    assert data["gui"]["default_main_window_size"] == "900x700"
    assert data["gui"]["toasts_persist_seconds"] == 3
    assert "app" in data


@pytest.mark.ui
def test_reload_from_disk_button_triggers_refresh(
    qtbot, main_window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _config_page(main_window)
    calls: list[int] = []
    real_refresh = page._refresh_from_disk

    def spy_refresh() -> None:
        calls.append(1)
        return real_refresh()

    monkeypatch.setattr(page, "_refresh_from_disk", spy_refresh)
    qtbot.mouseClick(page._btn_reload, Qt.MouseButton.LeftButton)
    assert sum(calls) == 1


@pytest.mark.ui
def test_set_default_writes_user_application_yaml(
    qtbot, main_window: MainWindow, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_yaml = tmp_path / "appdata" / "application.yaml"
    monkeypatch.setattr(
        "ui.pages.config_page.get_user_application_config_path",
        lambda: user_yaml,
    )
    monkeypatch.setattr("ui.pages.config_page.qt_alert", lambda *a, **k: None)
    page = _config_page(main_window)
    page._on_set_default()
    assert user_yaml.is_file()
    data = yaml.safe_load(user_yaml.read_text(encoding="utf-8"))
    assert "gui" in data and "app" in data


@pytest.mark.ui
def test_page_title_non_empty_after_construct(qtbot, main_window: MainWindow) -> None:
    page = _config_page(main_window)
    assert len(page._page_title.text().strip()) > 0
