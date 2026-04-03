"""Unit tests for :func:`mb.utils.translations.normalize_gui_locale`."""

from mb.utils.translations import SUPPORTED_GUI_LOCALES, normalize_gui_locale


def test_supported_gui_locales_is_en_de() -> None:
    assert SUPPORTED_GUI_LOCALES == ("en", "de")


def test_normalize_gui_locale() -> None:
    assert normalize_gui_locale(None) is None
    assert normalize_gui_locale("") is None
    assert normalize_gui_locale("en") == "en"
    assert normalize_gui_locale("de") == "de"
    assert normalize_gui_locale("de_DE") == "de"
    assert normalize_gui_locale("en_US") == "en"
    assert normalize_gui_locale("fr") is None
