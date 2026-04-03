"""Edit application shell settings (``gui`` / ``app``)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mb.pipeline_config import reload_pipeline_config

from ui.lib.qt_alert import qt_alert
from utils.config import (
    default_application_config_dict,
    get_application_config,
    get_user_application_config_path,
    reload_application_config,
    resolve_application_save_path,
)

if TYPE_CHECKING:
    from ui.main_window import MainWindow

_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


class ConfigPage(QWidget):
    """View and save desktop shell settings (toasts, window size, debug, …)."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("config_page")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        title = QLabel("Config")
        tf = QFont(title.font())
        tf.setPointSizeF(tf.pointSizeF() + 3)
        tf.setBold(True)
        title.setFont(tf)
        title.setContentsMargins(0, 0, 0, 0)
        root.addWidget(title)

        desc = QLabel(
            "Application shell settings (<code>gui</code>, <code>app</code>). "
            "Save writes to your active config file, or to the user data folder when "
            "the packaged example is the only source."
        )
        desc.setWordWrap(True)
        desc.setContentsMargins(0, 0, 0, 0)
        root.addWidget(desc)

        self._path_label = QLabel()
        self._path_label.setWordWrap(True)
        self._path_label.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._path_label)

        root.addSpacing(6)
        root.addWidget(self._build_gui_group())
        root.addWidget(self._build_app_group())

        root.addSpacing(4)
        row = QHBoxLayout()
        self._btn_reload = QPushButton("Reload from disk")
        self._btn_reload.clicked.connect(self._refresh_from_disk)
        self._btn_save = QPushButton("Save")
        self._btn_save.setObjectName("config_save_btn")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_default = QPushButton("Set Default…")
        self._btn_default.setObjectName("config_set_default_btn")
        self._btn_default.setToolTip(
            "Replace your user application.yaml (next to the log folder) with built-in defaults. "
            "Does not change workspace-specific YAML."
        )
        self._btn_default.clicked.connect(self._on_set_default)
        row.addWidget(self._btn_reload)
        row.addWidget(self._btn_save)
        row.addWidget(self._btn_default)
        row.addStretch(1)
        root.addLayout(row)

        self._refresh_from_disk()

    @staticmethod
    def _line_edit_avoid_inactive_dimming(edit: QLineEdit) -> None:
        """Scroll-area / unfocused Qt style can paint ``QLineEdit`` text with muted inactive colors."""
        pal = edit.palette()
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text):
            c = pal.color(QPalette.ColorGroup.Active, role)
            pal.setColor(QPalette.ColorGroup.Inactive, role, c)
        edit.setPalette(pal)

    def _apply_line_edit_text_contrast(self) -> None:
        for le in self.findChildren(QLineEdit):
            self._line_edit_avoid_inactive_dimming(le)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_line_edit_text_contrast()

    def _build_gui_group(self) -> QGroupBox:
        box = QGroupBox("gui")
        form = QFormLayout(box)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(14)
        form.setContentsMargins(12, 14, 12, 12)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._locale = QLineEdit()
        self._locale.setPlaceholderText("empty — use OS locale")
        form.addRow("Locale", self._locale)

        self._fg_color = QLineEdit()
        self._fg_color.setPlaceholderText("#ececec")
        form.addRow("Foreground color", self._fg_color)

        self._bg_color = QLineEdit()
        self._bg_color.setPlaceholderText("#1a1a1a")
        form.addRow("Background color", self._bg_color)

        self._toast_warn = QLineEdit()
        self._toast_warn.setPlaceholderText("#5D4037")
        form.addRow("Toast color (warning)", self._toast_warn)

        self._toast_ok = QLineEdit()
        self._toast_ok.setPlaceholderText("#33691E")
        form.addRow("Toast color (success)", self._toast_ok)

        self._accent = QLineEdit()
        self._accent.setPlaceholderText("#F57C00")
        form.addRow("Accent (primary)", self._accent)

        self._accent_sec = QLineEdit()
        self._accent_sec.setPlaceholderText("#FFC107")
        form.addRow("Accent (secondary)", self._accent_sec)

        self._show_toasts = QCheckBox("Show toasts")
        form.addRow("", self._show_toasts)

        self._main_size = QLineEdit()
        self._main_size.setPlaceholderText("e.g. 1200x960")
        form.addRow("Default main window size", self._main_size)

        self._toast_sec = QSpinBox()
        self._toast_sec.setRange(0, 3600)
        form.addRow("Toast duration (seconds)", self._toast_sec)

        self._title_sec = QSpinBox()
        self._title_sec.setRange(0, 3600)
        form.addRow("Title notify duration (seconds)", self._title_sec)

        self._font_size = QSpinBox()
        self._font_size.setRange(6, 72)
        form.addRow("Font size", self._font_size)

        self._cache_interval = QDoubleSpinBox()
        self._cache_interval.setRange(1.0, 86400.0)
        self._cache_interval.setDecimals(1)
        self._cache_interval.setSuffix(" s")
        form.addRow("GUI cache store interval", self._cache_interval)

        return box

    def _build_app_group(self) -> QGroupBox:
        box = QGroupBox("app")
        form = QFormLayout(box)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(14)
        form.setContentsMargins(12, 14, 12, 12)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._debug = QCheckBox("Debug logging")
        form.addRow("", self._debug)

        self._debug2 = QCheckBox("Debug logging (verbose)")
        form.addRow("", self._debug2)

        self._log_level = QComboBox()
        self._log_level.addItems(["debug", "info", "warning", "error", "critical"])
        self._log_level.setEditable(True)
        form.addRow("Log level", self._log_level)

        self._print_settings = QCheckBox("Print settings to log on load")
        form.addRow("", self._print_settings)

        return box

    def collect_gui_state(self) -> dict:
        return {}

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return

    def _main_window(self) -> MainWindow | None:
        from ui.main_window import MainWindow

        w = self.window()
        return w if isinstance(w, MainWindow) else None

    @staticmethod
    def _line_or_none(edit: QLineEdit) -> Any:
        t = edit.text().strip()
        return None if t == "" else t

    def _apply_dict_to_form(self, data: dict[str, Any]) -> None:
        g = data.get("gui") or {}
        a = data.get("app") or {}

        self._locale.setText("" if g.get("locale") in (None, "") else str(g["locale"]))
        for key, edit in (
            ("foreground_color", self._fg_color),
            ("background_color", self._bg_color),
            ("toast_color_warning", self._toast_warn),
            ("toast_color_success", self._toast_ok),
            ("accent_color", self._accent),
            ("accent_secondary_color", self._accent_sec),
        ):
            v = g.get(key)
            edit.setText("" if v is None else str(v))

        self._show_toasts.setChecked(bool(g.get("show_toasts", True)))
        self._main_size.setText(str(g.get("default_main_window_size", "1200x960")))
        self._toast_sec.setValue(int(g.get("toasts_persist_seconds", 2)))
        self._title_sec.setValue(int(g.get("title_notify_persist_seconds", 5)))
        self._font_size.setValue(int(g.get("font_size", 8)))
        self._cache_interval.setValue(float(g.get("cache_store_interval_seconds", 120.0)))

        self._debug.setChecked(bool(a.get("debug", False)))
        self._debug2.setChecked(bool(a.get("debug2", False)))
        lvl = str(a.get("log_level", "info"))
        i = self._log_level.findText(lvl, Qt.MatchFlag.MatchExactly)
        if i >= 0:
            self._log_level.setCurrentIndex(i)
        else:
            self._log_level.setEditText(lvl)
        self._print_settings.setChecked(bool(a.get("print_settings", True)))

    def _gather_from_form(self) -> dict[str, Any]:
        debug2_on = self._debug2.isChecked()
        debug_on = self._debug.isChecked() or debug2_on
        gui: dict[str, Any] = {
            "locale": self._line_or_none(self._locale),
            "foreground_color": self._line_or_none(self._fg_color),
            "background_color": self._line_or_none(self._bg_color),
            "toast_color_warning": self._line_or_none(self._toast_warn),
            "toast_color_success": self._line_or_none(self._toast_ok),
            "accent_color": self._line_or_none(self._accent),
            "accent_secondary_color": self._line_or_none(self._accent_sec),
            "show_toasts": self._show_toasts.isChecked(),
            "default_main_window_size": self._main_size.text().strip() or "1200x960",
            "toasts_persist_seconds": self._toast_sec.value(),
            "title_notify_persist_seconds": self._title_sec.value(),
            "font_size": self._font_size.value(),
            "cache_store_interval_seconds": self._cache_interval.value(),
        }
        app: dict[str, Any] = {
            "debug": debug_on,
            "debug2": debug2_on,
            "log_level": self._log_level.currentText().strip() or "info",
            "print_settings": self._print_settings.isChecked(),
        }
        return {"gui": gui, "app": app}

    def _refresh_from_disk(self) -> None:
        from utils.config import _resolve_application_yaml_path

        ac = get_application_config()
        path = ac.active_path
        if path is None:
            path = _resolve_application_yaml_path(None)
        if path is not None and path.is_file():
            self._path_label.setText(f"Reading: {path}")
        else:
            self._path_label.setText("Reading: (built-in defaults — no file on disk)")
        self._apply_dict_to_form(ac.to_dict())
        self._apply_line_edit_text_contrast()

    def _on_save(self) -> None:
        size = self._main_size.text().strip()
        if not _SIZE_RE.match(size):
            qt_alert(
                self,
                "Config",
                "Default main window size must look like WIDTHxHEIGHT (e.g. 1200x960).",
            )
            return

        filtered = self._gather_from_form()
        target = resolve_application_save_path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(filtered, f, default_flow_style=False, sort_keys=False)
        except OSError as e:
            qt_alert(self, "Config", f"Could not write:\n{target}\n\n{e}")
            return

        reload_application_config(target, force=True)
        mw = self._main_window()
        if mw is not None:
            reload_pipeline_config(mw._effective_pipeline_config_path(), force=True)
            mw.refresh_application_shell_settings()
        else:
            reload_pipeline_config(None, force=True)
        self._refresh_from_disk()
        qt_alert(self, "Config", f"Saved to:\n{target}")

    def _on_set_default(self) -> None:
        target = get_user_application_config_path()
        data = default_application_config_dict()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except OSError as e:
            qt_alert(self, "Config", f"Could not write defaults to:\n{target}\n\n{e}")
            return
        reload_application_config(target, force=True)
        mw = self._main_window()
        if mw is not None:
            reload_pipeline_config(mw._effective_pipeline_config_path(), force=True)
            mw.refresh_application_shell_settings()
        else:
            reload_pipeline_config(None, force=True)
        self._refresh_from_disk()
        qt_alert(self, "Config", f"Reset user config to defaults:\n{target}")
