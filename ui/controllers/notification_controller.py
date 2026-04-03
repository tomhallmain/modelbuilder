"""
NotificationController — toasts, title notifications, alerts.

Thread-safe toast/title via Qt signals. Adapted for Model Builder ``MainWindow``.
Shares base title state with :mod:`utils.notification_manager` for any code that
queues notifications via the global ``notification_manager``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from ui.app_theme import AppStyle
from ui.lib.qt_alert import qt_alert
from utils.config import get_application_config
from utils.logging_setup import get_logger
from utils.notification_manager import notification_manager

logger = get_logger("ui.controllers.notification_controller")


class _NotificationSignals(QObject):
    toast_requested = Signal(str, int, str)
    title_notify_requested = Signal(str, str, int)


class NotificationController:
    """Toast, temporary title changes, and modal alerts — safe to call from worker threads."""

    def __init__(self, main_window: QWidget):
        self._app = main_window
        self._signals = _NotificationSignals()
        self._signals.toast_requested.connect(self._do_toast)
        self._signals.title_notify_requested.connect(self._do_title_notify)
        self._status_title_override_active = False
        wid = getattr(main_window, "window_id", 0)
        notification_manager.set_current_title(
            main_window.windowTitle() or "",
            window_id=wid,
        )

    def toast(
        self,
        message: str,
        time_in_seconds: int | None = None,
        bg_color: Optional[str] = None,
    ) -> None:
        if time_in_seconds is None:
            time_in_seconds = get_application_config().gui.toasts_persist_seconds
        logger.info("Toast: %s", message.replace("\n", " "))
        if not get_application_config().gui.show_toasts:
            return
        color = bg_color or AppStyle.BG_COLOR
        self._signals.toast_requested.emit(message, time_in_seconds, color)

    def _do_toast(self, message: str, time_in_seconds: int, bg_color: str) -> None:
        parent = self._app
        width = 300
        height = 100
        parent_geo = parent.geometry()
        x = parent_geo.x() + parent_geo.width() - width
        y = parent_geo.y()

        previously_active = QApplication.activeWindow()

        toast_widget = QWidget(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        toast_widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        toast_widget.setFixedSize(width, height)
        toast_widget.move(x, y)
        toast_widget.setStyleSheet(
            f"background-color: {bg_color}; border: 1px solid {AppStyle.FG_COLOR};"
        )

        layout = QVBoxLayout(toast_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        label = QLabel(message.strip())
        label.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 10pt; border: none;")
        label.setWordWrap(True)
        layout.addWidget(label)

        toast_widget.show()

        should_restore_owned_secondary = (
            previously_active is not None
            and previously_active is not toast_widget
            and previously_active is not self._app
            and previously_active.isVisible()
            and (
                previously_active.parent() is self._app
                or self._app.isAncestorOf(previously_active)
            )
        )
        if should_restore_owned_secondary:
            try:
                previously_active.activateWindow()
            except Exception:
                pass

        QTimer.singleShot(
            time_in_seconds * 1000,
            lambda: toast_widget.close() if toast_widget else None,
        )

    def title_notify(
        self,
        message: str,
        base_message: str = "",
        time_in_seconds: int = 0,
        **kwargs,
    ) -> None:
        if not get_application_config().gui.show_toasts:
            return
        if time_in_seconds == 0:
            time_in_seconds = get_application_config().gui.title_notify_persist_seconds
        base = base_message or self._app.get_title_from_base_dir()
        self._signals.title_notify_requested.emit(message, base, time_in_seconds)

    def _do_title_notify(self, message: str, base_message: str, time_in_seconds: int) -> None:
        self._app.setWindowTitle(message)
        QTimer.singleShot(
            time_in_seconds * 1000,
            lambda: self._app.setWindowTitle(base_message),
        )

    def set_status_title(self, message: Optional[str]) -> None:
        if message and message.strip() != "":
            base_title = self._app.get_title_from_base_dir()
            self._app.setWindowTitle(f"{base_title} — {message}")
            self._status_title_override_active = True
            return
        if self._status_title_override_active:
            self._app.setWindowTitle(self._app.get_title_from_base_dir())
            self._status_title_override_active = False

    def alert(
        self,
        title: str,
        message: str,
        kind: str = "info",
        severity: str = "normal",
        master: Optional[QWidget] = None,
    ) -> bool:
        logger.warning('Alert - Title: "%s" Message: %s', title, message)
        parent = master or self._app

        if severity == "high" and kind == "askokcancel":
            from ui.lib.custom_dialogs_qt import show_high_severity_dialog

            return show_high_severity_dialog(parent, title, message)

        result = qt_alert(parent, title, message, kind=kind)
        return True if result is None else bool(result)

    def handle_error(self, error_text: str, title: Optional[str] = None, kind: str = "error") -> None:
        import traceback

        traceback.print_exc()
        t = title or "Error"
        self.alert(t, error_text, kind=kind)

    def set_label_state(self, *args, **kwargs) -> None:
        """Not used in Model Builder (no sidebar state label); kept for API compatibility."""
        logger.debug("set_label_state ignored in Model Builder UI")
