"""
Typed callback bundle for wiring pages and workers to main-window affordances.

Maps action names to callables (toast, alert, title, etc.). Pass ``master`` as the
``MainWindow`` for modal dialogs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ui.app_theme import AppStyle
from utils.config import get_application_config


class AppActions:
    """Required keys must be present in the ``actions`` dict passed to ``__init__``."""

    REQUIRED_ACTIONS = frozenset(
        {
            "get_window",
            "toast",
            "_alert",
            "title_notify",
            "refresh",
            "title",
        }
    )

    def __init__(
        self,
        actions: Dict[str, Callable[..., Any]],
        master: Optional[object] = None,
    ) -> None:
        missing = self.REQUIRED_ACTIONS - set(actions.keys())
        if missing:
            raise ValueError(f"Missing required actions: {missing}")
        self._actions = dict(actions)
        self._master = master

    def __getattr__(self, name: str) -> Any:
        if name in self._actions:
            return self._actions[name]
        raise AttributeError(f"Action '{name}' not found")

    def alert(
        self,
        title: str,
        message: str,
        kind: str = "info",
        severity: str = "normal",
        master: Optional[object] = None,
    ) -> Any:
        parent_window = master if master is not None else self._master
        return self._actions["_alert"](title, message, kind=kind, severity=severity, master=parent_window)

    def warn(self, message: str, time_in_seconds: Optional[int] = None) -> None:
        if time_in_seconds is None:
            time_in_seconds = get_application_config().gui.toasts_persist_seconds
        return self.toast(message, time_in_seconds=time_in_seconds, bg_color=AppStyle.TOAST_COLOR_WARNING)

    def success(self, message: str, time_in_seconds: Optional[int] = None) -> None:
        if time_in_seconds is None:
            time_in_seconds = get_application_config().gui.toasts_persist_seconds
        return self.toast(message, time_in_seconds=time_in_seconds, bg_color=AppStyle.TOAST_COLOR_SUCCESS)

    def get_master(self) -> Optional[object]:
        return self._master

    def is_fullscreen(self) -> bool:
        """Used by ``notification_manager`` when choosing toast vs title; Model Builder uses title updates."""
        return False
