"""
Application (desktop shell) configuration: ``gui`` and ``app`` YAML + defaults.

Pipeline / ML job settings (model, data, training, paths) live in
:mod:`mb.pipeline_config` — not here.

Prefer typed accessors, e.g. ``get_application_config().gui.toasts_persist_seconds``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.logging_setup import get_logger
from utils.utils import Utils

logger = get_logger("config")

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Prefer configs/application.yaml; configs/default.yaml may still exist as a legacy combined file.
DEFAULT_APPLICATION_YAML = _REPO_ROOT / "configs" / "application.yaml"
LEGACY_DEFAULT_YAML = _REPO_ROOT / "configs" / "default.yaml"

_APPLICATION_KEYS = frozenset({"gui", "app"})

_global_application: Optional["ApplicationConfig"] = None


class GuiConfig:
    """Typed read-only view of the ``gui:`` subtree."""

    __slots__ = ("_root",)

    def __init__(self, root: "ApplicationConfig") -> None:
        object.__setattr__(self, "_root", root)

    def _get(self, leaf: str, default: Any) -> Any:
        return self._root.get(f"gui.{leaf}", default)

    @property
    def locale(self) -> Optional[str]:
        v = self._get("locale", None)
        return str(v) if v is not None and v != "" else None

    @property
    def foreground_color(self) -> Optional[str]:
        v = self._get("foreground_color", None)
        return str(v) if v is not None else None

    @property
    def background_color(self) -> Optional[str]:
        v = self._get("background_color", None)
        return str(v) if v is not None else None

    @property
    def toast_color_warning(self) -> Optional[str]:
        v = self._get("toast_color_warning", None)
        return str(v) if v is not None else None

    @property
    def toast_color_success(self) -> Optional[str]:
        v = self._get("toast_color_success", None)
        return str(v) if v is not None else None

    @property
    def show_toasts(self) -> bool:
        return bool(self._get("show_toasts", True))

    @property
    def default_main_window_size(self) -> str:
        v = self._get("default_main_window_size", "960x640")
        return str(v) if v is not None else "960x640"

    @property
    def toasts_persist_seconds(self) -> int:
        v = self._get("toasts_persist_seconds", 2)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 2

    @property
    def title_notify_persist_seconds(self) -> int:
        v = self._get("title_notify_persist_seconds", 5)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 5

    @property
    def font_size(self) -> int:
        v = self._get("font_size", 8)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 8

    @property
    def always_open_new_windows(self) -> bool:
        return bool(self._get("always_open_new_windows", False))

    @property
    def enable_videos(self) -> bool:
        return bool(self._get("enable_videos", False))

    @property
    def cache_store_interval_seconds(self) -> float:
        v = self._get("cache_store_interval_seconds", 120.0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 120.0


class AppConfig:
    """Typed read-only view of the ``app:`` subtree."""

    __slots__ = ("_root",)

    def __init__(self, root: "ApplicationConfig") -> None:
        object.__setattr__(self, "_root", root)

    def _get(self, leaf: str, default: Any) -> Any:
        return self._root.get(f"app.{leaf}", default)

    @property
    def debug(self) -> bool:
        return bool(self._get("debug", False))

    @property
    def debug2(self) -> bool:
        return bool(self._get("debug2", False))

    @property
    def log_level(self) -> str:
        v = self._get("log_level", "info")
        return str(v) if v is not None else "info"

    @property
    def print_settings(self) -> bool:
        return bool(self._get("print_settings", True))


class ApplicationConfig:
    """YAML-backed ``gui`` + ``app`` settings merged over built-in defaults."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._active_path: Optional[Path] = None
        self._defaults = self._get_defaults()
        self._config: Dict[str, Any] = self._deep_merge({}, self._defaults)
        self._gui_view: Optional[GuiConfig] = None
        self._app_view: Optional[AppConfig] = None

        if config_path and config_path.exists():
            self.load_from_file(config_path)
        elif config_path:
            logger.warning("Application config file not found: %s, using defaults", config_path)

        self._post_load()

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "gui": {
                "locale": None,
                "foreground_color": None,
                "background_color": None,
                "toast_color_warning": None,
                "toast_color_success": None,
                "show_toasts": True,
                "default_main_window_size": "1400x950",
                "toasts_persist_seconds": 2,
                "title_notify_persist_seconds": 5,
                "font_size": 8,
                "always_open_new_windows": False,
                "enable_videos": False,
                "cache_store_interval_seconds": 120.0,
            },
            "app": {
                "debug": False,
                "debug2": False,
                "log_level": "info",
                "print_settings": True,
            },
        }

    def load_from_file(self, config_path: Path) -> None:
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            filtered = {k: v for k, v in raw.items() if k in _APPLICATION_KEYS}
            self._config = self._deep_merge(self._deep_merge({}, self._defaults), filtered)
            self._active_path = config_path
            logger.info("Loaded application configuration from %s", config_path)
        except Exception:
            logger.exception("Failed to load application config from %s", config_path)
            self._config = self._deep_merge({}, self._defaults)
            self._active_path = None

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value: Any = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        cfg = self._config
        for key in keys[:-1]:
            if key not in cfg:
                cfg[key] = {}
            cfg = cfg[key]
        cfg[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)

    def save_to_file(self, config_path: Path) -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved application configuration to %s", config_path)

    @property
    def active_path(self) -> Optional[Path]:
        return self._active_path

    @property
    def gui(self) -> GuiConfig:
        if self._gui_view is None:
            self._gui_view = GuiConfig(self)
        return self._gui_view

    @property
    def app(self) -> AppConfig:
        if self._app_view is None:
            self._app_view = AppConfig(self)
        return self._app_view

    def _post_load(self) -> None:
        loc = self.gui.locale
        if loc is None or loc == "":
            loc = Utils.get_default_user_language()
            self.set("gui.locale", loc)
        os.environ["LANG"] = str(loc)

        if self.app.print_settings:
            self.print_config_settings()

    def print_config_settings(self) -> None:
        logger.info("Application settings active:")
        extra = "" if self.gui.show_toasts else ": False — NO toasts will be shown!"
        logger.info(" - Show toasts%s", extra)
        if self.app.debug:
            logger.info(" - Debug logging enabled")


def _resolve_application_yaml_path(config_path: Optional[Path]) -> Optional[Path]:
    if config_path is not None:
        return config_path
    if DEFAULT_APPLICATION_YAML.is_file():
        return DEFAULT_APPLICATION_YAML
    if LEGACY_DEFAULT_YAML.is_file():
        return LEGACY_DEFAULT_YAML
    return None


def reload_application_config(config_path: Optional[Path] = None) -> ApplicationConfig:
    """
    Replace the global application config singleton.

    ``config_path`` ``None`` loads ``configs/application.yaml`` when present, else
    ``configs/default.yaml`` (legacy), else in-memory defaults only.
    """
    global _global_application
    path = _resolve_application_yaml_path(config_path)
    _global_application = ApplicationConfig(path)
    return _global_application


def get_application_config(config_path: Optional[Path] = None) -> ApplicationConfig:
    """
    Return the active :class:`ApplicationConfig` singleton.

    - If ``config_path`` is set, always reload from that file.
    - If ``None`` and no config loaded yet, load repo defaults when present.
    - If ``None`` and already loaded, return the cached instance.
    """
    global _global_application
    if config_path is not None:
        reload_application_config(config_path)
    elif _global_application is None:
        reload_application_config(None)
    assert _global_application is not None
    return _global_application


def reset_application_config() -> None:
    global _global_application
    _global_application = None


# Backward-compatible aliases (deprecated naming)
MbConfig = ApplicationConfig


def reload_config(config_path: Optional[Path] = None) -> ApplicationConfig:
    """Alias for :func:`reload_application_config`."""
    return reload_application_config(config_path)


def get_config(config_path: Optional[Path] = None) -> ApplicationConfig:
    """Alias for :func:`get_application_config`."""
    return get_application_config(config_path)


def reset_config() -> None:
    """Alias for :func:`reset_application_config`."""
    reset_application_config()
