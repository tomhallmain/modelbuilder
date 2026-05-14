"""
CacheController -- persistence: loading and storing the app info cache.

Extracted from: load_info_cache, store_info_cache, apply_cached_display_position,
do_periodic_store_cache.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QTimer

from utils.app_info_cache import app_info_cache
from utils.config import get_application_config
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ui.main_window import MainWindow

logger = get_logger("cache_controller")

GUI_FORM_STATE_META_KEY = "gui_form_state"

# Order matches :attr:`MainWindow.NAV_PAGE_SPECS` (page widget list).
_PAGE_KEYS = (
    "home",
    ModelBuilderTaskType.DATA.value,
    ModelBuilderTaskType.TRAIN.value,
    ModelBuilderTaskType.CONVERT.value,
    ModelBuilderTaskType.EXPORT.value,
    ModelBuilderTaskType.EVALUATE.value,
    "config",
    "pipeline",
    ModelBuilderTaskType.INFO.value,
)


class CacheController:
    """
    Owns persistence: loading and storing the application info cache.
    Also owns the periodic cache-store timer.
    """

    def __init__(self, app_window: MainWindow) -> None:
        self._app = app_window
        self._store_cache_timer: Optional[QTimer] = None
        self._persist_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_info_cache(self) -> Optional[str]:
        """
        Load cached application state. Returns the cached base directory
        if one exists, or None.

        Model Builder: applies GUI form state from meta (sidebar + pages).
        """
        try:
            raw = app_info_cache.get(GUI_FORM_STATE_META_KEY, default_val=None)
            if isinstance(raw, dict):
                nav = raw.get("nav_index")
                nav_w = self._app.nav_widget
                if isinstance(nav, int) and 0 <= nav < nav_w.count():
                    nav_w.setCurrentRow(nav)

                pages = self._app.page_widgets
                for i, key in enumerate(_PAGE_KEYS):
                    if i >= len(pages):
                        break
                    chunk = raw.get(key)
                    if not isinstance(chunk, dict) or not chunk:
                        continue
                    restore = getattr(pages[i], "restore_gui_state", None)
                    if callable(restore):
                        restore(chunk)
        except Exception as e:
            logger.warning("Error applying cached GUI state: %s", e)

        return app_info_cache.get("base_dir")

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------
    def store_info_cache(self, *, sync: bool = False) -> None:
        """
        Persist current application state to the info cache.

        Model Builder: writes GUI form state to meta, then ``store()``.

        By default ``store()`` runs in a **background thread** so the first encrypted write
        (OQS key generation, keyring) does not freeze the Qt GUI. Use ``sync=True`` when the
        process is exiting and a best-effort flush must complete on the calling thread.
        """
        logger.info("Storing app info cache")

        try:
            blob: dict = {"nav_index": int(self._app.nav_widget.currentRow())}
            pages = self._app.page_widgets
            for i, key in enumerate(_PAGE_KEYS):
                if i >= len(pages):
                    break
                collect = getattr(pages[i], "collect_gui_state", None)
                if not callable(collect):
                    continue
                try:
                    data = collect()
                except Exception as e:
                    logger.debug("collect_gui_state skipped for %s: %s", key, e)
                    continue
                if isinstance(data, dict) and data:
                    blob[key] = data
            app_info_cache.set(GUI_FORM_STATE_META_KEY, blob)
        except Exception as e:
            logger.warning("Failed to merge GUI form state into cache: %s", e)

        if sync:
            try:
                app_info_cache.store()
            except Exception as e:
                logger.warning("Failed to persist app info cache (sync): %s", e)
            return

        if not self._persist_lock.acquire(blocking=False):
            return

        def _persist() -> None:
            try:
                try:
                    app_info_cache.store()
                except Exception as e:
                    logger.warning("Failed to persist app info cache: %s", e)
            finally:
                self._persist_lock.release()

        threading.Thread(
            target=_persist, daemon=True, name="app_info_cache_persist"
        ).start()

    # ------------------------------------------------------------------
    # Display position
    # ------------------------------------------------------------------
    def apply_cached_display_position(self) -> bool:
        """
        Restore the window geometry from the cached display position.
        Returns True if a position was applied.

        Not used in Model Builder (no multi-window / saved geometry path).
        """
        return False

    # ------------------------------------------------------------------
    # Periodic cache store (replaces start_thread + async do_periodic_store_cache)
    # ------------------------------------------------------------------
    def start_periodic_store(self) -> None:
        """
        Start a periodic timer to store the cache at intervals.

        Replaces the async ``do_periodic_store_cache`` coroutine.
        """
        interval_ms = int(get_application_config().gui.cache_store_interval_seconds * 1000)
        if interval_ms <= 0:
            return

        self._store_cache_timer = QTimer()
        self._store_cache_timer.timeout.connect(self._on_periodic_store)
        self._store_cache_timer.start(interval_ms)

    def restart_periodic_store(self) -> None:
        """Stop and start the periodic timer (e.g. after YAML config path changes)."""
        self.stop_periodic_store()
        self.start_periodic_store()

    def stop_periodic_store(self) -> None:
        if self._store_cache_timer is not None:
            self._store_cache_timer.stop()
            self._store_cache_timer = None

    def _on_periodic_store(self) -> None:
        """
        Called on the main thread by QTimer.

        Ported from App.do_periodic_store_cache + App._store_info_cache_main_thread.
        """
        try:
            self.store_info_cache()
        except Exception as e:
            logger.debug("Error in periodic store info cache: %s", e)
