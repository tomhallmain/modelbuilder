"""
ContextMenuBuilder — image/media context menus from the original app are **not** used
in Model Builder. Stub for import compatibility.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint

from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ContextMenuBuilder:
    """No-op placeholder."""

    def __init__(self, app_window: Any) -> None:
        self._app = app_window

    def show(self, global_pos: QPoint) -> None:
        logger.debug("ContextMenuBuilder.show is not implemented in Model Builder UI.")
