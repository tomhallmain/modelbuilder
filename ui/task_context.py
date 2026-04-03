"""
Execution context for GUI background tasks (cancellation + progress reporting).

Used with :func:`ui.task_runner.start_task` when ``pass_context=True``.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional


class TaskCancelled(Exception):
    """Raised when :meth:`LongTaskContext.raise_if_cancelled` sees a user cancel."""

    pass


class LongTaskContext:
    """Passed as the first argument to worker functions when ``pass_context=True``."""

    def __init__(
        self,
        cancel_event: threading.Event,
        emit_progress: Callable[[str, object, bool], None],
    ) -> None:
        self._cancel_event = cancel_event
        self._emit_progress = emit_progress

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TaskCancelled()

    def progress(self, message: str, percent: Optional[float] = None) -> None:
        """
        Report status to the GUI thread via ``TaskSignals.progress``.

        *percent* is 0.0–1.0 when known, or ``None`` for indeterminate mode.
        """
        indeterminate = percent is None
        self._emit_progress(message, percent, indeterminate)
