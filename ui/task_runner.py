"""Small helper to run blocking tasks in background threads."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class TaskSignals(QObject):
    """Signals emitted by a background task."""

    success = Signal(object)
    error = Signal(str)
    done = Signal()


class TaskRunnable(QRunnable):
    """QRunnable wrapper that executes a Python callable."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.success.emit(result)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.signals.error.emit(str(exc))
        finally:
            self.signals.done.emit()


def start_task(
    fn: Callable[..., Any],
    on_success: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_done: Callable[[], None] | None = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Run a callable on the global thread pool and bind result callbacks."""
    runnable = TaskRunnable(fn, *args, **kwargs)
    runnable.signals.success.connect(on_success)
    runnable.signals.error.connect(on_error)
    if on_done:
        runnable.signals.done.connect(on_done)
    QThreadPool.globalInstance().start(runnable)
