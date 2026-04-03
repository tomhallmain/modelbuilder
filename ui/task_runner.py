"""Small helper to run blocking tasks in background threads.

Callbacks connected to :class:`TaskSignals` run on the **GUI thread**: the
``TaskSignals`` instance is created in the thread that calls :func:`start_task`
(typically main), and connections use :attr:`Qt.ConnectionType.QueuedConnection`
so slots fire in the receiver's thread when the worker emits from the pool.

Use ``pass_context=True`` to receive a :class:`~ui.task_context.LongTaskContext`
(cancel + progress) as the first argument to the worker callable. Cooperative
cooperative cancel in ``mb`` raises :class:`mb.cancellation.OperationCancelled`
(subclass :class:`~mb.cancellation.TrainingCancelled` for training), which maps
to :attr:`TaskSignals.cancelled`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, QRunnable, QThreadPool, Signal, Slot

from mb.cancellation import OperationCancelled
from utils.logging_setup import get_logger

from ui.task_context import LongTaskContext, TaskCancelled

_logger = get_logger("task_runner")


class TaskSignals(QObject):
    """Signals emitted by a background task."""

    success = Signal(object)
    error = Signal(str)
    cancelled = Signal()
    done = Signal()
    progress = Signal(str, object, bool)


@dataclass
class BackgroundTaskHandle:
    """Returned by :func:`start_task` for wiring progress UI and cancel controls."""

    signals: TaskSignals
    cancel_event: threading.Event


class TaskRunnable(QRunnable):
    """QRunnable wrapper that executes a Python callable."""

    def __init__(
        self,
        fn: Callable[..., Any],
        cancel_event: threading.Event,
        pass_context: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.cancel_event = cancel_event
        self.pass_context = pass_context
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        def _emit_progress(message: str, percent: object, indeterminate: bool) -> None:
            self.signals.progress.emit(message, percent, indeterminate)

        _emit_progress("Running…", None, True)
        try:
            if self.pass_context:
                ctx = LongTaskContext(self.cancel_event, _emit_progress)
                result = self.fn(ctx, *self.args, **self.kwargs)
            else:
                result = self.fn(*self.args, **self.kwargs)
            self.signals.success.emit(result)
        except OperationCancelled:
            self.signals.cancelled.emit()
        except TaskCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # pragma: no cover - runtime safety
            _logger.exception("Background task failed")
            self.signals.error.emit(str(exc))
        finally:
            self.signals.done.emit()


def start_task(
    fn: Callable[..., Any],
    on_success: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_done: Callable[[], None] | None = None,
    *args: Any,
    pass_context: bool = False,
    on_cancelled: Callable[[], None] | None = None,
    on_progress: Callable[[str, object, bool], None] | None = None,
    **kwargs: Any,
) -> BackgroundTaskHandle:
    """Run a callable on the global thread pool and bind result callbacks.

    Returns a :class:`BackgroundTaskHandle` (signals + shared ``cancel_event``)
    for attaching a progress dialog or other UI.
    """
    cancel_event = threading.Event()
    runnable = TaskRunnable(fn, cancel_event, pass_context, *args, **kwargs)
    ct = Qt.ConnectionType.QueuedConnection
    runnable.signals.success.connect(on_success, ct)
    runnable.signals.error.connect(on_error, ct)
    if on_cancelled is not None:
        runnable.signals.cancelled.connect(on_cancelled, ct)
    if on_done:
        runnable.signals.done.connect(on_done, ct)
    if on_progress:
        runnable.signals.progress.connect(on_progress, ct)
    QThreadPool.globalInstance().start(runnable)
    return BackgroundTaskHandle(signals=runnable.signals, cancel_event=cancel_event)
