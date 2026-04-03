"""
Marshal arbitrary callables onto the Qt GUI thread.

Used when code runs on a worker thread, ``threading.Timer``, or any non-GUI
thread but must call QWidget APIs or other Qt operations that must run on the
thread that owns the QApplication.

:class:`TaskSignals` in :mod:`ui.task_runner` already queues task callbacks;
this bridge covers ad-hoc paths such as :class:`utils.notification_manager`
title updates and :meth:`ui.app_actions.AppActions.alert`.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, TypeVar

from PySide6.QtCore import QMetaObject, QThread, Qt, Slot
from PySide6.QtWidgets import QApplication, QWidget

F = TypeVar("F", bound=Callable[..., Any])


class MainThreadBridge(QWidget):
    """Marshals callables from non-GUI threads to the main / GUI thread.

    Uses ``QMetaObject.invokeMethod`` with ``BlockingQueuedConnection`` so that
    the calling thread blocks until the callable finishes on the main thread.
    When already on the main thread the callable runs directly (avoids deadlock).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.hide()
        self._lock = threading.Lock()
        self._func: Callable[..., Any] | None = None
        self._args: tuple[Any, ...] = ()
        self._kwargs: dict[str, Any] = {}
        self._result: Any = None
        self._error: BaseException | None = None

    @Slot()
    def _execute(self) -> None:
        assert self._func is not None
        try:
            self._result = self._func(*self._args, **self._kwargs)
        except BaseException as e:
            self._error = e

    def invoke(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call *func* on the GUI thread, blocking until it returns."""
        app = QApplication.instance()
        if app is None or QThread.currentThread() == app.thread():
            return func(*args, **kwargs)
        with self._lock:
            self._func = func
            self._args = args
            self._kwargs = kwargs
            self._result = None
            self._error = None
            QMetaObject.invokeMethod(
                self,
                "_execute",
                Qt.ConnectionType.BlockingQueuedConnection,
            )
            if self._error is not None:
                raise self._error
            return self._result

    def wrap(self, func: F) -> F:
        """Return a wrapper that always invokes *func* on the GUI thread."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.invoke(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]
