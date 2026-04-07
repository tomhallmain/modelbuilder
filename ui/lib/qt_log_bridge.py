"""Forward :mod:`logging` records to a Qt widget via a thread-safe signal."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from PySide6.QtCore import QObject, Qt, Signal


class QtLogBridge(QObject):
    """Emit one line per log record. Connect with ``QueuedConnection`` to a GUI slot."""

    line = Signal(str)


class QtPlainLogHandler(logging.Handler):
    """Handler that emits formatted lines through :class:`QtLogBridge` (any thread → GUI thread)."""

    def __init__(self, bridge: QtLogBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._bridge.line.emit(msg)
        except Exception:
            self.handleError(record)


@contextmanager
def tee_logger_to_qt(bridge: QtLogBridge, *logger_names: str, level: int = logging.INFO) -> Iterator[None]:
    """
    Temporarily attach a :class:`QtPlainLogHandler` to each named logger (e.g. ``"modelbuilder.mb.convert"``).

    Safe when the log calls originate from a worker thread: the bridge uses Qt signals.
    """
    handlers: list[tuple[logging.Logger, QtPlainLogHandler]] = []
    for name in logger_names:
        log = logging.getLogger(name)
        h = QtPlainLogHandler(bridge)
        h.setLevel(level)
        log.addHandler(h)
        handlers.append((log, h))
    try:
        yield
    finally:
        for log, h in handlers:
            try:
                log.removeHandler(h)
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
