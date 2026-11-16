"""Qt's own messages (qDebug/qWarning/…) routed through Python logging.

Install before the first ``QApplication``. In a Windows build started from a
terminal (``--windows-console-mode=attach``) the C-level stderr is in
wide-character mode, and Qt's default handler writing byte strings to it makes
the C runtime abort the process at the first Qt warning.
"""

from __future__ import annotations

import logging

_LOGGER_NAME = "qt"
# The installed handler; held so it stays alive for the process.
_handler = None


def _logger() -> logging.Logger:
    from mb.utils.logging_setup import get_logger

    return get_logger(_LOGGER_NAME)


def install_qt_message_handler() -> None:
    """Route Qt's messages to the ``qt`` logger at the matching level. Idempotent."""
    global _handler
    if _handler is not None:
        return
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    logger = _logger()

    def handler(mode, _context, message) -> None:
        # Qt's text as is: it is not an app message and has no msgid.
        logger.log(levels.get(mode, logging.WARNING), "Qt: %s", message)

    _handler = handler
    qInstallMessageHandler(handler)


def qt_message_reaches_logging() -> bool:
    """Emit a ``qWarning`` and report whether the ``qt`` logger received it."""
    from PySide6.QtCore import qWarning

    marker = "qt message routing check"
    received = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            received.append(record.getMessage())

    logger = _logger()
    capture = _Capture(level=logging.WARNING)
    logger.addHandler(capture)
    try:
        qWarning(marker)
    finally:
        logger.removeHandler(capture)
    return any(marker in message for message in received)
