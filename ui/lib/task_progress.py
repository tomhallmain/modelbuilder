"""
Modal progress dialog for :mod:`ui.task_runner` background tasks.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QWidget

from mb.utils.translations import _
from ui.task_runner import BackgroundTaskHandle


def attach_progress_dialog(
    parent: Optional[QWidget],
    title: str,
    handle: BackgroundTaskHandle,
    *,
    cancellable: bool = True,
) -> QProgressDialog:
    """
    Show a progress dialog wired to *handle*'s signals.

    When *cancellable* is True, the Cancel button sets ``handle.cancel_event``,
    which cooperative workers (e.g. training loops) should observe.
    """
    dlg = QProgressDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setLabelText(_("Starting…"))
    dlg.setRange(0, 0)
    dlg.setMinimumDuration(0)
    dlg.setModal(True)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)

    if cancellable:
        dlg.setCancelButtonText(_("Cancel"))
        dlg.canceled.connect(handle.cancel_event.set)
    else:
        dlg.setCancelButton(None)

    def on_progress(message: str, percent: object, indeterminate: bool) -> None:
        dlg.setLabelText(message)
        if indeterminate:
            dlg.setRange(0, 0)
        else:
            dlg.setRange(0, 100)
            if isinstance(percent, (int, float)):
                p = float(percent)
                v = int(p * 100) if p <= 1.0 else int(p)
                dlg.setValue(min(100, max(0, v)))

    # TaskRunnable emits *success* (or *error* / *cancelled*) and then always emits *done*
    # from ``finally``. Closing on both would call :meth:`QProgressDialog.reset` twice on the
    # happy path, which can destabilize or crash Qt on Windows (modal teardown + second reset).
    closed = False

    def close_dlg(*_args: object) -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            dlg.reset()
        except Exception:
            pass

    ct = Qt.ConnectionType.QueuedConnection
    handle.signals.progress.connect(on_progress, ct)
    handle.signals.success.connect(close_dlg, ct)
    handle.signals.error.connect(close_dlg, ct)
    handle.signals.cancelled.connect(close_dlg, ct)
    handle.signals.done.connect(close_dlg, ct)

    dlg.show()
    return dlg
