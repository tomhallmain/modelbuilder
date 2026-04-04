"""
Modal training progress: bar, status line, and ETA (hours/days when needed).

Wired to :class:`ui.task_runner.BackgroundTaskHandle` like :func:`ui.lib.task_progress.attach_progress_dialog`.
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mb.utils.translations import _
from ui.lib.eta_format import format_eta_seconds
from ui.task_runner import BackgroundTaskHandle


def _format_eta_label(seconds: float | None) -> str:
    if seconds is None:
        return _("Estimated time remaining: —")
    if seconds != seconds or seconds < 0:
        return _("Estimated time remaining: —")
    return _("Estimated time remaining: {eta}").format(
        eta=format_eta_seconds(float(seconds)),
    )


def _format_elapsed_label(seconds: float) -> str:
    return _("Elapsed: {t}").format(t=format_eta_seconds(float(seconds)))


class TrainingProgressDialog(QDialog):
    """Training progress with determinate bar (when available) and ETA estimate."""

    def __init__(self, parent: Optional[QWidget], title: str, *, cancellable: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        flags = self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)

        root = QVBoxLayout(self)
        self._status = QLabel(_("Starting…"))
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        root.addWidget(self._bar)

        self._eta = QLabel(_format_eta_label(None))
        # Match normal dialog text (do not use palette(mid) — unreadable on dark themes).
        self._eta.setWordWrap(True)
        root.addWidget(self._eta)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn: Optional[QPushButton] = None
        if cancellable:
            self._cancel_btn = QPushButton(_("Cancel"))
            btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self._t_eta_start: float | None = None
        self._t_open = time.monotonic()

    def set_cancel_handler(self, on_cancel: object) -> None:
        if self._cancel_btn is not None:
            self._cancel_btn.clicked.connect(on_cancel)

    def on_progress(self, message: str, percent: object, indeterminate: bool) -> None:
        self._status.setText(message)
        if indeterminate:
            self._bar.setRange(0, 0)
            elapsed = time.monotonic() - self._t_open
            self._eta.setText(_format_elapsed_label(elapsed) + " · " + _("ETA unknown"))
            self._t_eta_start = None
            return

        self._bar.setRange(0, 100)
        if not isinstance(percent, (int, float)):
            self._bar.setRange(0, 0)
            elapsed = time.monotonic() - self._t_open
            self._eta.setText(_format_elapsed_label(elapsed) + " · " + _("ETA unknown"))
            return

        p = float(percent)
        v = int(p * 100) if p <= 1.0 else int(p)
        self._bar.setValue(min(100, max(0, v)))

        frac = p if p <= 1.0 else p / 100.0
        if frac >= 1.0:
            self._eta.setText(_("Estimated time remaining: almost done"))
            return
        if frac <= 0:
            self._eta.setText(_("Estimated time remaining: —"))
            return

        now = time.monotonic()
        if self._t_eta_start is None:
            self._t_eta_start = now
        elapsed = now - self._t_eta_start
        # elapsed / frac estimates total time from first in-range sample; remaining = T - elapsed = elapsed*(1/frac - 1)
        remaining = elapsed * (1.0 / frac - 1.0)
        if frac < 0.005 and elapsed < 8.0:
            self._eta.setText(_("Estimated time remaining: calculating…"))
            return
        if remaining > 365 * 24 * 3600:
            self._eta.setText(_("Estimated time remaining: —"))
            return
        self._eta.setText(_format_eta_label(remaining))


def attach_training_progress_dialog(
    parent: Optional[QWidget],
    title: str,
    handle: BackgroundTaskHandle,
    *,
    cancellable: bool = True,
) -> TrainingProgressDialog:
    """
    Show a training progress dialog with ETA, wired to *handle*'s signals.

    *percent* from the worker should be 0.0–1.0 overall job progress when known; the
    trainer emits coarse fractions during setup/eval. Indeterminate ``percent is None``
    (e.g. initial task runner message) shows elapsed time only.
    """
    dlg = TrainingProgressDialog(parent, title, cancellable=cancellable)

    def on_cancel_clicked() -> None:
        reply = QMessageBox.question(
            dlg,
            _("Cancel training"),
            _("Stop training? Partial checkpoints may already exist in the output folder."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if dlg._cancel_btn is not None:
                dlg._cancel_btn.setEnabled(False)
                dlg._cancel_btn.setText(_("Cancelling…"))
            dlg._status.setText(_("Cancelling…"))
            handle.cancel_event.set()

    if cancellable:
        dlg.set_cancel_handler(on_cancel_clicked)

    ct = Qt.ConnectionType.QueuedConnection
    handle.signals.progress.connect(dlg.on_progress, ct)
    handle.signals.done.connect(dlg.accept, ct)

    dlg.show()
    return dlg
