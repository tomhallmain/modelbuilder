"""
Modal training progress: bar, status line, and ETA (hours/days when needed).

Wired to :class:`ui.task_runner.BackgroundTaskHandle` like :func:`ui.lib.task_progress.attach_progress_dialog`.
"""

from __future__ import annotations

import re
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
        self._eta_epoch = QLabel(_("Current epoch ETA: —"))
        self._eta_epoch.setWordWrap(True)
        root.addWidget(self._eta_epoch)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn: Optional[QPushButton] = None
        if cancellable:
            self._cancel_btn = QPushButton(_("Cancel"))
            btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self._t_eta_start: float | None = None
        self._eta_frac_start: float | None = None
        self._epoch_key: str | None = None
        self._epoch_batch_start: int | None = None
        self._epoch_t_start: float | None = None
        self._t_open = time.monotonic()

    def set_cancel_handler(self, on_cancel: object) -> None:
        if self._cancel_btn is not None:
            self._cancel_btn.clicked.connect(on_cancel)

    def on_progress(self, message: str, percent: object, indeterminate: bool) -> None:
        self._status.setText(message)
        self._update_epoch_eta(message)
        if indeterminate:
            self._bar.setRange(0, 0)
            elapsed = time.monotonic() - self._t_open
            self._eta.setText(_format_elapsed_label(elapsed) + " · " + _("ETA unknown"))
            self._eta_epoch.setText(_("Current epoch ETA: —"))
            self._t_eta_start = None
            self._eta_frac_start = None
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
            # Baseline may be >0 (e.g. resumed jobs or phase-weighted progress).
            # ETA should use progress achieved in *this* run, not absolute global fraction.
            self._eta_frac_start = frac
        elapsed = now - self._t_eta_start
        frac0 = self._eta_frac_start if self._eta_frac_start is not None else 0.0
        delta_frac = frac - frac0
        if delta_frac < 0.002 and elapsed < 8.0:
            self._eta.setText(_("Estimated time remaining: calculating…"))
            return
        rate = max(1e-9, delta_frac / max(elapsed, 1e-9))
        # Global ETA: time remaining until overall progress reaches 100%.
        remaining = max(0.0, (1.0 - frac) / rate)
        if remaining > 365 * 24 * 3600:
            self._eta.setText(_("Estimated time remaining: —"))
            return
        self._eta.setText(_format_eta_label(remaining))

    def _update_epoch_eta(self, message: str) -> None:
        # Expected pattern from training emitters: "<epoch label> — batch <n>/<total>"
        m = re.search(r"^(?P<label>.+?)\s+—\s+batch\s+(?P<n>\d+)/(?P<tot>\d+)$", str(message).strip())
        if not m:
            self._eta_epoch.setText(_("Current epoch ETA: —"))
            self._epoch_key = None
            self._epoch_batch_start = None
            self._epoch_t_start = None
            return
        key = m.group("label")
        n = int(m.group("n"))
        tot = int(m.group("tot"))
        now = time.monotonic()
        if key != self._epoch_key:
            self._epoch_key = key
            self._epoch_batch_start = n
            self._epoch_t_start = now
            self._eta_epoch.setText(_("Current epoch ETA: calculating…"))
            return
        if self._epoch_t_start is None or self._epoch_batch_start is None:
            self._epoch_t_start = now
            self._epoch_batch_start = n
            self._eta_epoch.setText(_("Current epoch ETA: calculating…"))
            return
        delta_batches = n - self._epoch_batch_start
        elapsed = now - self._epoch_t_start
        if delta_batches < 1 or elapsed < 1.0:
            self._eta_epoch.setText(_("Current epoch ETA: calculating…"))
            return
        rate = delta_batches / max(elapsed, 1e-9)  # batches/sec
        remaining_batches = max(0, tot - n)
        remaining_s = remaining_batches / max(rate, 1e-9)
        self._eta_epoch.setText(
            _("Current epoch ETA: {eta}").format(eta=format_eta_seconds(float(remaining_s)))
        )


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
