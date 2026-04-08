"""
Regression tests for :func:`ui.lib.task_progress.attach_progress_dialog`.

``TaskRunnable`` emits a terminal outcome signal (*success*, *error*, or *cancelled*)
and then always emits *done*. The progress dialog was wired to close on both the
outcome and *done*, so ``QProgressDialog.reset()`` could run twice on the happy
path and destabilize Qt (especially on Windows). The implementation must close
at most once per dialog lifetime.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QProgressDialog, QWidget

from mb.cancellation import OperationCancelled
from ui.lib.task_progress import attach_progress_dialog
from ui.task_context import LongTaskContext
from ui.task_runner import start_task


def _track_progress_dialog_reset():
    """Return (counter list, patch target) so each real ``reset()`` appends to *counter*."""
    counter: list[int] = []

    real = QProgressDialog.reset

    def tracked(self: QProgressDialog) -> None:
        counter.append(1)
        return real(self)

    return counter, patch.object(QProgressDialog, "reset", tracked)


@pytest.mark.ui
def test_progress_dialog_reset_once_on_success(qtbot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    finished = {"ok": False}

    def worker(ctx: LongTaskContext) -> bool:
        return True

    def on_success(_: object) -> None:
        pass

    def on_error(_: str) -> None:
        pytest.fail("unexpected error")

    def on_done() -> None:
        finished["ok"] = True

    counter, p_reset = _track_progress_dialog_reset()
    with p_reset:
        handle = start_task(
            worker,
            on_success,
            on_error,
            on_done,
            pass_context=True,
        )
        dlg = attach_progress_dialog(parent, "task", handle)
        qtbot.addWidget(dlg)
        qtbot.waitUntil(lambda: finished["ok"] is True, timeout=10_000)
    assert sum(counter) == 1


@pytest.mark.ui
def test_progress_dialog_reset_once_on_error(qtbot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    finished = {"ok": False}

    def worker(ctx: LongTaskContext) -> bool:
        raise RuntimeError("expected failure")

    def on_success(_: object) -> None:
        pytest.fail("unexpected success")

    def on_error(_: str) -> None:
        pass

    def on_done() -> None:
        finished["ok"] = True

    counter, p_reset = _track_progress_dialog_reset()
    with p_reset:
        handle = start_task(
            worker,
            on_success,
            on_error,
            on_done,
            pass_context=True,
        )
        dlg = attach_progress_dialog(parent, "task", handle)
        qtbot.addWidget(dlg)
        qtbot.waitUntil(lambda: finished["ok"] is True, timeout=10_000)
    assert sum(counter) == 1


@pytest.mark.ui
def test_progress_dialog_reset_once_on_cancelled(qtbot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    finished = {"ok": False}

    def worker(ctx: LongTaskContext) -> bool:
        raise OperationCancelled()

    def on_success(_: object) -> None:
        pytest.fail("unexpected success")

    def on_error(_: str) -> None:
        pytest.fail("unexpected error")

    def on_done() -> None:
        finished["ok"] = True

    counter, p_reset = _track_progress_dialog_reset()
    with p_reset:
        handle = start_task(
            worker,
            on_success,
            on_error,
            on_done,
            pass_context=True,
        )
        dlg = attach_progress_dialog(parent, "task", handle)
        qtbot.addWidget(dlg)
        qtbot.waitUntil(lambda: finished["ok"] is True, timeout=10_000)
    assert sum(counter) == 1
