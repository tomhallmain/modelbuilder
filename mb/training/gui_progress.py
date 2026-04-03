"""
Shared progress fraction helpers for desktop training (ETA-friendly sub-epoch updates).

Keras callback factories import TensorFlow lazily inside each function so importing
:func:`subepoch_progress_emit` alone does not load TF.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from mb.cancellation import check_cancel_event


def subepoch_progress_emit(
    progress_cb: Optional[Callable[[str, Optional[float]], None]],
    total_plan_epochs: int,
    epochs_done_before: int,
    steps_per_epoch: int,
    epoch_label: str,
    min_interval: float = 0.35,
) -> Callable[[int], None]:
    """Emit (message, overall fraction) after train/val batches for GUI ETA."""
    if progress_cb is None:
        return lambda _step: None
    spe = max(steps_per_epoch, 1)
    last_emit = [0.0]

    def emit(step_in_epoch: int) -> None:
        now = time.monotonic()
        denom = max(total_plan_epochs * spe, 1)
        num = epochs_done_before * spe + min(step_in_epoch, spe)
        pct = min(num / denom, 1.0)
        stride = max(1, spe // 40)
        if (
            pct < 1.0
            and now - last_emit[0] < min_interval
            and step_in_epoch % stride != 0
            and step_in_epoch < spe
        ):
            return
        last_emit[0] = now
        progress_cb(
            f"{epoch_label} — batch {step_in_epoch}/{spe}",
            pct,
        )

    return emit


def make_keras_frozen_gui_progress(
    progress_cb: Optional[Callable[[str, Optional[float]], None]],
    cancel_event: Optional[threading.Event],
    total_plan_epochs: int,
    frozen_epochs: int,
    frozen_epochs_completed: int,
    train_loader: Any,
    val_loader: Any,
) -> Any:
    """Keras callback: frozen-phase training with sub-epoch progress for the GUI."""
    from tensorflow import keras

    spe = max(len(train_loader) + len(val_loader), 1)
    train_steps = len(train_loader)

    class FrozenGuiProgress(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._emit: Optional[Callable[[int], None]] = None

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)
            if progress_cb is None:
                return
            epoch_num = frozen_epochs_completed + epoch + 1
            epochs_done_before = frozen_epochs_completed + epoch
            label = f"Frozen phase: epoch {epoch_num}/{frozen_epochs}"
            self._emit = subepoch_progress_emit(
                progress_cb,
                total_plan_epochs,
                epochs_done_before,
                spe,
                label,
            )

        def on_train_batch_begin(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)

        def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)
            if self._emit is not None:
                self._emit(batch + 1)

        def on_test_batch_begin(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)

        def on_test_batch_end(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)
            if self._emit is not None:
                self._emit(train_steps + batch + 1)

    return FrozenGuiProgress()


def make_keras_unfrozen_gui_progress(
    progress_cb: Optional[Callable[[str, Optional[float]], None]],
    cancel_event: Optional[threading.Event],
    total_plan_epochs: int,
    frozen_epochs: int,
    unfrozen_epochs: int,
    unfrozen_epochs_completed: int,
    train_loader: Any,
    val_loader: Any,
) -> Any:
    """Keras callback: fine-tuning phase with sub-epoch progress for the GUI."""
    from tensorflow import keras

    spe = max(len(train_loader) + len(val_loader), 1)
    train_steps = len(train_loader)

    class UnfrozenGuiProgress(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._emit: Optional[Callable[[int], None]] = None

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)
            if progress_cb is None:
                return
            epoch_num = unfrozen_epochs_completed + epoch + 1
            epochs_done_before = frozen_epochs + unfrozen_epochs_completed + epoch
            label = f"Fine-tune: epoch {epoch_num}/{unfrozen_epochs}"
            self._emit = subepoch_progress_emit(
                progress_cb,
                total_plan_epochs,
                epochs_done_before,
                spe,
                label,
            )

        def on_train_batch_begin(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)

        def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)
            if self._emit is not None:
                self._emit(batch + 1)

        def on_test_batch_begin(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)

        def on_test_batch_end(self, batch: int, logs: Any = None) -> None:
            check_cancel_event(cancel_event)
            if self._emit is not None:
                self._emit(train_steps + batch + 1)

    return UnfrozenGuiProgress()
