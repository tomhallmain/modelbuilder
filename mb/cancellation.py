"""Cooperative cancellation for long-running ``mb`` operations (GUI Cancel, etc.)."""

from __future__ import annotations

import threading
from typing import Optional


class OperationCancelled(Exception):
    """Raised when a long operation should stop because ``cancel_event`` was set."""


class TrainingCancelled(OperationCancelled):
    """Backward-compatible name for training-specific call sites."""


def check_cancel_event(cancel_event: Optional[threading.Event]) -> None:
    """Raise :class:`OperationCancelled` if *cancel_event* is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled()
