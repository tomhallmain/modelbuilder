"""Unit tests for GUI background task context (no Qt required)."""

import threading
import unittest

from ui.task_context import LongTaskContext, TaskCancelled


class TestLongTaskContext(unittest.TestCase):
    def test_progress_emits_and_cancel_raises(self) -> None:
        emitted: list[tuple[str, object, bool]] = []

        def emit(message: str, percent: object, indeterminate: bool) -> None:
            emitted.append((message, percent, indeterminate))

        ev = threading.Event()
        ctx = LongTaskContext(ev, emit)
        ctx.progress("phase", 0.5)
        self.assertEqual(emitted[-1], ("phase", 0.5, False))
        ctx.progress("indeterminate")
        self.assertEqual(emitted[-1][2], True)
        ev.set()
        with self.assertRaises(TaskCancelled):
            ctx.raise_if_cancelled()


if __name__ == "__main__":
    unittest.main()
