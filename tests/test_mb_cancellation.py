"""Unit tests for cooperative training cancellation helpers."""

import threading
import unittest

from mb.cancellation import OperationCancelled, TrainingCancelled, check_cancel_event


class TestMbCancellation(unittest.TestCase):
    def test_check_cancel_event_no_event_is_noop(self) -> None:
        check_cancel_event(None)

    def test_check_cancel_event_raises_when_set(self) -> None:
        ev = threading.Event()
        ev.set()
        with self.assertRaises(OperationCancelled):
            check_cancel_event(ev)
        self.assertTrue(issubclass(TrainingCancelled, OperationCancelled))


if __name__ == "__main__":
    unittest.main()
