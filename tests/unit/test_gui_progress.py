"""Unit tests for training GUI progress fractions (no Qt / no torch)."""

import unittest

from mb.training.gui_progress import subepoch_progress_emit


class TestSubepochProgressEmit(unittest.TestCase):
    def test_monotonic_fraction(self) -> None:
        seen: list[float] = []

        def cb(_msg: str, pct: float | None) -> None:
            if pct is not None:
                seen.append(float(pct))

        emit = subepoch_progress_emit(cb, total_plan_epochs=10, epochs_done_before=0, steps_per_epoch=4, epoch_label="e")
        for s in (1, 2, 3, 4):
            emit(s)
        self.assertTrue(all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1)))
        self.assertAlmostEqual(seen[-1], 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
