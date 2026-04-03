"""Unit tests for ETA formatting (no Qt)."""

import unittest

from ui.lib.eta_format import format_eta_seconds


class TestEtaFormat(unittest.TestCase):
    def test_seconds(self) -> None:
        self.assertIn("45", format_eta_seconds(45))

    def test_minutes(self) -> None:
        s = format_eta_seconds(125)
        self.assertIn("2", s)
        self.assertIn("min", s)

    def test_hours(self) -> None:
        s = format_eta_seconds(7200)
        self.assertIn("2", s)
        self.assertIn("h", s)

    def test_days(self) -> None:
        s = format_eta_seconds(90000)
        self.assertIn("day", s)

    def test_invalid(self) -> None:
        self.assertEqual(format_eta_seconds(float("nan")), "—")


if __name__ == "__main__":
    unittest.main()
