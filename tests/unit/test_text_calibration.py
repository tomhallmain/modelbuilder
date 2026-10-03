"""Tests for ``mb.evaluate.classification.text_calibration``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mb.evaluate.classification.text_calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
    sigmoid,
)
from mb.models.types import TextCalibrationMethod


def _overconfident(n: int = 40_000, factor: float = 2.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < sigmoid(z)).astype(np.int64)
    return sigmoid(factor * z), y


def test_temperature_recovers_scaling_factor() -> None:
    p, y = _overconfident(factor=2.0)
    cal = TemperatureCalibrator().fit(p, y)
    assert cal.temperature == pytest.approx(2.0, rel=0.08)


def test_platt_recovers_slope() -> None:
    p, y = _overconfident(factor=2.0)
    cal = PlattCalibrator().fit(p, y)
    assert cal.a == pytest.approx(0.5, rel=0.08)
    assert abs(cal.b) < 0.1


def test_isotonic_is_monotone_and_fits_steps() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([0, 1, 0, 1, 1, 1])
    cal = IsotonicCalibrator().fit(p, y)
    out = cal.transform(np.linspace(0.0, 1.0, 50))
    assert np.all(np.diff(out) >= -1e-12)
    assert cal.transform(np.array([0.6]))[0] == pytest.approx(1.0)


@pytest.mark.parametrize("method", [TextCalibrationMethod.TEMPERATURE, TextCalibrationMethod.PLATT, TextCalibrationMethod.ISOTONIC])
def test_round_trip_through_json(tmp_path: Path, method: TextCalibrationMethod) -> None:
    p, y = _overconfident(n=2000)
    cal = fit_calibrator(method, p, y)
    path = tmp_path / "calibrator.json"
    save_calibrator(cal, path)
    loaded = load_calibrator(path)
    grid = np.linspace(0.01, 0.99, 25)
    assert np.allclose(loaded.transform(grid), cal.transform(grid))


def test_fit_needs_both_labels() -> None:
    with pytest.raises(ValueError):
        fit_calibrator(TextCalibrationMethod.PLATT, np.array([0.2, 0.3]), np.array([1, 1]))
