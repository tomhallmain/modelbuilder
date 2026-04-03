"""Tests for the synthetic raw-data factory used by pipeline / E2E tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR, SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.data.dataset import MIN_FILE_SIZE

from tests.fixtures.synthetic_dataset import build_synthetic_raw_data_dir


def test_build_synthetic_raw_data_default_total(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    counts = build_synthetic_raw_data_dir(root, total_images=100, seed=1)
    assert sum(counts.values()) == 100
    assert set(counts.keys()) == set(SYNTHETIC_DEFAULT_CLASS_NAMES)
    assert all(c >= 1 for c in counts.values())

    for name in SYNTHETIC_DEFAULT_CLASS_NAMES:
        jpeg_dir = root / name / CONVERTED_MEDIA_SUBDIR
        assert jpeg_dir.is_dir()
        jpgs = list(jpeg_dir.glob("*.jpg"))
        assert len(jpgs) == counts[name]
        for p in jpgs:
            assert p.stat().st_size >= MIN_FILE_SIZE


def test_build_synthetic_requires_enough_images_for_all_classes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="total_images"):
        build_synthetic_raw_data_dir(tmp_path / "r", total_images=2)


def test_synthetic_raw_data_dir_fixture(synthetic_raw_data_dir: Path) -> None:
    assert synthetic_raw_data_dir.name == "raw_data"
    total = 0
    for name in SYNTHETIC_DEFAULT_CLASS_NAMES:
        n = len(list((synthetic_raw_data_dir / name / CONVERTED_MEDIA_SUBDIR).glob("*.jpg")))
        total += n
    assert total == 100
