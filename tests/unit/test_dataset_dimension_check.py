"""Unit tests: create-dataset pixel-dimension floor vs pipeline ``data.image_size``."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from mb.data.dataset import (
    MIN_AREA_FRACTION_OF_IMAGE_SIZE_SQUARED,
    MIN_FILE_SIZE,
    DatasetCreator,
    min_area_pixels_for_pipeline,
    min_edge_pixels_for_pipeline,
)


def test_min_edge_pixels_for_pipeline_scales_with_image_size() -> None:
    assert min_edge_pixels_for_pipeline(224) == 56  # max(32, 224 // 4)
    assert min_edge_pixels_for_pipeline(128) == 32  # max(32, 32)
    assert min_edge_pixels_for_pipeline(512) == 128


def test_min_edge_pixels_for_pipeline_bad_values_fallback() -> None:
    assert min_edge_pixels_for_pipeline(0) == 56  # treated as 224
    assert min_edge_pixels_for_pipeline(-1) == 56


def test_min_area_pixels_for_pipeline_matches_fraction_of_image_size_sq() -> None:
    s = 224
    expected = max(32 * 32, int(MIN_AREA_FRACTION_OF_IMAGE_SIZE_SQUARED * s * s))
    assert min_area_pixels_for_pipeline(224) == expected
    assert min_area_pixels_for_pipeline(224) == 9031


def _write_jpeg_at_least(path: Path, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(width * height * 3))
    img = Image.frombytes("RGB", (width, height), data)
    path.parent.mkdir(parents=True, exist_ok=True)
    for q in (95, 92, 85, 75):
        img.save(path, format="JPEG", quality=q, optimize=False)
        if path.stat().st_size >= MIN_FILE_SIZE:
            return
    raise AssertionError(f"could not reach MIN_FILE_SIZE for {width}x{height}")


def test_remove_invalid_sized_images_moves_too_small_dimensions(tmp_path: Path) -> None:
    """Shorter edge below min_edge_pixels_for_pipeline (224 → 56) goes to review."""
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    tiny = train_c / "tiny.jpg"
    # Not square 55×55: that few pixels cannot produce a JPEG >= MIN_FILE_SIZE (6 KiB).
    # 55×400 keeps min(width,height)==55 but yields a large enough file for the byte check.
    _write_jpeg_at_least(tiny, 55, 400, seed=1)

    dc.remove_invalid_sized_images()

    assert not tiny.exists()
    reviewed = list((dc.review_dir / "c").glob("*.jpg"))
    assert len(reviewed) == 1
    assert dc.stats["files_moved_dimensions"]["c"] == 1
    assert dc.stats["files_moved_size"]["c"] == 0


def test_remove_invalid_sized_images_keeps_large_enough_dimensions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    ok_path = train_c / "ok.jpg"
    _write_jpeg_at_least(ok_path, 96, 96, seed=2)

    dc.remove_invalid_sized_images()

    assert ok_path.is_file()
    assert dc.stats["files_moved_dimensions"]["c"] == 0


def test_remove_invalid_sized_images_moves_insufficient_total_pixels(tmp_path: Path) -> None:
    """95×95 is below min area for image_size 224 (9031 px²) but passes min edge."""
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    p = train_c / "low_area.jpg"
    _write_jpeg_at_least(p, 95, 95, seed=3)
    assert 95 * 95 < min_area_pixels_for_pipeline(224)

    dc.remove_invalid_sized_images()

    assert not p.exists()
    assert dc.stats["files_moved_dimensions"]["c"] == 1


def test_remove_invalid_sized_images_moves_extreme_aspect_ratio(tmp_path: Path) -> None:
    """Wide strip: max/min > 10:1 goes to review."""
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    p = train_c / "strip.jpg"
    _write_jpeg_at_least(p, 80, 900, seed=4)

    dc.remove_invalid_sized_images()

    assert not p.exists()
    assert dc.stats["files_moved_dimensions"]["c"] == 1
