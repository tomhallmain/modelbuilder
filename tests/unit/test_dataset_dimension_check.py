"""Unit tests: create-dataset pixel-dimension floor vs pipeline ``data.image_size``."""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from mb.data.dataset import (
    MIN_AREA_FRACTION_OF_IMAGE_SIZE_SQUARED,
    MIN_FILE_SIZE,
    DatasetCreator,
    min_area_pixels_for_pipeline,
    min_edge_pixels_for_pipeline,
)
from tests.fixtures.pipeline_image_size import HIGH_RES_PIPELINE_IMAGE_SIZE


def test_min_edge_pixels_for_pipeline_scales_with_image_size() -> None:
    assert min_edge_pixels_for_pipeline(224) == 44  # max(24, 224 // 5)
    assert min_edge_pixels_for_pipeline(128) == 25  # max(24, 128 // 5)
    assert min_edge_pixels_for_pipeline(512) == 102
    assert min_edge_pixels_for_pipeline(HIGH_RES_PIPELINE_IMAGE_SIZE) == 64  # max(24, 320 // 5)


def test_min_edge_pixels_for_pipeline_bad_values_fallback() -> None:
    assert min_edge_pixels_for_pipeline(0) == 44  # treated as 224
    assert min_edge_pixels_for_pipeline(-1) == 44


def test_min_area_pixels_for_pipeline_matches_fraction_of_image_size_sq() -> None:
    s = 224
    expected = max(32 * 32, int(MIN_AREA_FRACTION_OF_IMAGE_SIZE_SQUARED * s * s))
    assert min_area_pixels_for_pipeline(224) == expected
    assert min_area_pixels_for_pipeline(224) == 4515


def test_min_area_pixels_for_pipeline_high_res_anchor() -> None:
    """Regression anchor for training above 300px (see ``pipeline_image_size`` fixture)."""
    s = HIGH_RES_PIPELINE_IMAGE_SIZE
    expected = max(32 * 32, int(MIN_AREA_FRACTION_OF_IMAGE_SIZE_SQUARED * s * s))
    assert min_area_pixels_for_pipeline(s) == expected
    assert min_area_pixels_for_pipeline(s) == 9216


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


def test_remove_invalid_sized_images_moves_too_small_shorter_edge(tmp_path: Path) -> None:
    """Shorter edge below 50 px goes to review (relaxed vs old pipeline-scaled edge)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    tiny = train_c / "tiny.jpg"
    # 49×400 keeps file size above MIN_FILE_SIZE but min side < 50.
    _write_jpeg_at_least(tiny, 49, 400, seed=1)

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


def test_remove_invalid_sized_images_keeps_moderate_area_below_old_strict_min(tmp_path: Path) -> None:
    """Moderate squares pass relaxed tiny-image policy (would have failed legacy strict area)."""
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

    dc.remove_invalid_sized_images()

    assert p.is_file()
    assert dc.stats["files_moved_dimensions"]["c"] == 0


def test_remove_invalid_sized_images_moves_when_area_below_relaxed_floor_large_image_size(
    tmp_path: Path,
) -> None:
    """With a large pipeline image_size, area floor can exceed 50²; 54×54 may still fail."""
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    p = train_c / "small_sq.jpg"
    _write_jpeg_at_least(p, 54, 54, seed=5)

    with patch.object(dc, "_pipeline_image_size", return_value=512):
        dc.remove_invalid_sized_images()

    assert not p.exists()
    assert dc.stats["files_moved_dimensions"]["c"] == 1


def test_remove_invalid_sized_images_keeps_ok_square_at_high_res_pipeline_size(tmp_path: Path) -> None:
    """At image_size 320, relaxed area floor is still 2500 px²; 56×56 passes."""
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "data"
    dc = DatasetCreator(raw_data_dir=raw, data_dir=data)
    dc._class_names = ["c"]
    dc.unified_snapshot = None
    train_c = dc.train_dir / "c"
    train_c.mkdir(parents=True)
    p = train_c / "sq.jpg"
    _write_jpeg_at_least(p, 56, 56, seed=6)

    with patch.object(dc, "_pipeline_image_size", return_value=HIGH_RES_PIPELINE_IMAGE_SIZE):
        dc.remove_invalid_sized_images()

    assert p.is_file()
    assert dc.stats["files_moved_dimensions"]["c"] == 0


def test_remove_invalid_sized_images_keeps_extreme_aspect_ratio(tmp_path: Path) -> None:
    """Wide strip: aspect alone no longer sends images to review."""
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

    assert p.is_file()
    assert dc.stats["files_moved_dimensions"]["c"] == 0
