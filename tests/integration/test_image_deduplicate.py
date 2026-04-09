"""Integration: :class:`mb.data.deduplicate.ImageDeduplicator` duplicate removal (no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from mb.data.deduplicate import ImageDeduplicator


def _write_jpg(path: Path, size: tuple[int, int], rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path, quality=92)


def test_image_deduplicator_removes_identical_files_in_converted_folder(tmp_path: Path) -> None:
    """Two byte-identical JPEGs in ``coherent/CONVERTED`` → one removed.

    Step 0 keeps images in ``CONVERTED`` only when **min(width, height) ≥ 250**; smaller
    images are moved to ``small_images_review`` before duplicate scans run.
    """
    raw = tmp_path / "raw_data"
    converted = raw / "coherent" / "CONVERTED"
    _write_jpg(converted / "first.jpg", (300, 300), (10, 80, 160))
    shutil.copy2(converted / "first.jpg", converted / "duplicate.jpg")

    dedup = ImageDeduplicator(raw_data_dir=raw)
    assert dedup.run() is True
    assert dedup.stats["duplicates_removed"] >= 1
    remaining = list(converted.glob("*.jpg"))
    assert len(remaining) == 1


def test_image_deduplicator_ignores_duplicates_outside_converted(tmp_path: Path) -> None:
    """Duplicates outside ``CONVERTED`` are out of scope and must remain untouched."""
    raw = tmp_path / "raw_data"
    class_dir = raw / "coherent"
    converted = class_dir / "CONVERTED"
    images = class_dir / "IMAGES"

    # Ensure deduplicate has in-scope input so the run does real work.
    _write_jpg(converted / "anchor.jpg", (300, 300), (1, 2, 3))

    # Out-of-scope duplicates in IMAGES should never be removed by deduplicate.
    _write_jpg(images / "dup_a.jpg", (300, 300), (10, 80, 160))
    shutil.copy2(images / "dup_a.jpg", images / "dup_b.jpg")

    dedup = ImageDeduplicator(raw_data_dir=raw)
    assert dedup.run() is True

    assert (images / "dup_a.jpg").exists()
    assert (images / "dup_b.jpg").exists()
