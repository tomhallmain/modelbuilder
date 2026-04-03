"""Integration: :class:`mb.data.deduplicate.ImageDeduplicator` duplicate removal (no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from mb.data.deduplicate import ImageDeduplicator


def _write_jpg(path: Path, size: tuple[int, int], rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path, quality=92)


def test_image_deduplicator_removes_identical_files_in_class_folder(tmp_path: Path) -> None:
    """Two byte-identical JPEGs in ``coherent/`` → one removed.

    Step 0 keeps images in ``coherent`` only when **min(width, height) ≥ 250**; smaller
    images are moved to ``small_images_review`` before duplicate scans run.
    """
    raw = tmp_path / "raw_data"
    coherent = raw / "coherent"
    _write_jpg(coherent / "first.jpg", (300, 300), (10, 80, 160))
    shutil.copy2(coherent / "first.jpg", coherent / "duplicate.jpg")

    dedup = ImageDeduplicator(raw_data_dir=raw)
    assert dedup.run() is True
    assert dedup.stats["duplicates_removed"] >= 1
    remaining = list(coherent.glob("*.jpg"))
    assert len(remaining) == 1
