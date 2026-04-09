"""Integration: :class:`mb.data.deduplicate.ImageDeduplicator` duplicate removal (no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from mb.data.deduplicate import ImageDeduplicator
from mb.utils.snapshot import UnifiedSnapshot


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


def test_image_deduplicator_list_only_removes_intra_class_keeps_cross_class_for_snapshot(
    tmp_path: Path,
) -> None:
    """List-only: within-class dups removed; only cross-class groups go to snapshot review."""
    raw = tmp_path / "raw_data"
    cats_c = raw / "cats" / "CONVERTED"
    dogs_c = raw / "dogs" / "CONVERTED"
    _write_jpg(cats_c / "unique_c.jpg", (300, 300), (1, 2, 3))
    _write_jpg(cats_c / "intra_a.jpg", (300, 300), (10, 20, 30))
    shutil.copy2(cats_c / "intra_a.jpg", cats_c / "intra_b.jpg")
    _write_jpg(dogs_c / "unique_d.jpg", (300, 300), (4, 5, 6))
    _write_jpg(dogs_c / "intra_x.jpg", (300, 300), (40, 50, 60))
    shutil.copy2(dogs_c / "intra_x.jpg", dogs_c / "intra_y.jpg")
    _write_jpg(cats_c / "cross.jpg", (300, 300), (100, 101, 102))
    shutil.copy2(cats_c / "cross.jpg", dogs_c / "cross_other_name.jpg")

    snap = UnifiedSnapshot(run_id="rid_dedup", raw_data_dir=str(raw))
    for key, rel, bn, cls in [
        ("k1", "cats/CONVERTED/unique_c.jpg", "unique_c.jpg", "cats"),
        ("k2", "cats/CONVERTED/intra_a.jpg", "intra_a.jpg", "cats"),
        ("k3", "cats/CONVERTED/intra_b.jpg", "intra_b.jpg", "cats"),
        ("k4", "dogs/CONVERTED/unique_d.jpg", "unique_d.jpg", "dogs"),
        ("k5", "dogs/CONVERTED/intra_x.jpg", "intra_x.jpg", "dogs"),
        ("k6", "dogs/CONVERTED/intra_y.jpg", "intra_y.jpg", "dogs"),
        ("k7", "cats/CONVERTED/cross.jpg", "cross.jpg", "cats"),
        ("k8", "dogs/CONVERTED/cross_other_name.jpg", "cross_other_name.jpg", "dogs"),
    ]:
        snap.images[key] = {
            "original": {"hash": key, "basename": bn, "path": rel.replace("/CONVERTED/", "/IMAGES/"), "format": ".jpg"},
            "converted": {"path": rel, "basename": bn, "class": cls},
            "dataset": None,
            "training": None,
        }
    assert snap.save(raw / "snapshot_rid_dedup.json")

    dedup = ImageDeduplicator(raw_data_dir=raw)
    assert dedup.run(list_only=True, run_id="rid_dedup") is True

    assert dedup.stats["duplicates_removed"] >= 2
    jpg_cats = list(cats_c.glob("*.jpg"))
    jpg_dogs = list(dogs_c.glob("*.jpg"))
    assert len(jpg_cats) == 3
    assert len(jpg_dogs) == 3
    intra_remain = {p.name for p in jpg_cats} & {"intra_a.jpg", "intra_b.jpg"}
    assert len(intra_remain) == 1
    intra_remain_d = {p.name for p in jpg_dogs} & {"intra_x.jpg", "intra_y.jpg"}
    assert len(intra_remain_d) == 1
    assert (cats_c / "cross.jpg").exists()
    assert (dogs_c / "cross_other_name.jpg").exists()

    assert len(dedup.duplicate_groups) == 1
    grp = dedup.duplicate_groups[0]
    paths = set(grp["files"])
    assert len(paths) == 2
    assert {Path(p).name for p in paths} == {"cross.jpg", "cross_other_name.jpg"}

    loaded = UnifiedSnapshot.load(raw / "snapshot_rid_dedup.json")
    assert loaded is not None
    assert len(loaded.deduplication_results) == 1
    assert len(loaded.deduplication_results[0]["files"]) == 2
