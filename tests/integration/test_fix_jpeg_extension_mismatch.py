"""Integration: :func:`mb.data.find_jpeg_extension_mismatches.repair_mislabeled_jpeg_extensions`."""

from __future__ import annotations

import os
import random
import shlex
import shutil
from pathlib import Path

from PIL import Image

from mb.cli import main as mb_main
from mb.models.types import ModelBuildStepCommand
from mb.utils.snapshot import UnifiedSnapshot, calculate_file_hash, save_unified_snapshot

from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR, VISUAL_MEDIA_REVIEW_SUBDIR
from mb.data.find_jpeg_extension_mismatches import (
    JPEG_MAGIC,
    read_file_header,
    repair_mislabeled_jpeg_extensions,
    sniff_container,
)
from mb.models.types import ModelType


def _write_animated_gif_at_jpg_path(path: Path) -> None:
    """Two-frame GIF saved to a ``.jpg`` filename (mislabeled source)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        Image.new("RGB", (12, 12), (200, 0, 0)),
        Image.new("RGB", (12, 12), (0, 200, 0)),
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        format="GIF",
    )


def _write_single_frame_gif_at_jpg_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), (50, 50, 50)).save(path, format="GIF")


def _assert_jpeg_file(path: Path) -> None:
    assert path.is_file()
    assert read_file_header(path)[:3] == JPEG_MAGIC
    with Image.open(path) as im:
        assert im.format == "JPEG"


def test_sniff_and_header_helpers_for_gif_named_jpg(tmp_path: Path) -> None:
    p = tmp_path / "fake.jpg"
    _write_animated_gif_at_jpg_path(p)
    head = read_file_header(p)
    assert sniff_container(head) == "gif"
    assert head[:6] == b"GIF89a"


def test_repair_dry_run_leaves_mislabeled_files_in_place(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    images = raw / "coherent" / "IMAGES"
    _write_animated_gif_at_jpg_path(images / "trick.jpg")

    ok, stats = repair_mislabeled_jpeg_extensions(
        raw,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        dry_run=True,
        rng=random.Random(0),
    )
    assert ok is True
    assert stats.dry_run_actions >= 1
    assert (images / "trick.jpg").is_file()
    assert not (images / "trick.gif").exists()


def test_repair_animated_gif_mislabeled_as_jpg_image_classification(
    tmp_path: Path,
) -> None:
    """
    Stale ``CONVERTED/*.jpg`` and ``small_images_review/*.jpg`` holding GIF bytes are removed only
    after a real JPEG is written (hash basename + visual review copy), and the source is renamed
    to ``.gif``.
    """
    raw = tmp_path / "raw_data"
    class_dir = raw / "coherent"
    images = class_dir / "IMAGES"
    converted = class_dir / CONVERTED_MEDIA_SUBDIR
    review = class_dir / VISUAL_MEDIA_REVIEW_SUBDIR
    small_review = class_dir / "small_images_review"

    source_jpg = images / "trick.jpg"
    _write_animated_gif_at_jpg_path(source_jpg)

    converted.mkdir(parents=True, exist_ok=True)
    stale_converted = converted / "trick.jpg"
    shutil.copy2(source_jpg, stale_converted)

    small_review.mkdir(parents=True, exist_ok=True)
    stale_small = small_review / "trick.jpg"
    shutil.copy2(source_jpg, stale_small)

    ok, stats = repair_mislabeled_jpeg_extensions(
        raw,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        dry_run=False,
        rng=random.Random(0),
    )
    assert ok is True
    assert stats.files_repaired >= 1

    assert not source_jpg.exists()
    assert (images / "trick.gif").is_file()
    assert sniff_container(read_file_header(images / "trick.gif")) == "gif"

    assert not stale_converted.exists()
    assert not stale_small.exists()

    jpgs = sorted(converted.glob("trick_*.jpg"))
    assert len(jpgs) == 1
    _assert_jpeg_file(jpgs[0])

    vjpgs = sorted(review.glob("trick_*.jpg"))
    assert len(vjpgs) == 1
    _assert_jpeg_file(vjpgs[0])


def test_repair_single_frame_gif_mislabeled_overwrites_converted_still(tmp_path: Path) -> None:
    """Single-frame GIF under a ``.jpg`` name uses the still path (plain CONVERTED basename)."""
    raw = tmp_path / "raw_data"
    class_dir = raw / "coherent"
    images = class_dir / "IMAGES"
    converted = class_dir / CONVERTED_MEDIA_SUBDIR

    source_jpg = images / "still.jpg"
    _write_single_frame_gif_at_jpg_path(source_jpg)

    converted.mkdir(parents=True, exist_ok=True)
    stale = converted / "still.jpg"
    shutil.copy2(source_jpg, stale)

    ok, stats = repair_mislabeled_jpeg_extensions(
        raw,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        dry_run=False,
        rng=random.Random(0),
    )
    assert ok is True
    assert stats.files_repaired >= 1

    assert (images / "still.gif").is_file()
    out = converted / "still.jpg"
    assert out.is_file()
    _assert_jpeg_file(out)


def test_repair_animated_gif_object_detection_uses_still_basename_not_visual_review(
    tmp_path: Path,
) -> None:
    """
    Non–image-classification model types use the still path (first frame) and plain ``stem.jpg``,
    not the hash-named random-frame output or ``visual_media_review`` (see
    :func:`~mb.data.find_jpeg_extension_mismatches.repair_mislabeled_jpeg_extensions`).
    """
    raw = tmp_path / "raw_data"
    class_dir = raw / "coherent"
    images = class_dir / "IMAGES"
    converted = class_dir / CONVERTED_MEDIA_SUBDIR
    review = class_dir / VISUAL_MEDIA_REVIEW_SUBDIR

    source_jpg = images / "anim.jpg"
    _write_animated_gif_at_jpg_path(source_jpg)
    converted.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_jpg, converted / "anim.jpg")

    ok, stats = repair_mislabeled_jpeg_extensions(
        raw,
        model_type=ModelType.OBJECT_DETECTION,
        dry_run=False,
        rng=random.Random(1),
    )
    assert ok is True
    assert stats.files_repaired >= 1

    assert (images / "anim.gif").is_file()
    _assert_jpeg_file(converted / "anim.jpg")
    assert not any(review.glob("*.jpg")), "visual review is only used for image-classification animated GIFs"


def test_cli_main_freeform_multiline_argv_matches_data_page_wildcard_split(tmp_path: Path) -> None:
    """
    The Data page Wildcard tab joins ``mb data <subcommand>`` with :func:`shlex.split` of the text
    box (POSIX rules on Unix, non-POSIX on Windows) and calls :func:`mb.cli.main`. Exercise the
    same shape here: multiline flags, then ``main(['data', 'fix-jpeg-extension-mismatch', ...])``.
    """
    raw = tmp_path / "raw_data"
    images = raw / "coherent" / "IMAGES"
    _write_animated_gif_at_jpg_path(images / "wild.jpg")

    # Same parsing as ``DataPage._parse_wildcard_extra_argv`` (line breaks allowed).
    text = f"""--dry-run
--raw-data-dir
{raw}"""
    extra = shlex.split(text.strip(), posix=os.name != "nt")
    argv = ["data", "fix-jpeg-extension-mismatch", *extra]
    code = mb_main(argv)
    # Handler exits non-zero when --dry-run finds at least one mislabeled .jpg (audit signal).
    assert code == 1

    text_clean = f"""--dry-run
--raw-data-dir
{tmp_path / "empty_raw"}"""
    (tmp_path / "empty_raw").mkdir()
    (tmp_path / "empty_raw" / "only_class").mkdir()
    extra_clean = shlex.split(text_clean.strip(), posix=os.name != "nt")
    argv_clean = ["data", "fix-jpeg-extension-mismatch", *extra_clean]
    assert mb_main(argv_clean) == 0


def test_repair_updates_unified_snapshot_original_and_converted(tmp_path: Path) -> None:
    """Mislabeled repair refreshes ``original`` / ``converted`` on matching snapshot rows only."""
    raw = tmp_path / "raw_data"
    class_dir = raw / "coherent"
    images = class_dir / "IMAGES"
    converted = class_dir / CONVERTED_MEDIA_SUBDIR
    small_review = class_dir / "small_images_review"

    source_jpg = images / "trick.jpg"
    _write_animated_gif_at_jpg_path(source_jpg)

    converted.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_jpg, converted / "trick.jpg")
    small_review.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_jpg, small_review / "trick.jpg")

    md5_orig = calculate_file_hash(source_jpg, "md5", raw_data_dir=raw)
    assert md5_orig
    snap = UnifiedSnapshot(run_id="snaptest", raw_data_dir=str(raw.resolve()))
    snap.images[md5_orig] = {
        "original": {
            "basename": "trick.jpg",
            "hash": md5_orig,
            "path": "coherent/IMAGES/trick.jpg",
            "format": ".jpg",
        },
        "converted": {
            "class": "coherent",
            "path": "coherent/CONVERTED/stale.jpg",
            "basename": "stale.jpg",
            "md5": "00",
            "sha256": "11",
            "was_converted": True,
        },
    }
    assert save_unified_snapshot(snap, raw, logger=None) is not None

    ok, stats = repair_mislabeled_jpeg_extensions(
        raw,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        dry_run=False,
        rng=random.Random(0),
    )
    assert ok is True
    assert stats.snapshot_records_updated >= 1

    loaded = UnifiedSnapshot.load(raw / "snapshot_snaptest.json", silent=True)
    assert loaded is not None
    rec = loaded.images.get(md5_orig)
    assert rec is not None
    assert rec["original"]["path"].endswith("trick.gif")
    assert rec["original"]["format"] == ".gif"
    conv = rec.get("converted") or {}
    assert conv.get("class") == "coherent"
    assert "CONVERTED" in (conv.get("path") or "")
    assert conv.get("basename", "").endswith(".jpg")

    se = loaded.step_errors.get(ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH.value)
    assert se
    assert any(
        any(m.startswith("wall_seconds:") for m in msgs) for msgs in se.values()
    )
