"""Integration: convert reruns, snapshot ``converted`` metadata, and downstream path matching.

**Tests that touch convert reruns / ``run_id`` resume**

- :func:`test_convert_records_converted_paths_for_plain_stem_outputs` — first convert run; snapshot
  ``converted`` block for plain ``{stem}.jpg`` when that source is the stem winner.
- :func:`test_convert_resume_skip_backfills_converted_metadata` — second convert with same ``run_id``,
  mix of skip (existing output) + new source file.
- :func:`test_convert_second_run_all_skipped_returns_false` — resume with nothing left to do;
  ``run()`` is false (no bytes processed) but snapshot is still valid.
- ``tests/integration/test_image_converter.py`` — :func:`test_image_converter_resume_updates_same_snapshot_file`
  (same ``snapshot_<run_id>.json`` path; adds a new PNG on the second run).

**Create-dataset + :func:`calculate_file_hash` (training-style path)**

- :func:`test_convert_then_dataset_preserves_single_snapshot_record` — convert → create-dataset with
  ``run_id``; single snapshot row gets ``dataset``.
- :func:`test_after_create_dataset_calculate_file_hash_matches_snapshot_via_dataset_path` — full
  pipeline then MD5 via ``dataset.path`` (same mechanism as training snapshot updates).
- :func:`test_calculate_file_hash_returns_converted_md5_via_dataset_path` — synthetic snapshot only
  (no filesystem pipeline).
- :func:`test_run_two_retries_new_sources_without_duplicating_run_one_outputs` — resume with
  :func:`~mb.utils.utils.assign_still_convert_output_basenames` (one plain ``{stem}.jpg`` per stem,
  hash affix for stem collisions).
- :func:`test_convert_resume_plain_stem_output_idempotent` — second run skips when ``{stem}.jpg``
  already exists.
- :func:`test_convert_promotes_legacy_hash_suffixed_file_to_plain_stem` — hash-only output on disk
  is renamed to plain ``{stem}.jpg`` when that is the assigned target for a sole source.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image

from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR
from mb.data.dataset import MIN_FILE_SIZE
from mb.data.convert import ImageConverter
from mb.data.dataset import DatasetCreator
from mb.utils.snapshot import UnifiedSnapshot, calculate_file_hash, find_unified_snapshot
from mb.utils.utils import (
    assign_still_convert_output_basenames,
    convert_output_jpeg_filename,
    plain_still_jpeg_basename,
)


def _rng_png(path: Path, seed: int = 42) -> None:
    """Large enough PNG that convert → JPEG still meets :data:`MIN_FILE_SIZE` for dataset."""
    rng = random.Random(seed)
    w = h = 512
    data = bytes(rng.randrange(256) for _ in range(w * h * 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (w, h), data).save(path, format="PNG")


def _write_jpeg(path: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes(
        "RGB",
        (200, 200),
        bytes(rng.randrange(256) for _ in range(200 * 200 * 3)),
    ).save(path, format="JPEG", quality=92)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _converted_jpeg_tree(conv_dir: Path) -> dict[str, str]:
    """Sorted basenames of ``*.jpg`` under *conv_dir* mapped to SHA-256 hex of file contents."""
    return {p.name: _sha256_file(p) for p in sorted(conv_dir.glob("*.jpg"))}


def test_convert_records_converted_paths_for_plain_stem_outputs(tmp_path: Path) -> None:
    """Still-image convert must populate ``converted`` so create-dataset can match by path (not stem-only)."""
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png = cdir / "photo.png"
    _rng_png(png)

    conv = ImageConverter(raw_data_dir=raw)
    assert conv.run(skip_space_check=True) is True

    rid = conv.run_id
    assert rid
    snap_path = next(raw.glob(f"snapshot_{rid}.json"))
    blob = json.loads(snap_path.read_text(encoding="utf-8"))
    assert len(blob["images"]) == 1
    rec = blob["images"][0]
    assert rec.get("converted") is not None
    out_dir = cdir / CONVERTED_MEDIA_SUBDIR
    expected_name = plain_still_jpeg_basename(png.stem)
    assert rec["converted"]["basename"] == expected_name
    rel = rec["converted"]["path"].replace("\\", "/")
    assert rel.endswith(f"solo/{CONVERTED_MEDIA_SUBDIR}/{expected_name}")


def test_convert_resume_skip_backfills_converted_metadata(tmp_path: Path) -> None:
    """Re-run with same run ID: skips refresh ``converted``; new sources still convert (``run()`` requires work)."""
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png_a = cdir / "photo.png"
    _rng_png(png_a, seed=1)

    first = ImageConverter(raw_data_dir=raw)
    assert first.run(skip_space_check=True) is True
    rid = first.run_id

    png_b = cdir / "other.png"
    _rng_png(png_b, seed=2)

    second = ImageConverter(raw_data_dir=raw)
    assert second.run(skip_space_check=True, run_id=rid) is True

    snap = find_unified_snapshot([raw], run_id=rid)
    assert snap is not None
    assert len(snap.images) == 2
    for rec in snap.images.values():
        assert rec.get("converted") is not None
        assert rec["converted"]["md5"]


def test_convert_then_dataset_preserves_single_snapshot_record(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png = cdir / "img.png"
    _rng_png(png, seed=99)

    conv = ImageConverter(raw_data_dir=raw)
    assert conv.run(skip_space_check=True) is True
    rid = conv.run_id

    out_jpg = cdir / CONVERTED_MEDIA_SUBDIR / plain_still_jpeg_basename(png.stem)
    assert out_jpg.stat().st_size >= MIN_FILE_SIZE

    data_dir = tmp_path / "data"
    ds = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=1,
        class_names=["solo"],
        run_id=rid,
        skip_space_check=True,
    )
    assert ds.run() is True

    # Dataset writes the updated snapshot under ``data_dir``; ``raw_data`` still has convert-only JSON.
    snap = find_unified_snapshot([data_dir, raw], run_id=rid)
    assert snap is not None
    assert len(snap.images) == 1
    rec = next(iter(snap.images.values()))
    assert rec.get("dataset") is not None
    assert rec["dataset"]["class"] == "solo"


def test_convert_second_run_all_skipped_returns_false(tmp_path: Path) -> None:
    """Pure resume: outputs already exist — no conversions; ``run()`` success flag is false (see convert implementation)."""
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png = cdir / "only.png"
    _rng_png(png, seed=7)

    first = ImageConverter(raw_data_dir=raw)
    assert first.run(skip_space_check=True) is True
    rid = first.run_id

    second = ImageConverter(raw_data_dir=raw)
    assert second.run(skip_space_check=True, run_id=rid) is False

    snap = find_unified_snapshot([raw], run_id=rid)
    assert snap is not None
    assert len(snap.images) == 1
    assert next(iter(snap.images.values())).get("converted") is not None


def test_after_create_dataset_calculate_file_hash_matches_snapshot_via_dataset_path(
    tmp_path: Path,
) -> None:
    """
    After create-dataset, train/test JPEGs are byte-identical to ``CONVERTED``; training uses
    :func:`~mb.utils.snapshot.calculate_file_hash` with *relative_path* matching ``dataset.path``
    to reuse ``converted.md5`` (see ``snapshot.py`` MD5 branch).
    """
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png = cdir / "img.png"
    _rng_png(png, seed=101)

    conv = ImageConverter(raw_data_dir=raw)
    assert conv.run(skip_space_check=True) is True
    rid = conv.run_id

    data_dir = tmp_path / "data"
    assert DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=1,
        class_names=["solo"],
        run_id=rid,
        skip_space_check=True,
    ).run() is True

    snap = find_unified_snapshot([data_dir, raw], run_id=rid)
    assert snap is not None
    rec = next(iter(snap.images.values()))
    ds_path = str(rec["dataset"]["path"]).replace("\\", "/")
    assert ds_path.startswith("train/") or ds_path.startswith("test/")
    disk_file = data_dir / Path(ds_path)
    assert disk_file.is_file()

    h = calculate_file_hash(
        disk_file,
        algorithm="md5",
        raw_data_dir=raw,
        unified_snapshot=snap,
        relative_path=ds_path,
        logger=None,
    )
    assert h == rec["converted"]["md5"]


def test_run_two_retries_new_sources_without_duplicating_run_one_outputs(tmp_path: Path) -> None:
    """
    Scenario similar to a large re-run: Run 1 produced four good ``CONVERTED`` JPEGs from
    ``a.jpg``, ``b.jpg``, ``c.png``, ``d.webp``; ``e.jpg`` was invalid (copy fails); ``a.png`` was not
    present yet (stands in for a source that had no output / pending retry).

    Run 2 uses the same ``run_id``. Per-class basenames come from
    :func:`~mb.utils.utils.assign_still_convert_output_basenames`: one plain ``{stem}.jpg`` per stem
    (JPEG sources preferred), hash-affixed names for additional same-stem sources. Skip/promote logic
    in ``ImageConverter.process_files`` avoids rewriting existing outputs.

    Before run 2 we take ``_converted_jpeg_tree`` (sorted basenames to SHA-256 of file bytes).
    After run 2 we require that tree to equal ``{**tree_before_run2, new_a_png: …, new_e: …}`` so the
    only changes are exactly those two new files; the four prior outputs are byte-identical.
    """
    raw = tmp_path / "raw_data"
    cls_dir = raw / "mixcls"
    conv_dir = cls_dir / CONVERTED_MEDIA_SUBDIR

    _write_jpeg(cls_dir / "a.jpg", seed=1)
    _write_jpeg(cls_dir / "b.jpg", seed=2)
    _rng_png(cls_dir / "c.png", seed=3)
    _rng_png(cls_dir / "d.webp", seed=4)
    (cls_dir / "e.jpg").write_bytes(b"not a jpeg")
    # a.png added only before run 2 (pending retry)

    run_one = ImageConverter(raw_data_dir=raw)
    assert run_one.run(skip_space_check=True) is True
    rid = run_one.run_id
    assert run_one.stats["files_copied"] + run_one.stats["files_converted"] == 4
    assert run_one.stats["files_skipped"] >= 1

    after_one = {p.name for p in conv_dir.glob("*.jpg")}
    assert len(after_one) == 4

    _rng_png(cls_dir / "a.png", seed=5)
    _write_jpeg(cls_dir / "e.jpg", seed=6)

    tree_before_run2 = _converted_jpeg_tree(conv_dir)
    assert list(tree_before_run2.keys()) == sorted(tree_before_run2.keys())
    assert len(tree_before_run2) == 4

    run_two = ImageConverter(raw_data_dir=raw)
    assert run_two.run(skip_space_check=True, run_id=rid) is True

    after_two = {p.name for p in conv_dir.glob("*.jpg")}
    assert after_one <= after_two
    assert len(after_two) == 6

    tree_after_run2 = _converted_jpeg_tree(conv_dir)
    assert list(tree_after_run2.keys()) == sorted(tree_after_run2.keys())
    assert len(tree_after_run2) == 6
    # Full directory state: nothing from run 1 may change on skip (same bytes).
    for name, digest in tree_before_run2.items():
        assert name in tree_after_run2
        assert tree_after_run2[name] == digest
    added_names = set(tree_after_run2) - set(tree_before_run2)
    assert len(added_names) == 2
    peers_r2 = [
        cls_dir / "a.jpg",
        cls_dir / "b.jpg",
        cls_dir / "c.png",
        cls_dir / "d.webp",
        cls_dir / "a.png",
        cls_dir / "e.jpg",
    ]
    m2 = assign_still_convert_output_basenames(peers_r2, output_dir=conv_dir)
    exp_a_png = m2[cls_dir / "a.png"]
    exp_e = m2[cls_dir / "e.jpg"]
    assert added_names == {exp_a_png, exp_e}
    # Full tree after run 2 is exactly run-1 bytes plus two new files (explicit merge).
    assert tree_after_run2 == {
        **tree_before_run2,
        exp_a_png: tree_after_run2[exp_a_png],
        exp_e: tree_after_run2[exp_e],
    }
    # Each source maps to exactly one deterministic basename — no second file for the same source.
    for stem in ("a.jpg", "b.jpg", "c.png", "d.webp", "a.png"):
        p = cls_dir / stem
        expected = m2[p]
        assert expected in after_two
        assert (conv_dir / expected).stat().st_size > 0
    assert exp_e in after_two

    assert run_two.stats["files_skipped"] >= 4
    assert run_two.stats["files_copied"] + run_two.stats["files_converted"] == 2

    snap = find_unified_snapshot([raw], run_id=rid)
    assert snap is not None
    with_converted = sum(1 for r in snap.images.values() if r.get("converted") is not None)
    assert with_converted == 6
    # ``e.jpg`` was invalid in run 1 then replaced: pre-conversion keys are MD5-of-file, so the old
    # failed-bytes row can remain alongside the new successful row (snapshot does not prune orphans).
    assert len(snap.images) in (6, 7)


def test_convert_resume_plain_stem_output_idempotent(tmp_path: Path) -> None:
    """Second convert with same ``run_id`` skips when ``CONVERTED/{stem}.jpg`` already exists."""
    raw = tmp_path / "raw_data"
    cls_dir = raw / "solo"
    conv_dir = cls_dir / CONVERTED_MEDIA_SUBDIR
    _write_jpeg(cls_dir / "u01.jpg", seed=11)

    first = ImageConverter(raw_data_dir=raw)
    assert first.run(skip_space_check=True) is True
    rid = first.run_id
    assert {p.name for p in conv_dir.glob("*.jpg")} == {"u01.jpg"}

    second = ImageConverter(raw_data_dir=raw)
    assert second.run(skip_space_check=True, run_id=rid) is False
    assert {p.name for p in conv_dir.glob("*.jpg")} == {"u01.jpg"}


def test_convert_promotes_legacy_hash_suffixed_file_to_plain_stem(tmp_path: Path) -> None:
    """
    If only the old hash-suffixed file exists for a sole source, rename it to plain ``{stem}.jpg``
    and sync the snapshot (no second JPEG).
    """
    raw = tmp_path / "raw_data"
    cls_dir = raw / "solo"
    conv_dir = cls_dir / CONVERTED_MEDIA_SUBDIR
    src = cls_dir / "u01.jpg"
    _write_jpeg(src, seed=11)
    legacy_name = convert_output_jpeg_filename(src, output_dir=conv_dir)
    assert legacy_name != "u01.jpg"
    conv_dir.mkdir(parents=True, exist_ok=True)
    jpeg_bytes = src.read_bytes()
    (conv_dir / legacy_name).write_bytes(jpeg_bytes)

    conv = ImageConverter(raw_data_dir=raw)
    assert conv.run(skip_space_check=True) is True
    assert conv.stats["files_promoted_to_plain"] == 1
    assert not (conv_dir / legacy_name).exists()
    assert (conv_dir / "u01.jpg").is_file()
    assert {p.name for p in conv_dir.glob("*.jpg")} == {"u01.jpg"}


def test_calculate_file_hash_returns_converted_md5_via_dataset_path(tmp_path: Path) -> None:
    """:func:`calculate_file_hash` matches ``dataset.path`` like training snapshot integration."""
    raw = tmp_path / "raw_data"
    snap = UnifiedSnapshot(run_id="t", raw_data_dir=str(raw))
    data_rel = "train/c/abc123.jpg"
    snap.images["origkey"] = {
        "original": {"hash": "origkey", "basename": "x.png", "path": "c/x.png", "format": ".png"},
        "converted": {
            "md5": "deadbeefcafe",
            "path": f"c/{CONVERTED_MEDIA_SUBDIR}/stem_hash.jpg",
            "basename": "stem_hash.jpg",
        },
        "dataset": {
            "class": "c",
            "path": data_rel,
            "basename": "abc123.jpg",
            "split": "train",
            "sha256": "x",
        },
        "training": None,
    }
    phantom = tmp_path / "data" / data_rel
    h = calculate_file_hash(
        phantom,
        algorithm="md5",
        unified_snapshot=snap,
        relative_path=data_rel,
        logger=None,
    )
    assert h == "deadbeefcafe"
