"""Unit tests for :mod:`mb.utils.snapshot` (unified snapshot + hashing helpers)."""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import pytest

from mb.utils import snapshot as snapshot_mod
from mb.utils.snapshot import (
    UnifiedSnapshot,
    calculate_file_hash,
    find_latest_unified_snapshot_path,
    find_unified_snapshot,
    generate_run_id,
    preload_gather_cache,
    save_unified_snapshot,
)


@pytest.fixture(autouse=True)
def reset_gather_cache() -> None:
    """Isolate tests that touch the global gather-cache."""
    snapshot_mod._gather_cache = None
    snapshot_mod._gather_cache_path = None
    yield
    snapshot_mod._gather_cache = None
    snapshot_mod._gather_cache_path = None


def test_generate_run_id_shape() -> None:
    rid = generate_run_id()
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", rid), rid


def test_calculate_file_hash_md5_and_sha256(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello-world")
    assert calculate_file_hash(p, algorithm="md5") == hashlib.md5(b"hello-world").hexdigest()
    assert calculate_file_hash(p, algorithm="sha256") == hashlib.sha256(b"hello-world").hexdigest()


def test_calculate_file_hash_missing_returns_none(tmp_path: Path) -> None:
    assert calculate_file_hash(tmp_path / "nope.bin", algorithm="md5") is None


def test_preload_gather_cache_with_file_uses_cache_for_md5(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    raw.mkdir()
    fake = raw / "a.jpg"
    fake.write_bytes(b"x")
    known_md5 = hashlib.md5(b"x").hexdigest()
    cache_path = raw / ".gather_cache.pkl"
    cache_path.write_bytes(pickle.dumps({str(fake.resolve()): known_md5}))

    assert preload_gather_cache(raw) is True
    assert calculate_file_hash(fake, algorithm="md5", raw_data_dir=raw) == known_md5


def test_unified_snapshot_save_load_roundtrip(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    raw.mkdir()
    snap = UnifiedSnapshot(run_id="unit_run", raw_data_dir=str(raw), data_dir=str(tmp_path / "data"))
    snap.images["deadbeef"] = {
        "original": {
            "basename": "a.jpg",
            "hash": "deadbeef",
            "path": "class_a/a.jpg",
            "format": ".jpg",
        },
        "converted": None,
        "dataset": None,
        "training": None,
    }
    snap.training_timing = {
        "version": 1,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "framework": "pytorch",
        "architecture": "resnet18",
        "model_type": "image_classification",
        "seconds": {"train": 1.0, "evaluate": 0.5, "total": 1.5},
    }
    out = tmp_path / "snap.json"
    assert snap.save(out)
    loaded = UnifiedSnapshot.load(out)
    assert loaded is not None
    assert loaded.run_id == "unit_run"
    assert loaded.raw_data_directory == str(raw)
    assert "deadbeef" in loaded.images
    assert loaded.images["deadbeef"]["original"]["basename"] == "a.jpg"
    assert loaded.training_timing is not None
    assert loaded.training_timing["seconds"]["total"] == 1.5
    assert loaded.to_dict().get("training_timing", {}).get("framework") == "pytorch"


def test_find_latest_unified_snapshot_path_prefers_newest_mtime(tmp_path: Path) -> None:
    d = tmp_path / "raw"
    d.mkdir()
    older = d / "snapshot_old.json"
    newer = d / "snapshot_new.json"
    older.write_text('{"run_id":"old"}', encoding="utf-8")
    newer.write_text('{"run_id":"new"}', encoding="utf-8")
    import os

    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    got = find_latest_unified_snapshot_path([d])
    assert got is not None and got.name == "snapshot_new.json"


def test_find_unified_snapshot_by_run_id(tmp_path: Path) -> None:
    d = tmp_path / "search"
    d.mkdir()
    s1 = UnifiedSnapshot(run_id="r1", raw_data_dir="raw")
    assert s1.save(d / "snapshot_r1.json")
    s2 = UnifiedSnapshot(run_id="r2", raw_data_dir="raw")
    assert s2.save(d / "snapshot_r2.json")

    got = find_unified_snapshot([d], run_id="r1")
    assert got is not None and got.run_id == "r1"


def test_find_unified_snapshot_latest_picks_lexicographically_last(tmp_path: Path) -> None:
    d = tmp_path / "search"
    d.mkdir()
    for rid in ("aaa", "zzz"):
        UnifiedSnapshot(run_id=rid, raw_data_dir="raw").save(d / f"snapshot_{rid}.json")

    got = find_unified_snapshot([d], run_id=None)
    assert got is not None and got.run_id == "zzz"


def test_save_unified_snapshot_returns_path(tmp_path: Path) -> None:
    snap = UnifiedSnapshot(run_id="rid", raw_data_dir="raw")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = save_unified_snapshot(snap, out_dir, logger=None)
    assert path is not None and path.name == "snapshot_rid.json" and path.exists()


def test_add_training_image_normalizes_path_separators() -> None:
    """Dataset paths are stored with ``/``; training scan may pass Windows ``\\`` paths."""
    snap = UnifiedSnapshot(run_id="x", raw_data_dir="raw")
    snap.images["k"] = {
        "original": {"hash": "k", "basename": "a.jpg", "path": "p", "format": ".jpg"},
        "converted": None,
        "dataset": {"path": "train/c/a.jpg", "split": "train", "class": "c"},
        "training": None,
    }
    snap.add_training_image("train", "c", r"train\c\a.jpg", "deadbeef", "a.jpg")
    tr = snap.images["k"].get("training")
    assert tr is not None
    assert tr["path"] == "train/c/a.jpg"


def test_unified_snapshot_to_dict_summary_counts() -> None:
    snap = UnifiedSnapshot(run_id="x", raw_data_dir="raw")
    snap.images["a"] = {
        "original": {"hash": "a", "basename": "1.jpg", "path": "p", "format": ".jpg"},
        "converted": {"md5": "m"},
        "dataset": {"path": "train/c/1.jpg", "split": "train"},
        "training": {"split": "train"},
    }
    d = snap.to_dict()
    assert d["summary"]["total_images"] == 1
    assert d["summary"]["converted_count"] == 1
    assert d["summary"]["dataset_count"] == 1
    assert d["summary"]["training_count"] == 1
