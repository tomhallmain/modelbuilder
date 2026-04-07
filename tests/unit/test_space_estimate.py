"""Unit tests for :mod:`mb.space_estimate`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.models.types import ModelType
from mb.space_estimate import (
    SpaceEstimateReport,
    check_convert_allowed,
    check_create_dataset_allowed,
    estimate_convert_additional_bytes,
    format_bytes,
    merge_convert_estimate_into_snapshot,
    run_convert_estimate,
    run_create_dataset_estimate,
    _fingerprint_files,
)
from mb.utils.snapshot import UnifiedSnapshot


def test_format_bytes() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024).endswith("KiB")
    assert format_bytes(1024 * 1024).endswith("MiB")
    assert format_bytes(-99) == "0 B"


def test_fingerprint_files_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"yy")
    base = tmp_path
    fp1 = _fingerprint_files([a, b], base)
    fp2 = _fingerprint_files([b, a], base)
    assert fp1 == fp2


def test_estimate_convert_additional_bytes_counts_files(tmp_path: Path) -> None:
    p = tmp_path / "s.png"
    p.write_bytes(b"\x00" * 1000)
    need, n, src = estimate_convert_additional_bytes([p], ModelType.IMAGE_CLASSIFICATION)
    assert n == 1
    assert src == 1000
    assert need >= 64 * 1024


def test_run_convert_estimate_empty_raw_dir(tmp_path: Path) -> None:
    r = run_convert_estimate(tmp_path, ModelType.IMAGE_CLASSIFICATION)
    assert r.operation == "convert"
    assert r.file_count == 0
    assert r.estimated_need_bytes == 64 * 1024
    assert r.free_bytes >= 0


def test_run_create_dataset_estimate_no_jpegs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    r = run_create_dataset_estimate(tmp_path, out)
    assert r.operation == "create_dataset"
    assert r.file_count == 0
    assert r.estimated_need_bytes == 0
    assert r.ok


def test_check_convert_allowed_skip_skips_heavy_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("run_convert_estimate should not run when skip_space_check is True")

    monkeypatch.setattr("mb.space_estimate.run_convert_estimate", _boom)
    ok, rep = check_convert_allowed(
        tmp_path, ModelType.IMAGE_CLASSIFICATION, skip_space_check=True
    )
    assert ok is True
    assert rep.fingerprint == ""
    assert "skipped" in rep.message.lower()


def test_check_create_dataset_allowed_skip_skips_heavy_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("run_create_dataset_estimate should not run when skip_space_check is True")

    monkeypatch.setattr("mb.space_estimate.run_create_dataset_estimate", _boom)
    out = tmp_path / "out"
    out.mkdir()
    ok, rep = check_create_dataset_allowed(tmp_path, out, skip_space_check=True)
    assert ok is True
    assert rep.fingerprint == ""
    assert "skipped" in rep.message.lower()


def test_merge_convert_estimate_into_snapshot() -> None:
    snap = UnifiedSnapshot("run1", "/raw")
    rep = SpaceEstimateReport(
        operation="convert",
        fingerprint="abc",
        estimated_need_bytes=100,
        free_bytes=200,
        target_path="/raw",
        ok=True,
        file_count=0,
        source_total_bytes=0,
        message="m",
        computed_at="t",
    )
    merge_convert_estimate_into_snapshot(snap, rep)
    assert snap.space_estimates is not None
    assert snap.space_estimates["convert"]["fingerprint"] == "abc"
    assert snap.space_estimates["convert"]["estimated_need_bytes"] == 100
