"""Unit tests for :mod:`mb.data.class_layout`, especially the discovery cache."""

from __future__ import annotations

from pathlib import Path

import mb.data.class_layout as class_layout
from mb.data.class_layout import (
    DISCOVERY_CACHE_MAX_AGE_SECONDS,
    clear_class_discovery_cache,
    discover_class_names,
)


def _make_class_dirs(root: Path, *names: str) -> None:
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


def test_discover_class_names_implicit_sorted(tmp_path: Path) -> None:
    clear_class_discovery_cache()
    _make_class_dirs(tmp_path, "b", "a")
    (tmp_path / "not_a_dir.txt").write_text("x")
    assert discover_class_names(tmp_path) == ["a", "b"]


def test_discover_class_names_explicit_filters_missing_and_preserves_order(tmp_path: Path) -> None:
    clear_class_discovery_cache()
    _make_class_dirs(tmp_path, "a", "b")
    names = discover_class_names(tmp_path, explicit=["b", "missing", "a"])
    assert names == ["b", "a"]


def test_discover_class_names_repeat_call_reuses_cache_regardless_of_elapsed_time(
    tmp_path: Path, monkeypatch
) -> None:
    """A second call for the same root/args reuses the cache even if a lot of real time
    passes in between (e.g. slow unrelated work on another thread) -- as long as it
    stays under the staleness safety net."""
    clear_class_discovery_cache()
    fake_now = [1000.0]
    monkeypatch.setattr(class_layout.time, "monotonic", lambda: fake_now[0])

    _make_class_dirs(tmp_path, "a")
    assert discover_class_names(tmp_path) == ["a"]

    # New class folder appears, but nothing has invalidated the cache.
    _make_class_dirs(tmp_path, "b")
    fake_now[0] += DISCOVERY_CACHE_MAX_AGE_SECONDS / 2
    assert discover_class_names(tmp_path) == ["a"]


def test_discover_class_names_rescans_after_safety_net_age_exceeded(
    tmp_path: Path, monkeypatch
) -> None:
    clear_class_discovery_cache()
    fake_now = [2000.0]
    monkeypatch.setattr(class_layout.time, "monotonic", lambda: fake_now[0])

    _make_class_dirs(tmp_path, "a")
    assert discover_class_names(tmp_path) == ["a"]

    _make_class_dirs(tmp_path, "b")
    fake_now[0] += DISCOVERY_CACHE_MAX_AGE_SECONDS + 0.1
    assert discover_class_names(tmp_path) == ["a", "b"]


def test_clear_class_discovery_cache_forces_rescan(tmp_path: Path, monkeypatch) -> None:
    clear_class_discovery_cache()
    fake_now = [3000.0]
    monkeypatch.setattr(class_layout.time, "monotonic", lambda: fake_now[0])

    _make_class_dirs(tmp_path, "a")
    assert discover_class_names(tmp_path) == ["a"]

    _make_class_dirs(tmp_path, "b")
    clear_class_discovery_cache()
    assert discover_class_names(tmp_path) == ["a", "b"]


def test_discover_class_names_cache_keys_are_distinct_per_args(tmp_path: Path, monkeypatch) -> None:
    """Different explicit/qualifier args for the same root must not share a cache entry."""
    clear_class_discovery_cache()
    fake_now = [4000.0]
    monkeypatch.setattr(class_layout.time, "monotonic", lambda: fake_now[0])

    _make_class_dirs(tmp_path, "a", "b")
    assert discover_class_names(tmp_path) == ["a", "b"]
    assert discover_class_names(tmp_path, explicit=["a"]) == ["a"]
    # The implicit-discovery cache entry from above must be untouched.
    assert discover_class_names(tmp_path) == ["a", "b"]
