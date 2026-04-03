"""Unit tests for :mod:`ui.lib.fast_directory_picker_qt` helpers."""

from __future__ import annotations

from pathlib import Path

from ui.lib.fast_directory_picker_qt import list_matching_files, parse_qt_name_filter


def test_parse_qt_name_filter_all_files() -> None:
    assert parse_qt_name_filter("") is None
    assert parse_qt_name_filter("All files (*.*)") is None
    assert parse_qt_name_filter("x (*.*)") is None


def test_parse_qt_name_filter_extensions() -> None:
    f = "YAML (*.yaml *.yml);;All files (*.*)"
    exts = parse_qt_name_filter(f)
    assert exts is None


def test_parse_qt_name_filter_extensions_only() -> None:
    exts = parse_qt_name_filter("Models (*.pth *.pt)")
    assert exts == [".pt", ".pth"]


def test_list_matching_files_filters(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.pth").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    got = list_matching_files(str(tmp_path), [".pth"])
    assert [n for _, n in got] == ["b.pth"]


def test_list_matching_files_all(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("z")
    got = list_matching_files(str(tmp_path), None)
    names = sorted(n for _, n in got)
    assert names == ["z.txt"]
