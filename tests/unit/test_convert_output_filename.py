"""Naming for CONVERTED outputs: stem + path-hash affix (see :mod:`mb.utils.utils`)."""

from __future__ import annotations

from pathlib import Path

from mb.utils.utils import convert_output_filename, convert_output_jpeg_filename


def test_convert_output_jpeg_filename_format(tmp_path: Path) -> None:
    p = tmp_path / "cls" / "photo.png"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    name = convert_output_jpeg_filename(p)
    assert name.endswith(".jpg")
    assert name.startswith("photo_")
    assert len(name) == len("photo_") + 8 + len(".jpg")


def test_convert_output_jpeg_filename_matches_convert_output_filename_jpg(tmp_path: Path) -> None:
    p = tmp_path / "cls" / "photo.png"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert convert_output_jpeg_filename(p) == convert_output_filename(p, output_suffix=".jpg")


def test_convert_output_jpeg_filename_differs_for_same_stem_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "c1" / "x.png"
    b = tmp_path / "c2" / "x.png"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    assert convert_output_jpeg_filename(a) != convert_output_jpeg_filename(b)


def test_convert_output_jpeg_filename_respects_single_component_length(tmp_path: Path) -> None:
    # Windows rejects a single path component >255 chars; use max legal stem for .png (251 chars).
    long_stem = "a" * 251
    p = tmp_path / f"{long_stem}.png"
    p.write_bytes(b"x")
    name = convert_output_jpeg_filename(p)
    assert name.endswith(".jpg")
    assert len(name) <= 255


def test_convert_output_filename_custom_suffix(tmp_path: Path) -> None:
    p = tmp_path / "cls" / "photo.png"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert convert_output_filename(p, output_suffix=".webp").endswith(".webp")
    assert convert_output_filename(p, output_suffix="png").endswith(".png")
