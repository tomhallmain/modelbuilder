"""Unit tests for :mod:`mb.info_inspect` (shared with CLI and Info page)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.info_inspect import dataset_info_text, model_info_text


def test_model_info_text_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError):
        model_info_text(missing)


def test_model_info_text_rejects_directory(tmp_path: Path) -> None:
    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        model_info_text(d)


def test_model_info_text_onnx_minimal(tmp_path: Path) -> None:
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"")
    text = model_info_text(onnx)
    assert str(onnx.resolve()) in text or onnx.name in text
    assert "onnx" in text.lower()


def test_dataset_info_text_two_class_layout(two_class_classification_data_dir: Path) -> None:
    text = dataset_info_text(two_class_classification_data_dir)
    assert "train" in text.lower()
    assert "class_a" in text or "class_b" in text


def test_dataset_info_text_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nodata"
    with pytest.raises(FileNotFoundError):
        dataset_info_text(missing)
