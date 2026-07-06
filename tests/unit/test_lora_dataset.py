"""Tests for ``mb.data.lora_dataset`` and ``mb data create-dataset --model-type image_generation_lora``."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mb.cli import main
from mb.data.lora_captions import read_caption, write_caption
from mb.data.lora_dataset import LoraDatasetCreator
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


def _write_image(path: Path, color: tuple[int, int, int], fmt: str = "JPEG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path, format=fmt)


def test_lora_dataset_creator_flat_copy_with_captions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_image(raw / "a.jpg", (10, 0, 0))
    write_caption(raw / "a.jpg", "a photo of a red square")
    _write_image(raw / "b.jpg", (0, 10, 0))
    # b.jpg has no caption sidecar — should still be copied, just uncaptioned.

    data_dir = tmp_path / "data"
    creator = LoraDatasetCreator(raw_data_dir=raw, data_dir=data_dir)
    assert creator.run() is True

    report = creator.report
    assert report is not None
    assert report.n_scanned == 2
    assert report.n_copied == 2
    assert report.n_captioned == 1
    assert report.n_skipped == 0

    # Flat output: no class folders, no train/test split.
    assert data_dir.is_dir()
    assert not (data_dir / "train").exists()
    assert not (data_dir / "test").exists()
    output_images = sorted(data_dir.glob("*.jpg"))
    assert len(output_images) == 2

    captions = [read_caption(p) for p in output_images]
    assert "a photo of a red square" in captions
    assert captions.count(None) == 1


def test_lora_dataset_creator_normalizes_non_jpeg(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_image(raw / "c.png", (0, 0, 10), fmt="PNG")

    data_dir = tmp_path / "data"
    creator = LoraDatasetCreator(raw_data_dir=raw, data_dir=data_dir)
    assert creator.run() is True
    assert creator.report.n_copied == 1
    output_images = list(data_dir.glob("*.jpg"))
    assert len(output_images) == 1


def test_lora_dataset_creator_empty_source_dir_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    data_dir = tmp_path / "data"
    creator = LoraDatasetCreator(raw_data_dir=raw, data_dir=data_dir)
    assert creator.run() is False
    assert creator.report is None


def test_lora_dataset_creator_missing_source_dir_fails(tmp_path: Path) -> None:
    creator = LoraDatasetCreator(raw_data_dir=tmp_path / "nope", data_dir=tmp_path / "data")
    assert creator.run() is False


def test_cli_create_dataset_image_generation_lora_flat_copy(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_image(raw / "x.jpg", (5, 5, 5))
    write_caption(raw / "x.jpg", "a test caption")
    data_dir = tmp_path / "data"

    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.DATA.value),
                "create-dataset",
                "--raw-data-dir",
                str(raw),
                "--data-dir",
                str(data_dir),
                "--model-type",
                "image_generation_lora",
            ]
        )
        == 0
    )
    assert not (data_dir / "train").exists()
    assert len(list(data_dir.glob("*.jpg"))) == 1
