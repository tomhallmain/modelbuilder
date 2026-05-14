"""Tests for ``mb.evaluate.compare`` and CLI ``evaluate compare``."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from mb.cli import main
from mb.evaluate.compare import build_compare_request, run_compare
from mb.models.types import FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


def test_evaluate_compare_dry_run_requires_arch_for_pytorch_a(tmp_path: Path) -> None:
    ma = tmp_path / "a.pt"
    mb = tmp_path / "b.pt"
    ma.write_bytes(b"")
    mb.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.EVALUATE.value),
                "compare",
                "--dry-run",
                "--model-a",
                str(ma),
                "--model-b",
                str(mb),
                "--data-dir",
                str(data),
            ]
        )
        == 1
    )


def test_evaluate_compare_dry_run_ok_pytorch(tmp_path: Path) -> None:
    ma = tmp_path / "a.pt"
    mb = tmp_path / "b.pt"
    ma.write_bytes(b"")
    mb.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.EVALUATE.value),
                "compare",
                "--dry-run",
                "--model-a",
                str(ma),
                "--model-b",
                str(mb),
                "--data-dir",
                str(data),
                "--architecture",
                "resnet18",
            ]
        )
        == 0
    )


def test_run_compare_object_detection_not_implemented(tmp_path: Path) -> None:
    ma = tmp_path / "m.pth"
    mb = tmp_path / "m2.pth"
    ma.write_bytes(b"")
    mb.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    (data / "a").mkdir()
    ns = Namespace(
        model_a=ma,
        model_b=mb,
        data_dir=data,
        model_type=ModelType.OBJECT_DETECTION.value,
        framework=FrameworkType.PYTORCH.value,
        framework_b=None,
        architecture="resnet18",
        architecture_b=None,
        num_classes=1,
        image_size=224,
        batch_size=2,
        num_workers=0,
        device=None,
        max_disagreement_report=None,
    )
    req = build_compare_request(ns)
    with pytest.raises(NotImplementedError):
        run_compare(req)
