"""Tests for ``mb.evaluate.metrics`` and CLI ``evaluate metrics``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.cli import main
from mb.evaluate._weights import extract_pytorch_state_dict
from mb.evaluate.metrics import build_metrics_request, run_metrics
from mb.models.types import FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


def test_extract_pytorch_state_dict_plain_state_dict() -> None:
    pytest.importorskip("torch")
    import torch

    sd = {"layer.weight": torch.zeros(1)}
    assert extract_pytorch_state_dict(sd) is sd


def test_extract_pytorch_state_dict_nested() -> None:
    pytest.importorskip("torch")
    import torch

    inner = {"w": torch.ones(2)}
    blob = {"model_state_dict": inner, "epoch": 3}
    assert extract_pytorch_state_dict(blob) is inner


def test_evaluate_metrics_dry_run_requires_arch_for_pytorch(tmp_path: Path) -> None:
    model = tmp_path / "w.pt"
    model.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.EVALUATE.value),
                "metrics",
                "--dry-run",
                "--model",
                str(model),
                "--data-dir",
                str(data),
            ]
        )
        == 1
    )


def test_evaluate_metrics_dry_run_ok_pytorch(tmp_path: Path) -> None:
    model = tmp_path / "w.pt"
    model.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.EVALUATE.value),
                "metrics",
                "--dry-run",
                "--model",
                str(model),
                "--data-dir",
                str(data),
                "--architecture",
                "resnet18",
            ]
        )
        == 0
    )


def test_run_metrics_object_detection_not_implemented(tmp_path: Path) -> None:
    from argparse import Namespace

    model = tmp_path / "m.pth"
    model.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    (data / "a").mkdir()
    ns = Namespace(
        model=model,
        data_dir=data,
        model_type=ModelType.OBJECT_DETECTION.value,
        framework=FrameworkType.PYTORCH.value,
        architecture="resnet18",
        num_classes=1,
        image_size=224,
        batch_size=2,
        num_workers=0,
        device=None,
    )
    req = build_metrics_request(ns)
    with pytest.raises(NotImplementedError):
        run_metrics(req)
