"""Tests for ``mb.evaluate.misclassified`` and CLI ``evaluate misclassified``."""

from __future__ import annotations

from argparse import Namespace

import pytest

from mb.cli import main
from mb.evaluate.metrics import build_metrics_request
from mb.evaluate.misclassified import run_misclassified
from mb.models.types import FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


def test_evaluate_misclassified_dry_run_ok_pytorch(tmp_path: Path) -> None:
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
                "misclassified",
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


def test_run_misclassified_object_detection_not_implemented(tmp_path: Path) -> None:
    model = tmp_path / "m2.pth"
    model.write_bytes(b"")
    data = tmp_path / "data2"
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
        run_misclassified(req)
