"""Tests for ``mb.evaluate.metrics`` and CLI ``evaluate metrics``."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from mb.cli import main
from mb.evaluate.metrics import build_metrics_request, run_evaluate_metrics, run_metrics
from mb.models.types import FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


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


@pytest.mark.requires_torch
def test_run_evaluate_metrics_pytorch_confusion_matrix_shape(tmp_path: Path) -> None:
    """
    Real (untrained) PyTorch inference through the split ``run_evaluate_metrics`` entry
    point used by the Evaluate GUI's confusion matrix viewer (not just ``run_metrics``
    directly), confirming it returns a well-formed square confusion matrix alongside a
    non-zero exit code contract.
    """
    torch = pytest.importorskip("torch")
    from PIL import Image

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    class_names = ["a", "b", "c"]
    n_per_class = 3
    data_dir = tmp_path / "data"
    for cls in class_names:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            Image.new("RGB", (32, 32), color=(i * 40 % 256, 0, 0)).save(cls_dir / f"{i}.jpg")

    torch.manual_seed(0)
    trainer = PyTorchTrainer(device="cpu")
    model = trainer.create_model("resnet18", num_classes=len(class_names), pretrained=False)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    ns = Namespace(
        model=model_path,
        data_dir=data_dir,
        model_type=ModelType.IMAGE_CLASSIFICATION.value,
        framework=FrameworkType.PYTORCH.value,
        architecture="resnet18",
        num_classes=len(class_names),
        image_size=32,
        batch_size=4,
        num_workers=0,
        device=None,
        dry_run=False,
        verbose=False,
    )

    code, report = run_evaluate_metrics(ns)
    assert code == 0
    assert report is not None
    n_total = len(class_names) * n_per_class
    assert report.n_samples == n_total
    assert report.class_names == class_names
    assert len(report.confusion_matrix) == len(class_names)
    assert all(len(row) == len(class_names) for row in report.confusion_matrix)
    assert sum(sum(row) for row in report.confusion_matrix) == n_total
    assert 0.0 <= report.accuracy_percent <= 100.0
