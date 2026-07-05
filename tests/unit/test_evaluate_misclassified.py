"""Tests for ``mb.evaluate.misclassified`` and CLI ``evaluate misclassified``."""

from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import pytest

from mb.cli import main
from mb.evaluate._contracts import MetricsRequest
from mb.evaluate.metrics import build_metrics_request
from mb.evaluate.misclassified import run_misclassified
from mb.models.types import FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


def _write_classified_images(data_dir: Path, class_names: list[str], n_per_class: int) -> None:
    from PIL import Image

    for cls in class_names:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            Image.new("RGB", (32, 32), color=(i * 40 % 256, 0, 0)).save(cls_dir / f"{i}.jpg")


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


@pytest.mark.requires_torch
def test_run_misclassified_pytorch_counts_and_cap(tmp_path: Path) -> None:
    """Real (untrained) PyTorch inference: totals stay stable while ``max_report`` only caps rows."""
    torch = pytest.importorskip("torch")

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    class_names = ["a", "b", "c"]
    n_per_class = 3
    data_dir = tmp_path / "data"
    _write_classified_images(data_dir, class_names, n_per_class)

    torch.manual_seed(0)
    trainer = PyTorchTrainer(device="cpu")
    model = trainer.create_model("resnet18", num_classes=len(class_names), pretrained=False)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    req = MetricsRequest(
        model_path=model_path,
        data_dir=data_dir,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.PYTORCH,
        architecture="resnet18",
        num_classes=len(class_names),
        image_size=32,
        batch_size=4,
    )

    n_total = len(class_names) * n_per_class
    listing = run_misclassified(req)
    assert listing.n_scanned == n_total
    assert 0 <= listing.n_misclassified <= n_total
    assert len(listing.samples) == listing.n_misclassified

    capped = run_misclassified(req, max_report=1)
    assert capped.n_scanned == n_total
    assert capped.n_misclassified == listing.n_misclassified
    assert len(capped.samples) == min(1, capped.n_misclassified)
    for s in capped.samples:
        assert Path(s.path).is_file()
        assert s.true_label in class_names
        assert s.predicted_label in class_names
        assert 0.0 <= s.confidence <= 1.0


@pytest.mark.requires_torch
def test_evaluate_misclassified_cli_writes_csv(tmp_path: Path) -> None:
    """Full CLI round trip with real inference: exit code, capped stdout rows, uncapped CSV totals."""
    torch = pytest.importorskip("torch")

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    class_names = ["cat", "dog"]
    n_per_class = 3
    data_dir = tmp_path / "data"
    _write_classified_images(data_dir, class_names, n_per_class)

    torch.manual_seed(0)
    trainer = PyTorchTrainer(device="cpu")
    model = trainer.create_model("resnet18", num_classes=len(class_names), pretrained=False)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    out_csv = tmp_path / "misclassified.csv"
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.EVALUATE.value),
                "misclassified",
                "--model",
                str(model_path),
                "--data-dir",
                str(data_dir),
                "--architecture",
                "resnet18",
                "--num-classes",
                str(len(class_names)),
                "--image-size",
                "32",
                "--batch-size",
                "4",
                "--max-report",
                "1",
                "--output",
                str(out_csv),
            ]
        )
        == 0
    )

    assert out_csv.is_file()
    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    assert header == ["path", "true_label", "predicted_label", "confidence"]
    # --max-report caps rows to at most 1, even though up to n_total images may be misclassified.
    assert len(body) <= 1
    for path, true_label, predicted_label, confidence in body:
        assert Path(path).is_file()
        assert true_label in class_names
        assert predicted_label in class_names
        assert 0.0 <= float(confidence) <= 1.0
