"""
CLI / library E2E: synthetic raw → dataset → ``mb.cli.main`` train → ``convert``.

See also ``tests/ui/test_ui_e2e_headless.py``: a **fast** shell smoke test plus
``test_headless_ui_train_pytorch_and_convert_onnx`` (same data hyperparameters,
train/convert driven from **Train** / **Convert** pages, not ``mb.cli.main``).

**Why a test might be skipped:** the body starts with ``pytest.importorskip`` for
``torch``, ``torchvision``, and ``onnx``. If any import fails, the whole test is
**skipped** (not failed). Install extras: ``pip install -r requirements-ml.txt``
or ``pip install -e ".[pytorch]"`` plus ``pip install onnx``.

First successful train run may download Torchvision pretrained weights (network).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from mb.data.class_layout import SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.data.dataset import DatasetCreator
from mb.cli import main

from tests.test_utils import default_pipeline_config_path, prepare_synthetic_raw_with_snapshot


_RS = "install PyTorch extras: pip install -e \".[pytorch]\" or see requirements-ml.txt"
_ONNX = "install ONNX for conversion: pip install onnx (see requirements-ml.txt)"


@pytest.mark.slow
@pytest.mark.e2e
def test_image_classification_train_pytorch_and_export_onnx(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason=_RS)
    pytest.importorskip("torchvision", reason=_RS)
    pytest.importorskip("onnx", reason=_ONNX)

    random.seed(42)
    raw = prepare_synthetic_raw_with_snapshot(tmp_path, total_images=100)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    onnx_path = tmp_path / "model_e2e.onnx"

    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_images_per_class=10,
    )
    assert creator.run() is True

    assert (data_dir / "train").is_dir() and (data_dir / "test").is_dir()
    for name in SYNTHETIC_DEFAULT_CLASS_NAMES:
        assert (data_dir / "train" / name).is_dir()
        assert (data_dir / "test" / name).is_dir()

    cfg = default_pipeline_config_path()
    assert cfg.is_file(), f"missing pipeline config: {cfg}"

    train_argv = [
        "--config",
        str(cfg),
        "train",
        "--framework",
        "pytorch",
        "--architecture",
        "resnet18",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(models_dir),
        "--frozen-epochs",
        "0",
        "--unfrozen-epochs",
        "1",
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--skip-snapshot-update",
    ]
    assert main(train_argv) == 0

    model_pth = models_dir / "resnet18_model.pth"
    assert model_pth.is_file() and model_pth.stat().st_size > 0

    convert_argv = [
        "--config",
        str(cfg),
        "convert",
        "--input",
        str(model_pth),
        "--output",
        str(onnx_path),
        "--framework",
        "pytorch",
        "--target",
        "onnx",
        "--architecture",
        "resnet18",
        "--num-classes",
        "3",
        "--image-size",
        "224",
    ]
    assert main(convert_argv) == 0
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0
