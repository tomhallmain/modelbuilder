"""
Smoke: :class:`~mb.training.trainer.ModelTrainer.train` on tiny two-class folders.

Requires optional frameworks; use ``-m "not slow"`` to skip one-epoch runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.models.types import ArchitectureType, FrameworkType, ModelType
from mb.pipeline_config import PipelineConfig
from mb.training.run_args import TrainingRunArgs
from mb.training.trainer import ModelTrainer


@pytest.mark.slow
@pytest.mark.requires_torch
def test_model_trainer_pytorch_one_epoch_cpu_smoke(
    two_class_classification_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch", reason="PyTorch smoke test")
    pytest.importorskip("torchvision", reason="PyTorch smoke test")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    out_dir = tmp_path / "models"
    pipeline = PipelineConfig(config_path=None)
    trainer = ModelTrainer(
        framework=FrameworkType.PYTORCH,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=pipeline,
    )
    run_args = TrainingRunArgs(
        framework=FrameworkType.PYTORCH,
        architecture=ArchitectureType.RESNET18,
        data_dir=two_class_classification_data_dir,
        output_dir=out_dir,
        resume_from=None,
        run_id=None,
        update_snapshot=False,
        cli_hyperparams={
            "frozen_epochs": 1,
            "unfrozen_epochs": 0,
            "batch_size": 2,
            "num_workers": 0,
            "image_size": 224,
        },
    )
    model_path = trainer.train(run_args)
    assert model_path.is_file()
    assert model_path.stat().st_size > 0
    assert model_path.suffix == ".pth"


@pytest.mark.slow
@pytest.mark.requires_tf
def test_model_trainer_keras_one_epoch_smoke(
    two_class_classification_data_dir: Path,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tensorflow", reason="Keras smoke test")

    out_dir = tmp_path / "models"
    pipeline = PipelineConfig(config_path=None)
    trainer = ModelTrainer(
        framework=FrameworkType.KERAS,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=pipeline,
    )
    run_args = TrainingRunArgs(
        framework=FrameworkType.KERAS,
        architecture=ArchitectureType.RESNET50,
        data_dir=two_class_classification_data_dir,
        output_dir=out_dir,
        resume_from=None,
        run_id=None,
        update_snapshot=False,
        cli_hyperparams={
            "frozen_epochs": 1,
            "unfrozen_epochs": 0,
            "batch_size": 2,
            "num_workers": 0,
            "image_size": 224,
        },
    )
    model_path = trainer.train(run_args)
    assert model_path.is_file()
    assert model_path.stat().st_size > 0
    assert model_path.suffix == ".h5"
