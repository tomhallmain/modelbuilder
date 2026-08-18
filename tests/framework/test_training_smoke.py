"""
Smoke: :class:`~mb.training.trainer.ModelTrainer.train` on tiny two-class folders.

Requires optional frameworks; use ``-m "not slow"`` to skip one-epoch runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.models.types import ArchitectureType, FrameworkType, LabelMode, ModelType
from mb.pipeline_config import PipelineConfig
from mb.training.run_args import TrainingRunArgs
from mb.training.trainer import ModelTrainer

from tests.fixtures.pipeline_image_size import HIGH_RES_PIPELINE_IMAGE_SIZE


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
@pytest.mark.requires_torch
def test_model_trainer_pytorch_one_epoch_cpu_smoke_high_res_image_size(
    two_class_classification_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One epoch at ``HIGH_RES_PIPELINE_IMAGE_SIZE`` (>300px) to match fine-tuning above 224 baselines."""
    pytest.importorskip("torch", reason="PyTorch smoke test")
    pytest.importorskip("torchvision", reason="PyTorch smoke test")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    out_dir = tmp_path / "models_high_res"
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
            "batch_size": 1,
            "num_workers": 0,
            "image_size": HIGH_RES_PIPELINE_IMAGE_SIZE,
        },
    )
    model_path = trainer.train(run_args)
    assert model_path.is_file()
    assert model_path.stat().st_size > 0
    assert model_path.suffix == ".pth"


def _force_random_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make ``create_model`` ignore ``pretrained=True`` for the duration of a test.

    ``ModelTrainer.train`` always asks for pretrained weights, which is right in production
    and wrong here: a Hugging Face backbone would download hundreds of megabytes from the
    hub, making the suite network-dependent and slow. The trained weights are irrelevant to
    a smoke test — only that the loop runs end to end.
    """
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    original = PyTorchTrainer.create_model

    def _random_init(self, architecture, num_classes, pretrained=True, **kwargs):
        return original(self, architecture, num_classes, pretrained=False, **kwargs)

    monkeypatch.setattr(PyTorchTrainer, "create_model", _random_init)


@pytest.mark.slow
@pytest.mark.requires_torch
@pytest.mark.requires_transformers
def test_model_trainer_siglip2_one_epoch_cpu_smoke(
    two_class_classification_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    One epoch through a Hugging Face backbone.

    Covers what the unit tests cannot: that the adapter's logits tensor survives the loss,
    the frozen phase finds a head to unfreeze via ``head_parameters()``, and the resulting
    state dict saves. The architecture is resolution-locked, so the image size has to match
    its native 224 or training is rejected before it starts.
    """
    pytest.importorskip("torch", reason="PyTorch smoke test")
    pytest.importorskip("torchvision", reason="PyTorch smoke test")
    pytest.importorskip("transformers", reason="Hugging Face backbone smoke test")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _force_random_init(monkeypatch)

    out_dir = tmp_path / "models_siglip2"
    trainer = ModelTrainer(
        framework=FrameworkType.PYTORCH,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=PipelineConfig(config_path=None),
    )
    run_args = TrainingRunArgs(
        framework=FrameworkType.PYTORCH,
        architecture=ArchitectureType.SIGLIP2_BASE_PATCH16_224,
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

    # Saved through the adapter, so the keys carry its submodule prefix — which is what the
    # exported bundle stub keys its transformers layout off.
    state = torch.load(model_path, map_location="cpu")
    assert any(key.startswith("hf.") for key in state)


@pytest.mark.slow
@pytest.mark.requires_torch
@pytest.mark.requires_transformers
def test_siglip2_rejects_a_mismatched_image_size(
    two_class_classification_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Training at the wrong resolution fails up front rather than silently degrading.

    The rejection happens while resolving hyperparameters, before any model is built, so
    this never reaches the hub even though it asks for pretrained weights.
    """
    pytest.importorskip("torch", reason="PyTorch smoke test")
    pytest.importorskip("transformers", reason="Hugging Face backbone smoke test")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    trainer = ModelTrainer(
        framework=FrameworkType.PYTORCH,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=PipelineConfig(config_path=None),
    )
    run_args = TrainingRunArgs(
        framework=FrameworkType.PYTORCH,
        architecture=ArchitectureType.SIGLIP2_BASE_PATCH16_224,
        data_dir=two_class_classification_data_dir,
        output_dir=tmp_path / "models_bad_size",
        resume_from=None,
        run_id=None,
        update_snapshot=False,
        cli_hyperparams={
            "frozen_epochs": 1,
            "unfrozen_epochs": 0,
            "batch_size": 2,
            "num_workers": 0,
            "image_size": 320,
        },
    )
    with pytest.raises(ValueError):
        trainer.train(run_args)


@pytest.mark.slow
@pytest.mark.requires_torch
def test_model_trainer_multi_label_one_epoch_cpu_smoke(
    multi_label_classification_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    One epoch of multi-label training end to end.

    Covers the whole chain the unit tests stop short of: schema and manifest loading,
    multi-hot targets, BCE with per-label ``pos_weight``, the F1 epoch metric, and the
    post-training evaluation pass that would otherwise apply cross-entropy to float
    targets.
    """
    pytest.importorskip("torch", reason="PyTorch smoke test")
    pytest.importorskip("torchvision", reason="PyTorch smoke test")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    out_dir = tmp_path / "models_multilabel"
    trainer = ModelTrainer(
        framework=FrameworkType.PYTORCH,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=PipelineConfig(config_path=None),
    )
    run_args = TrainingRunArgs(
        framework=FrameworkType.PYTORCH,
        architecture=ArchitectureType.RESNET18,
        data_dir=multi_label_classification_data_dir,
        output_dir=out_dir,
        resume_from=None,
        run_id=None,
        update_snapshot=False,
        cli_hyperparams={
            "frozen_epochs": 1,
            "unfrozen_epochs": 0,
            "batch_size": 2,
            "num_workers": 0,
            "image_size": 64,
            "class_weighting": "inverse_frequency",
        },
        label_mode=LabelMode.MULTI_LABEL,
    )
    model_path = trainer.train(run_args)
    assert model_path.is_file()

    # Three outputs, from the schema — the folders only describe two. A model sized from
    # directories would be silently unable to ever predict the third label.
    state = torch.load(model_path, map_location="cpu")
    assert state["fc.weight"].shape[0] == 3

    # The checkpoint records which metric its stored score refers to.
    metadata = sorted(out_dir.glob("checkpoint_epoch_*.json"))
    assert metadata
    assert json.loads(metadata[-1].read_text(encoding="utf-8"))["primary_metric"] == "micro_f1"


@pytest.mark.slow
@pytest.mark.requires_torch
def test_single_label_training_still_records_accuracy(
    two_class_classification_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The default path is unchanged by multi-label support.

    Pairs with the multi-label smoke test above: same loop, same fixture shape, and the
    metric recorded in the checkpoint still says accuracy.
    """
    pytest.importorskip("torch", reason="PyTorch smoke test")
    pytest.importorskip("torchvision", reason="PyTorch smoke test")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    out_dir = tmp_path / "models_single_label"
    trainer = ModelTrainer(
        framework=FrameworkType.PYTORCH,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=PipelineConfig(config_path=None),
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
            "image_size": 64,
        },
    )
    model_path = trainer.train(run_args)
    assert model_path.is_file()

    state = torch.load(model_path, map_location="cpu")
    assert state["fc.weight"].shape[0] == 2

    metadata = sorted(out_dir.glob("checkpoint_epoch_*.json"))
    assert metadata
    payload = json.loads(metadata[-1].read_text(encoding="utf-8"))
    assert payload["primary_metric"] == "accuracy"
    assert payload["class_weights"] is None


@pytest.mark.requires_tf
def test_keras_rejects_multi_label_rather_than_training_softmax(
    multi_label_classification_data_dir: Path,
    tmp_path: Path,
) -> None:
    """
    Multi-label is PyTorch-only, and asking for it under Keras must fail loudly.

    Silently falling back would produce a softmax model that looks trained and scores
    nonsense against a multi-label schema. Not marked slow: it raises before any epoch.
    """
    pytest.importorskip("tensorflow", reason="Keras smoke test")

    trainer = ModelTrainer(
        framework=FrameworkType.KERAS,
        model_type=ModelType.IMAGE_CLASSIFICATION,
        pipeline_config=PipelineConfig(config_path=None),
    )
    run_args = TrainingRunArgs(
        framework=FrameworkType.KERAS,
        architecture=ArchitectureType.RESNET50,
        data_dir=multi_label_classification_data_dir,
        output_dir=tmp_path / "models_keras_multilabel",
        resume_from=None,
        run_id=None,
        update_snapshot=False,
        cli_hyperparams={"frozen_epochs": 1, "unfrozen_epochs": 0, "num_workers": 0},
        label_mode=LabelMode.MULTI_LABEL,
    )
    with pytest.raises(ValueError):
        trainer.train(run_args)


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
