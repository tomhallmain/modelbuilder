"""
Tests for ``ui.controllers.train_page_controller`` — pure Python, no Qt/PySide6 needed,
since the whole point of extracting this out of ``TrainPage`` was to make the
argv/config-building logic testable without a running GUI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.models.types import ModelType
from mb.training.lora_diffusion_trainer import LoraTrainingConfig
from mb.training.run_args import TrainingRunArgs
from ui.controllers.train_page_controller import TrainPageFieldValues, build_training_request


def _classification_values(**overrides) -> TrainPageFieldValues:
    defaults = dict(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework_text="pytorch",
        architecture_text="resnet34",
        data_dir_text=".",
        output_dir_text="data/models",
        batch_size=0,
        image_size=224,
        num_workers=0,
        resume_from_text="",
        run_id_text="",
        skip_snapshot=False,
        frozen_epochs=5,
        unfrozen_epochs=20,
        frozen_lr=0.001,
        unfrozen_lr_max=0.0003,
        unfrozen_lr_min=0.00001,
    )
    defaults.update(overrides)
    return TrainPageFieldValues(**defaults)


def _lora_values(**overrides) -> TrainPageFieldValues:
    defaults = dict(
        model_type=ModelType.IMAGE_GENERATION_LORA,
        framework_text="",
        architecture_text="some-base-model",
        data_dir_text=".",
        output_dir_text="data/models",
        batch_size=1,
        image_size=512,
        num_workers=0,
        base_model_architecture_text="stable_diffusion_1x",
        lora_rank=16,
        lora_alpha=0,
        learning_rate=1e-4,
        max_train_steps=1000,
        seed_text="",
    )
    defaults.update(overrides)
    return TrainPageFieldValues(**defaults)


def test_classification_builds_training_run_args() -> None:
    request = build_training_request(_classification_values())
    assert isinstance(request, TrainingRunArgs)
    assert request.framework.value == "pytorch"
    assert request.architecture.value == "resnet34"
    assert request.data_dir == Path(".")
    assert request.cli_hyperparams["frozen_epochs"] == 5
    assert "batch_size" not in request.cli_hyperparams  # 0 means "auto", omitted


def test_classification_missing_architecture_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_classification_values(architecture_text=""))


def test_classification_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_classification_values(architecture_text="not_a_real_arch"))


def test_classification_missing_data_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_training_request(
            _classification_values(data_dir_text=str(tmp_path / "does_not_exist"))
        )


def test_classification_unsupported_framework_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_classification_values(framework_text="not_a_framework"))


def test_lora_builds_config_with_explicit_architecture() -> None:
    request = build_training_request(_lora_values())
    assert isinstance(request, LoraTrainingConfig)
    assert request.base_model == "some-base-model"
    assert request.base_architecture.value == "stable_diffusion_1x"
    assert request.rank == 16
    assert request.alpha == 16  # 0 means "same as rank"
    assert request.resolution == 512
    assert request.max_train_steps == 1000


def test_lora_alpha_explicit_value_kept() -> None:
    request = build_training_request(_lora_values(lora_alpha=32))
    assert isinstance(request, LoraTrainingConfig)
    assert request.alpha == 32


def test_lora_missing_base_model_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_lora_values(architecture_text=""))


def test_lora_rejects_non_pytorch_framework() -> None:
    with pytest.raises(ValueError):
        build_training_request(_lora_values(framework_text="keras"))


def test_lora_missing_data_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_training_request(
            _lora_values(data_dir_text=str(tmp_path / "does_not_exist"))
        )


def test_lora_unimplemented_base_architecture_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_lora_values(base_model_architecture_text="stable_diffusion_xl"))


def test_lora_undetectable_architecture_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_lora_values(base_model_architecture_text=""))


def test_lora_invalid_seed_raises() -> None:
    with pytest.raises(ValueError):
        build_training_request(_lora_values(seed_text="not_an_int"))


def test_lora_valid_seed_parsed() -> None:
    request = build_training_request(_lora_values(seed_text="42"))
    assert isinstance(request, LoraTrainingConfig)
    assert request.seed == 42
