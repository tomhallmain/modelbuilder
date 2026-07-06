"""
Tests for the CLI-plumbing/validation parts of ``mb.training.lora_diffusion_trainer`` that
don't require the optional diffusers/peft/transformers dependencies — those are only
imported lazily inside :func:`~mb.training.lora_diffusion_trainer.train_image_generation_lora`,
which none of these tests reach.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import pytest

from mb.cli import main
from mb.models.generation_architectures import BaseGenerationArchitecture
from mb.training.lora_diffusion_trainer import (
    build_lora_training_config_from_args,
    run_train_image_generation_lora_cli,
)
from mb.utils.constants import ModelBuilderTaskType

from tests.test_utils import default_pipeline_config_path


class _FakePipeline:
    def __init__(self, values: Optional[dict] = None) -> None:
        self._values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


def _base_namespace(**overrides: Any) -> Namespace:
    ns = Namespace(
        architecture="sd15-base",
        base_model_architecture="stable_diffusion_1x",
        data_dir=None,
        output_dir=None,
        lora_rank=None,
        lora_alpha=None,
        image_size=None,
        batch_size=None,
        learning_rate=None,
        max_train_steps=None,
        num_workers=None,
        seed=None,
        framework=None,
        verbose=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_build_lora_training_config_defaults() -> None:
    config = build_lora_training_config_from_args(_base_namespace(), _FakePipeline())
    assert config.base_model == "sd15-base"
    assert config.base_architecture == BaseGenerationArchitecture.STABLE_DIFFUSION_1X
    assert config.data_dir == Path("data")
    assert config.output_dir == Path("data/models")
    assert config.rank == 16
    assert config.alpha == 16
    assert config.resolution == 512
    assert config.batch_size == 1
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.max_train_steps == 1000
    assert config.num_workers == 0
    assert config.seed is None


def test_build_lora_training_config_overrides() -> None:
    ns = _base_namespace(
        data_dir=Path("/tmp/lora_data"),
        output_dir=Path("/tmp/lora_out"),
        lora_rank=8,
        lora_alpha=32,
        image_size=768,
        batch_size=4,
        learning_rate=5e-5,
        max_train_steps=2000,
        num_workers=2,
        seed=123,
    )
    config = build_lora_training_config_from_args(ns, _FakePipeline())
    assert config.data_dir == Path("/tmp/lora_data")
    assert config.output_dir == Path("/tmp/lora_out")
    assert config.rank == 8
    assert config.alpha == 32
    assert config.resolution == 768
    assert config.batch_size == 4
    assert config.learning_rate == pytest.approx(5e-5)
    assert config.max_train_steps == 2000
    assert config.num_workers == 2
    assert config.seed == 123


def test_build_lora_training_config_alpha_defaults_to_rank() -> None:
    config = build_lora_training_config_from_args(
        _base_namespace(lora_rank=32), _FakePipeline()
    )
    assert config.rank == 32
    assert config.alpha == 32


def test_build_lora_training_config_missing_base_model_raises() -> None:
    ns = _base_namespace(architecture=None)
    with pytest.raises(ValueError):
        build_lora_training_config_from_args(ns, _FakePipeline())


def test_build_lora_training_config_falls_back_to_pipeline_architecture() -> None:
    ns = _base_namespace(architecture=None)
    config = build_lora_training_config_from_args(
        ns, _FakePipeline({"model.default_architecture": "sdxl-base"})
    )
    assert config.base_model == "sdxl-base"


def test_build_lora_training_config_explicit_architecture_skips_detection(tmp_path: Path) -> None:
    """An explicit --base-model-architecture wins even for a base_model path with no model_index.json."""
    ns = _base_namespace(
        architecture=str(tmp_path / "not_a_real_checkpoint"),
        base_model_architecture="stable_diffusion_xl",
    )
    config = build_lora_training_config_from_args(ns, _FakePipeline())
    assert config.base_architecture == BaseGenerationArchitecture.STABLE_DIFFUSION_XL


def test_build_lora_training_config_detects_architecture_from_model_index(tmp_path: Path) -> None:
    base_model_dir = tmp_path / "checkpoint"
    base_model_dir.mkdir()
    (base_model_dir / "model_index.json").write_text(
        '{"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"], '
        '"text_encoder": ["transformers", "CLIPTextModel"]}',
        encoding="utf-8",
    )
    ns = _base_namespace(architecture=str(base_model_dir), base_model_architecture=None)
    config = build_lora_training_config_from_args(ns, _FakePipeline())
    assert config.base_architecture == BaseGenerationArchitecture.STABLE_DIFFUSION_1X


def test_build_lora_training_config_undetectable_architecture_raises(tmp_path: Path) -> None:
    """A hub id (no local model_index.json) with no explicit override can't be resolved."""
    ns = _base_namespace(
        architecture="someorg/some-diffusion-model", base_model_architecture=None
    )
    with pytest.raises(ValueError):
        build_lora_training_config_from_args(ns, _FakePipeline())


def test_run_train_image_generation_lora_cli_rejects_unimplemented_architecture(
    tmp_path: Path,
) -> None:
    """SDXL/SD3 are recognized enum members but not implemented — rejected before any heavy import."""
    ns = _base_namespace(base_model_architecture="stable_diffusion_xl", data_dir=tmp_path)
    assert run_train_image_generation_lora_cli(ns, _FakePipeline()) == 1


def test_build_lora_training_config_accepts_flux() -> None:
    """Flux is implemented (the recommended default) — config resolution doesn't reject it."""
    config = build_lora_training_config_from_args(
        _base_namespace(base_model_architecture="flux"), _FakePipeline()
    )
    assert config.base_architecture == BaseGenerationArchitecture.FLUX


def test_build_lora_training_config_accepts_chroma() -> None:
    """Chroma is implemented (reuses the Flux path) — config resolution doesn't reject it."""
    config = build_lora_training_config_from_args(
        _base_namespace(base_model_architecture="chroma"), _FakePipeline()
    )
    assert config.base_architecture == BaseGenerationArchitecture.CHROMA


def test_run_train_image_generation_lora_cli_rejects_non_pytorch_framework(tmp_path: Path) -> None:
    ns = _base_namespace(framework="keras", data_dir=tmp_path)
    assert run_train_image_generation_lora_cli(ns, _FakePipeline()) == 1


def test_run_train_image_generation_lora_cli_missing_data_dir(tmp_path: Path) -> None:
    ns = _base_namespace(data_dir=tmp_path / "does_not_exist")
    assert run_train_image_generation_lora_cli(ns, _FakePipeline()) == 1


def test_run_train_image_generation_lora_cli_missing_base_model(tmp_path: Path) -> None:
    ns = _base_namespace(architecture=None, data_dir=tmp_path)
    assert run_train_image_generation_lora_cli(ns, _FakePipeline()) == 1


def test_cli_train_image_generation_lora_rejects_keras(tmp_path: Path) -> None:
    """Full ``mb train`` dispatch reaches the LoRA branch and rejects before any heavy import."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.TRAIN.value),
                "--model-type",
                "image_generation_lora",
                "--framework",
                "keras",
                "--architecture",
                "sd15-base",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 1
    )


def test_cli_train_image_generation_lora_rejects_unimplemented_architecture(tmp_path: Path) -> None:
    """Full ``mb train`` dispatch, --base-model-architecture parsed and rejected before heavy import."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert (
        main(
            [
                "--config",
                str(default_pipeline_config_path()),
                str(ModelBuilderTaskType.TRAIN.value),
                "--model-type",
                "image_generation_lora",
                "--architecture",
                "sdxl-base",
                "--base-model-architecture",
                "stable_diffusion_xl",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 1
    )
