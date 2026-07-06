"""
TrainPageController — resolves :class:`TrainPage` widget values into a training request:
:class:`~mb.training.run_args.TrainingRunArgs` for image classification, or
:class:`~mb.training.lora_diffusion_trainer.LoraTrainingConfig` for
``image_generation_lora``.

Extracted out of ``ui/pages/train_page.py`` because which of ~20 fields matter, and how
they validate, now differs by :class:`~mb.models.types.ModelType` — keeping that decision
logic in a plain module (not a ``QWidget`` method) means it's testable without Qt, and
keeps the page class focused on widget wiring rather than CLI/training-config semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from mb.models.types import ArchitectureType, FrameworkType, ModelType
from mb.training.lora_diffusion_trainer import LoraTrainingConfig
from mb.training.run_args import TrainingRunArgs
from mb.utils.translations import _


@dataclass
class TrainPageFieldValues:
    """Raw values collected from ``TrainPage`` widgets — plain data, no Qt types."""

    model_type: ModelType
    framework_text: str
    architecture_text: str
    data_dir_text: str
    output_dir_text: str
    batch_size: int
    image_size: int
    num_workers: int

    # image_classification only
    resume_from_text: str = ""
    run_id_text: str = ""
    skip_snapshot: bool = False
    frozen_epochs: int = 0
    unfrozen_epochs: int = 0
    frozen_lr: float = 0.0
    unfrozen_lr_max: float = 0.0
    unfrozen_lr_min: float = 0.0

    # image_generation_lora only
    base_model_architecture_text: str = ""
    lora_rank: int = 16
    lora_alpha: int = 0
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    seed_text: str = ""


TrainingRequest = Union[TrainingRunArgs, LoraTrainingConfig]


def build_training_request(values: TrainPageFieldValues) -> TrainingRequest:
    """Raises ``ValueError`` with a user-facing message on invalid input."""
    if values.model_type == ModelType.IMAGE_GENERATION_LORA:
        return _build_lora_config(values)
    return _build_classification_run_args(values)


def _build_classification_run_args(values: TrainPageFieldValues) -> TrainingRunArgs:
    data_dir = Path(values.data_dir_text.strip() or "data")
    output_dir = Path(values.output_dir_text.strip() or "data/models")
    fw = FrameworkType.try_from(values.framework_text)
    if fw is None:
        raise ValueError(_("Unsupported framework: {fw}").format(fw=values.framework_text))
    arch_raw = values.architecture_text.strip()
    if not arch_raw:
        raise ValueError(_("Architecture is required."))
    arch = ArchitectureType.try_from(arch_raw)
    if arch is None:
        raise ValueError(_("Unknown architecture: {a}").format(a=arch_raw))
    if not data_dir.exists():
        raise ValueError(_("Data directory does not exist."))
    resume_raw = values.resume_from_text.strip()
    resume_path = Path(resume_raw) if resume_raw else None
    if resume_path and not resume_path.exists():
        raise ValueError(_("Resume checkpoint path does not exist."))

    cli_hyperparams: dict = {
        "frozen_epochs": values.frozen_epochs,
        "unfrozen_epochs": values.unfrozen_epochs,
        "frozen_lr": values.frozen_lr,
        "unfrozen_lr_max": values.unfrozen_lr_max,
        "unfrozen_lr_min": values.unfrozen_lr_min,
        "image_size": values.image_size,
    }
    if values.batch_size > 0:
        cli_hyperparams["batch_size"] = values.batch_size
    if values.num_workers > 0:
        cli_hyperparams["num_workers"] = values.num_workers

    return TrainingRunArgs(
        framework=fw,
        architecture=arch,
        data_dir=data_dir,
        output_dir=output_dir,
        resume_from=resume_path,
        run_id=values.run_id_text.strip() or None,
        update_snapshot=not values.skip_snapshot,
        cli_hyperparams=cli_hyperparams,
    )


def _build_lora_config(values: TrainPageFieldValues) -> LoraTrainingConfig:
    from mb.models.generation_architectures import (
        BaseGenerationArchitecture,
        detect_base_architecture,
        get_descriptor,
    )

    fw_text = values.framework_text.strip()
    if fw_text and fw_text.lower() != "pytorch":
        raise ValueError(
            _("Image-generation LoRA training only supports the PyTorch framework (got: {fw}).").format(
                fw=fw_text
            )
        )

    base_model = values.architecture_text.strip()
    if not base_model:
        raise ValueError(
            _("Base model (a local checkpoint path or hub id) is required for image-generation LoRA.")
        )

    data_dir = Path(values.data_dir_text.strip() or "data")
    if not data_dir.exists():
        raise ValueError(_("Data directory does not exist."))
    output_dir = Path(values.output_dir_text.strip() or "data/models")

    base_arch = BaseGenerationArchitecture.try_from(values.base_model_architecture_text)
    if base_arch is None:
        base_arch = detect_base_architecture(base_model)
    if base_arch is None:
        raise ValueError(
            _(
                "Could not determine the base model's architecture from {path}. "
                "Select one explicitly."
            ).format(path=base_model)
        )
    if not get_descriptor(base_arch).implemented:
        raise ValueError(
            _("Base architecture '{arch}' is recognized but not implemented yet.").format(
                arch=base_arch.value
            )
        )

    seed_raw = values.seed_text.strip()
    seed: Optional[int]
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError:
            raise ValueError(_("Seed must be an integer.")) from None
    else:
        seed = None

    rank = values.lora_rank
    alpha = values.lora_alpha if values.lora_alpha > 0 else rank

    return LoraTrainingConfig(
        base_model=base_model,
        base_architecture=base_arch,
        data_dir=data_dir,
        output_dir=output_dir,
        rank=rank,
        alpha=alpha,
        resolution=values.image_size,
        batch_size=values.batch_size if values.batch_size > 0 else 1,
        learning_rate=values.learning_rate,
        max_train_steps=values.max_train_steps,
        num_workers=values.num_workers if values.num_workers > 0 else 0,
        seed=seed,
    )
