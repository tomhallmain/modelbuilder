"""Argument parser for ``mb train``."""

from __future__ import annotations

from pathlib import Path

from mb.cli.parsing.common import MODEL_TYPE_CLI_CHOICES
from mb.models.generation_architectures import BaseGenerationArchitecture
from mb.models.types import FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.translations import _


def register(subparsers) -> None:
    train_parser = subparsers.add_parser(
        ModelBuilderTaskType.TRAIN.value,
        help=_("Train a model"),
        description=_(
            "Train a machine learning model using the specified framework and architecture. "
            "Supports transfer learning with frozen/unfrozen training phases."
        ),
    )
    train_parser.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CLI_CHOICES,
        default=ModelType.IMAGE_CLASSIFICATION.value,
        help=_("Model type (default: image_classification)"),
    )
    train_parser.add_argument(
        "--framework",
        choices=[f.value for f in FrameworkType],
        help=_("Framework to use (default: from config)"),
    )
    train_parser.add_argument(
        "--architecture",
        help=_(
            "Model architecture (e.g., resnet34). For --model-type image_generation_lora, "
            "this instead names the base image-generation model to fine-tune (a local path or "
            "hub id, e.g. a Stable Diffusion checkpoint)."
        ),
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        help=_("Data directory (default: from config)"),
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help=_("Output directory for models (default: from config)"),
    )
    train_parser.add_argument(
        "--frozen-epochs",
        type=int,
        help=_("Number of frozen training epochs (default: from config)"),
    )
    train_parser.add_argument(
        "--unfrozen-epochs",
        type=int,
        help=_("Number of unfrozen training epochs (default: from config)"),
    )
    train_parser.add_argument(
        "--frozen-lr",
        type=float,
        help=_("Learning rate for frozen phase (default: from config)"),
    )
    train_parser.add_argument(
        "--unfrozen-lr-max",
        type=float,
        help=_("Maximum learning rate for unfrozen phase (default: from config)"),
    )
    train_parser.add_argument(
        "--unfrozen-lr-min",
        type=float,
        help=_("Minimum learning rate for unfrozen phase (default: from config)"),
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help=_("Batch size (default: from config or auto-detect)"),
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        help=_(
            "Image size (default: 224). For --model-type image_generation_lora, this is the "
            "training resolution instead (default: 512)."
        ),
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help=_("Number of data loading workers (default: from config)"),
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        help=_(
            "Random seed. Currently only consumed by --model-type image_generation_lora "
            "(reproducible LoRA weight init and noise sampling)."
        ),
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        help=_("LoRA adapter rank (--model-type image_generation_lora only; default: 16)"),
    )
    train_parser.add_argument(
        "--lora-alpha",
        type=int,
        help=_(
            "LoRA adapter alpha scaling (--model-type image_generation_lora only; "
            "default: same as --lora-rank)"
        ),
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        help=_(
            "Learning rate for the LoRA adapter parameters (--model-type "
            "image_generation_lora only; default: 1e-4). Unrelated to --frozen-lr/"
            "--unfrozen-lr-* (image_classification only)."
        ),
    )
    train_parser.add_argument(
        "--max-train-steps",
        type=int,
        help=_(
            "Total optimizer steps to train (--model-type image_generation_lora only; "
            "default: 1000). LoRA training here is step-based, not epoch-based."
        ),
    )
    train_parser.add_argument(
        "--base-model-architecture",
        choices=[a.value for a in BaseGenerationArchitecture],
        default=None,
        help=_(
            "--model-type image_generation_lora only: the base model's architecture "
            "(different architectures load different components and LoRA target layers). "
            "Auto-detected from a local checkpoint's model_index.json when omitted; required "
            "when --architecture is a hub id rather than a local path, since that can't be "
            "auto-detected without a network call. flux (recommended default), chroma, and "
            "stable_diffusion_1x are implemented today — stable_diffusion_xl, "
            "stable_diffusion_3, and z_image_turbo are recognized but not yet supported."
        ),
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help=_("Path to checkpoint to resume training from"),
    )
    train_parser.add_argument(
        "--run-id",
        type=str,
        help=_("Run ID of unified snapshot to update (auto-detects latest if not provided)"),
    )
    train_parser.add_argument(
        "--skip-snapshot-update",
        action="store_true",
        help=_("Skip updating the unified snapshot with training data"),
    )
    train_parser.add_argument(
        "--train-args-json",
        type=Path,
        metavar="PATH",
        help=_(
            "Load TrainingRunArgs from JSON (see mb.training.run_args); other train flags are ignored"
        ),
    )
