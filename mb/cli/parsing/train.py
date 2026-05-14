"""Argument parser for ``mb train``."""

from __future__ import annotations

from pathlib import Path

from mb.models.types import FrameworkType
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
        choices=["image_classification"],
        default="image_classification",
        help=_("Model type (default: image_classification)"),
    )
    train_parser.add_argument(
        "--framework",
        choices=[f.value for f in FrameworkType],
        help=_("Framework to use (default: from config)"),
    )
    train_parser.add_argument(
        "--architecture",
        help=_("Model architecture (e.g., resnet34)"),
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
        help=_("Image size (default: 224)"),
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help=_("Number of data loading workers (default: from config)"),
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
