"""Argument parsers for ``mb evaluate``."""

from __future__ import annotations

from pathlib import Path

from mb.cli.parsing.common import MODEL_TYPE_CLI_CHOICES
from mb.models.types import EvaluateSubcommand, FrameworkType, ModelType
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.translations import _


def register(subparsers) -> None:
    evaluate_parser = subparsers.add_parser(
        ModelBuilderTaskType.EVALUATE.value,
        help=_("Evaluate trained models or datasets"),
        description=_(
            "Score a trained classifier on your train/test tree, surface likely labeling mistakes, "
            "or compare two checkpoints on the same data."
        ),
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(
        dest="evaluate_command",
        help=_("Evaluate subcommands"),
        metavar="SUBCOMMAND",
    )

    evaluate_metrics_parser = evaluate_subparsers.add_parser(
        EvaluateSubcommand.METRICS.value,
        help=_("Compute classification metrics on a labeled split"),
        description=_(
            "Accuracy, confusion matrix, and per-class counts on an ImageFolder-style directory "
            "(class-named subfolders of images). PyTorch requires --architecture and a .pth/.pt file."
        ),
    )
    evaluate_metrics_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=_("Path to trained model file (.pth / .pt / .h5 / .keras)"),
    )
    evaluate_metrics_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=_("ImageFolder root (e.g. .../test with one subfolder per class)"),
    )
    evaluate_metrics_parser.add_argument(
        "--framework",
        choices=[FrameworkType.PYTORCH.value, FrameworkType.KERAS.value],
        default=None,
        help=_("Training framework (default: infer from file extension)"),
    )
    evaluate_metrics_parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help=_("Model architecture id (required for PyTorch .pth checkpoints, e.g. resnet34)"),
    )
    evaluate_metrics_parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=_("Override class count (default: infer from directory layout)"),
    )
    evaluate_metrics_parser.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CLI_CHOICES,
        default=ModelType.IMAGE_CLASSIFICATION.value,
        help=_("Pipeline model type (default: image_classification)"),
    )
    evaluate_metrics_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=_("Square input size (default: 224)"),
    )
    evaluate_metrics_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=_("Batch size (default: 32)"),
    )
    evaluate_metrics_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=_("DataLoader workers for PyTorch (default: 0)"),
    )
    evaluate_metrics_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=_("PyTorch device override, e.g. cuda or cpu (default: auto)"),
    )
    evaluate_metrics_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=_("Validate arguments and exit without running inference"),
    )

    evaluate_misclassified_parser = evaluate_subparsers.add_parser(
        EvaluateSubcommand.MISCLASSIFIED.value,
        help=_("List images whose predicted class differs from the folder label"),
        description=_(
            "Compares model predictions to ImageFolder-style on-disk class folders to surface "
            "label noise and borderline samples. Image classification only."
        ),
    )
    evaluate_misclassified_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=_("Path to trained model file (.pth / .pt / .h5 / .keras)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=_("ImageFolder root (e.g. .../test with one subfolder per class)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--framework",
        choices=[FrameworkType.PYTORCH.value, FrameworkType.KERAS.value],
        default=None,
        help=_("Training framework (default: infer from file extension)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help=_("Model architecture id (required for PyTorch .pth checkpoints, e.g. resnet34)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=_("Override class count (default: infer from directory layout)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CLI_CHOICES,
        default=ModelType.IMAGE_CLASSIFICATION.value,
        help=_("Pipeline model type (default: image_classification)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=_("Square input size (default: 224)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=_("Batch size (default: 32)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=_("DataLoader workers for PyTorch (default: 0)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=_("PyTorch device override, e.g. cuda or cpu (default: auto)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--max-report",
        type=int,
        default=None,
        metavar="N",
        help=_("Cap printed / exported misclassified rows (default: no cap; totals still full)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=_("Optional CSV path (still prints a summary to stdout)"),
    )
    evaluate_misclassified_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=_("Validate arguments and exit without running inference"),
    )

    evaluate_compare_parser = evaluate_subparsers.add_parser(
        EvaluateSubcommand.COMPARE.value,
        help=_("Compare two models on the same labeled split (paired outcomes & disagreements)"),
        description=_(
            "Runs both checkpoints on the same ImageFolder split in lockstep: top-1 accuracy each, "
            "a 2×2-style breakdown (both correct, only A, only B, both wrong vs folder labels), "
            "and how often A and B disagree on the predicted class. PyTorch and Keras supported "
            "when both files use the same framework."
        ),
    )
    evaluate_compare_parser.add_argument(
        "--model-a",
        type=Path,
        required=True,
        help=_("Path to first model file (.pth / .pt / .h5 / .keras)"),
    )
    evaluate_compare_parser.add_argument(
        "--model-b",
        type=Path,
        required=True,
        help=_("Path to second model file (.pth / .pt / .h5 / .keras)"),
    )
    evaluate_compare_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=_("ImageFolder root (e.g. .../test with one subfolder per class)"),
    )
    evaluate_compare_parser.add_argument(
        "--framework",
        choices=[FrameworkType.PYTORCH.value, FrameworkType.KERAS.value],
        default=None,
        help=_("Training framework hint for both models (default: infer from each file)"),
    )
    evaluate_compare_parser.add_argument(
        "--framework-b",
        choices=[FrameworkType.PYTORCH.value, FrameworkType.KERAS.value],
        default=None,
        help=_("Optional framework hint for model B only (default: same as --framework)"),
    )
    evaluate_compare_parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help=_("Model architecture id for PyTorch model A (e.g. resnet34)"),
    )
    evaluate_compare_parser.add_argument(
        "--architecture-b",
        type=str,
        default=None,
        help=_("Architecture for PyTorch model B when it differs from model A (default: --architecture)"),
    )
    evaluate_compare_parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=_("Override class count (default: infer from directory layout)"),
    )
    evaluate_compare_parser.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CLI_CHOICES,
        default=ModelType.IMAGE_CLASSIFICATION.value,
        help=_("Pipeline model type (default: image_classification)"),
    )
    evaluate_compare_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=_("Square input size (default: 224)"),
    )
    evaluate_compare_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=_("Batch size (default: 32)"),
    )
    evaluate_compare_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=_("DataLoader workers for PyTorch (default: 0)"),
    )
    evaluate_compare_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=_("PyTorch device override, e.g. cuda or cpu (default: auto)"),
    )
    evaluate_compare_parser.add_argument(
        "--max-disagreement-report",
        type=int,
        default=None,
        metavar="N",
        help=_(
            "Cap printed / exported rows where A and B disagree on class (default: no cap; "
            "totals still full)"
        ),
    )
    evaluate_compare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=_("Optional TSV path for disagreement rows (summary still prints to stdout)"),
    )
    evaluate_compare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=_("Validate arguments and exit without running inference"),
    )
