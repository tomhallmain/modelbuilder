"""Argument parser for ``mb convert`` (model format conversion)."""

from __future__ import annotations

from pathlib import Path

from mb.models.types import ConversionTargetFormat, FrameworkType
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.translations import _


def register(subparsers) -> None:
    convert_model_parser = subparsers.add_parser(
        ModelBuilderTaskType.CONVERT.value,
        help=_("Convert model between formats"),
        description=_(
            "Convert a trained model between different formats. "
            "Supports PyTorch -> ONNX, PyTorch -> SafeTensors, and Keras -> ONNX."
        ),
    )
    convert_model_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=_("Input model file"),
    )
    convert_model_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=_("Output model file"),
    )
    convert_model_parser.add_argument(
        "--framework",
        choices=[f.value for f in FrameworkType],
        help=_("Source framework (auto-detected if not specified)"),
    )
    convert_model_parser.add_argument(
        "--target",
        choices=[f.value for f in ConversionTargetFormat],
        required=True,
        help=_("Target format (onnx or safetensors)"),
    )
    convert_model_parser.add_argument(
        "--architecture",
        help=_("Model architecture (required for PyTorch -> ONNX conversion, e.g., 'resnet34')"),
    )
    convert_model_parser.add_argument(
        "--num-classes",
        type=int,
        help=_("Number of output classes (required for PyTorch -> ONNX conversion)"),
    )
    convert_model_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=_("Input image size (default: 224, used for ONNX conversion)"),
    )
