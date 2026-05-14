"""Argument parsers for ``mb export``."""

from __future__ import annotations

from pathlib import Path

from mb.models.types import ExportSubcommand
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.translations import _


def register(subparsers) -> None:
    export_parser = subparsers.add_parser(
        ModelBuilderTaskType.EXPORT.value,
        help=_("Export model bundle artifacts"),
        description=_(
            "Export deployment bundle artifacts such as safetensors weights and metadata manifests."
        ),
    )
    export_subparsers = export_parser.add_subparsers(
        dest="export_command",
        help=_("Export subcommands"),
        metavar="SUBCOMMAND",
    )

    export_bundle_parser = export_subparsers.add_parser(
        ExportSubcommand.BUNDLE.value,
        help=_("Export a model bundle (safetensors + manifest + optional architecture stub)"),
    )
    export_bundle_parser.add_argument("--input", type=Path, required=True, help=_("Input model file (.pth/.pt)"))
    export_bundle_parser.add_argument("--output-dir", type=Path, required=True, help=_("Output bundle directory"))
    export_bundle_parser.add_argument("--architecture", type=str, help=_("Model architecture id (e.g. resnet34)"))
    export_bundle_parser.add_argument("--num-classes", type=int, help=_("Number of output classes"))
    export_bundle_parser.add_argument("--class-names", nargs="+", help=_("Optional ordered class names"))
    export_bundle_parser.add_argument("--data-dir", type=Path, help=_("Optional data dir to infer class names from train/"))
    export_bundle_parser.add_argument("--image-size", type=int, help=_("Expected input image size (defaults to pipeline)"))
    export_bundle_parser.add_argument("--snapshot", type=Path, help=_("Optional snapshot JSON path"))
    export_bundle_parser.add_argument(
        "--no-architecture-py",
        action="store_true",
        help=_("Do not emit model_architecture.py stub"),
    )
    export_bundle_parser.add_argument("--run-id", type=str, help=_("Optional run ID to include in manifest"))
