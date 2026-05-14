"""Argument parsers for ``mb info``."""

from __future__ import annotations

from pathlib import Path

from mb.models.types import InfoSubcommand
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.translations import _


def register(subparsers) -> None:
    info_parser = subparsers.add_parser(
        ModelBuilderTaskType.INFO.value,
        help=_("Show information about models or datasets"),
        description=_(
            "Display information about trained models or datasets, including metadata, "
            "architecture details, and dataset statistics."
        ),
    )
    info_subparsers = info_parser.add_subparsers(
        dest="info_command",
        help=_("Info subcommands"),
        metavar="SUBCOMMAND",
    )

    # mb info model
    info_model_parser = info_subparsers.add_parser(
        InfoSubcommand.MODEL.value,
        help=_("Show model information"),
        description=_(
            "Display detailed information about a trained model, including architecture, "
            "framework, number of parameters, and training metadata."
        ),
    )
    info_model_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help=_("Path to model file"),
    )

    # mb info dataset
    info_dataset_parser = info_subparsers.add_parser(
        InfoSubcommand.DATASET.value,
        help=_("Show dataset information"),
        description=_(
            "Display statistics about a dataset, including class distributions, "
            "image counts, and data directory structure."
        ),
    )
    info_dataset_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=_("Path to data directory"),
    )
