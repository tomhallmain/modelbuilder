"""Argument parsers for ``mb text`` (text-classification operations other than training)."""

from __future__ import annotations

from pathlib import Path

from mb.models.types import TextSubcommand
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.translations import _


def register(subparsers) -> None:
    text_parser = subparsers.add_parser(
        ModelBuilderTaskType.TEXT.value,
        help=_("Text classification: verify data, evaluate, score, compare runs"),
        description=_(
            "Operations on text-classification datasets and run directories. Train with "
            "`mb train --model-type text_classification`; defaults come from the pipeline's "
            "text_classification section."
        ),
    )
    text_sub = text_parser.add_subparsers(dest="text_command", help=_("Text subcommands"), metavar="SUBCOMMAND")

    verify = text_sub.add_parser(
        TextSubcommand.VERIFY.value,
        help=_("Check a dataset directory's integrity"),
        description=_(
            "Compares file hashes and row/label/tier/split/group counts with manifest.json "
            "(when present), and checks that texts are unique and each group sits in one split. "
            "Exits non-zero on any failed check."
        ),
    )
    verify.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=_("Dataset directory (default: text_classification.data_dir)"),
    )

    evaluate = text_sub.add_parser(
        TextSubcommand.EVALUATE.value,
        help=_("Re-run thresholds, metrics and reports for a run directory"),
        description=_(
            "Reloads the run's model and calibrator, chooses thresholds on val gold, and rewrites "
            "metrics.json, thresholds.json, predictions, review queue, cut rescore and the model card."
        ),
    )
    evaluate.add_argument("--model", type=Path, required=True, help=_("Run directory"))
    evaluate.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=_("Dataset directory (default: the run's config.yaml)"),
    )
    evaluate.add_argument("--device", default=None, help=_("Device override, e.g. cuda or cpu"))

    score = text_sub.add_parser(
        TextSubcommand.SCORE.value,
        help=_("Score one string per line with a run's calibrated model"),
        description=_(
            "Writes term<TAB>score with a header, one row per input line in input order."
        ),
    )
    score.add_argument("--model", type=Path, required=True, help=_("Run directory"))
    score.add_argument("--input", type=Path, required=True, help=_("UTF-8 text file, one string per line"))
    score.add_argument("--output", type=Path, required=True, help=_("Output TSV path"))
    score.add_argument("--device", default=None, help=_("Device override, e.g. cuda or cpu"))

    compare = text_sub.add_parser(
        TextSubcommand.COMPARE.value,
        help=_("Tabulate headline metrics across run directories"),
        description=_(
            "One row per run: test gold AP, full AP, recall at precision 0.90, keep_reviewed "
            "false-positive rate and ECE. Wildcards are expanded (e.g. data/models/text_runs/*)."
        ),
    )
    compare.add_argument("runs", nargs="+", help=_("Run directories or wildcard patterns"))
