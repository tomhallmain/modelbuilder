"""Argparse wiring for the ``mb`` CLI (split from :mod:`mb.cli` for maintainability)."""

from __future__ import annotations

from typing import Any


def register_subparsers(subparsers: Any) -> None:
    """Attach all top-level command parsers to *subparsers*."""
    from mb.cli.parsing import data, evaluate, export, info, model_convert, train

    data.register(subparsers)
    train.register(subparsers)
    model_convert.register(subparsers)
    info.register(subparsers)
    export.register(subparsers)
    evaluate.register(subparsers)
