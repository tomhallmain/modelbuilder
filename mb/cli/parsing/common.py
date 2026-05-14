"""Shared argparse choices for CLI parsers."""

from mb.models.types import ModelType

MODEL_TYPE_CLI_CHOICES = [m.value for m in ModelType]
