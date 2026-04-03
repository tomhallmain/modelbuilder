"""Tests for get_training_hyperparams merge order (no torch, no disk I/O for merge cases)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mb.training.hyperparams import get_training_hyperparams


class TestGetTrainingHyperparamsMerge(unittest.TestCase):
    def test_cli_overrides_pipeline_overrides_model_defaults(self) -> None:
        """CLI > pipeline.training_hyperparams() > model_type_defaults."""
        model_defaults = {
            "frozen_epochs": 1,
            "unfrozen_epochs": 2,
            "frozen_lr": 0.01,
        }
        pipeline = MagicMock()
        pipeline.training_hyperparams.return_value = {
            "frozen_epochs": 50,
            "unfrozen_epochs": 60,
            "frozen_lr": 0.02,
            "batch_size": 16,
        }
        cli = {"frozen_epochs": 99, "batch_size": 8}

        out = get_training_hyperparams(
            model_type_defaults=model_defaults,
            pipeline_config=pipeline,
            cli_args=cli,
        )

        self.assertEqual(out["frozen_epochs"], 99)
        self.assertEqual(out["unfrozen_epochs"], 60)
        self.assertEqual(out["frozen_lr"], 0.02)
        self.assertEqual(out["batch_size"], 8)
        pipeline.training_hyperparams.assert_called_once()

    def test_pipeline_overrides_model_when_no_cli(self) -> None:
        model_defaults = {
            "frozen_epochs": 1,
            "unfrozen_epochs": 2,
            "frozen_lr": 0.01,
        }
        pipeline = MagicMock()
        pipeline.training_hyperparams.return_value = {
            "frozen_epochs": 11,
            "unfrozen_epochs": 22,
            "frozen_lr": 0.03,
        }

        out = get_training_hyperparams(
            model_type_defaults=model_defaults,
            pipeline_config=pipeline,
            cli_args=None,
        )

        self.assertEqual(out["frozen_epochs"], 11)
        self.assertEqual(out["unfrozen_epochs"], 22)
        self.assertEqual(out["frozen_lr"], 0.03)

    def test_model_defaults_only_when_no_pipeline_no_cli(self) -> None:
        model_defaults = {
            "frozen_epochs": 3,
            "unfrozen_epochs": 4,
            "frozen_lr": 0.05,
        }
        out = get_training_hyperparams(
            model_type_defaults=model_defaults,
            pipeline_config=None,
            cli_args=None,
        )
        self.assertEqual(out, model_defaults)


if __name__ == "__main__":
    unittest.main()
