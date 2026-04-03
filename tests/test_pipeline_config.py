"""Tests for mb.pipeline_config YAML loading (PyYAML only, no torch)."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from mb.pipeline_config import PipelineConfig


class TestPipelineConfigLoad(unittest.TestCase):
    def test_minimal_yaml_merges_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            yml = Path(td) / "pipe.yaml"
            yml.write_text(
                textwrap.dedent(
                    """
                    model:
                      default_architecture: tiny_net
                    training:
                      frozen_epochs: 2
                    """
                ).strip(),
                encoding="utf-8",
            )
            cfg = PipelineConfig(yml)
            self.assertEqual(cfg.active_path, yml)
            for key in ("model", "data", "training", "paths"):
                self.assertIn(key, cfg.to_dict())
            self.assertEqual(cfg.get("model.default_architecture"), "tiny_net")
            self.assertEqual(cfg.get("training.frozen_epochs"), 2)
            self.assertIsNotNone(cfg.get("data.raw_data_dir"))
            self.assertIsNotNone(cfg.get("paths.models_dir"))

    def test_unknown_top_level_keys_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            yml = Path(td) / "extra.yaml"
            yml.write_text(
                "extra_stuff: {foo: 1}\nmodel:\n  default_framework: pytorch\n",
                encoding="utf-8",
            )
            cfg = PipelineConfig(yml)
            self.assertNotIn("extra_stuff", cfg.to_dict())
            self.assertEqual(cfg.get("model.default_framework"), "pytorch")


if __name__ == "__main__":
    unittest.main()
