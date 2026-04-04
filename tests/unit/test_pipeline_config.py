"""Tests for mb.pipeline_config YAML loading (PyYAML only, no torch)."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from mb.models.types import ArchitectureType, FrameworkType
from mb.pipeline_config import (
    PipelineConfig,
    gather_pipeline_defaults,
    gather_subpath_for_display,
    gather_subpath_for_storage,
    reload_pipeline_config,
    resolve_gather_path_under_raw,
)


class TestPipelineConfigLoad(unittest.TestCase):
    def test_minimal_yaml_merges_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            yml = Path(td) / "pipe.yaml"
            yml.write_text(
                textwrap.dedent(
                    """
                    model:
                      default_architecture: resnet50
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
            self.assertEqual(cfg.get("model.default_architecture"), ArchitectureType.RESNET50.value)
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
            self.assertEqual(cfg.get("model.default_framework"), FrameworkType.PYTORCH.value)

    def test_gather_raw_data_uses_data_raw_data_dir(self) -> None:
        reload_pipeline_config(None, force=True)
        gd = gather_pipeline_defaults()
        self.assertEqual(gd["raw_data_dir"].name, "raw_data")
        self.assertEqual(gd["target_dir"], Path("raw_data") / "coherent")
        self.assertEqual(gd["rejected_dir"], Path("raw_data") / "rejected")

    def test_resolve_gather_relative_under_raw(self) -> None:
        r = resolve_gather_path_under_raw("raw_data", "coherent", "coherent")
        self.assertEqual(r, Path("raw_data") / "coherent")
        r2 = resolve_gather_path_under_raw("raw_data", "raw_data/coherent", "coherent")
        self.assertEqual(r2, Path("raw_data") / "coherent")

    def test_gather_subpath_roundtrip(self) -> None:
        self.assertEqual(
            gather_subpath_for_display("raw_data", "raw_data/coherent"),
            "coherent",
        )
        self.assertEqual(
            gather_subpath_for_storage("raw_data", "raw_data/coherent", "coherent"),
            "coherent",
        )
        self.assertEqual(
            gather_subpath_for_storage("raw_data", "coherent", "coherent"),
            "coherent",
        )


if __name__ == "__main__":
    unittest.main()
