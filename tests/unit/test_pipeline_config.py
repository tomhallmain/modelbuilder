"""Tests for mb.pipeline_config YAML loading (PyYAML only, no torch)."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from mb.models.types import ArchitectureType, FrameworkType
from argparse import Namespace

from mb.pipeline_config import (
    PipelineConfig,
    gather_pipeline_defaults,
    gather_subpath_for_display,
    gather_subpath_for_storage,
    reload_pipeline_config,
    resolve_create_dataset_cli,
    resolve_gather_path_under_raw,
)
from mb.utils.constants import DatasetSplitMode


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

    def test_coerce_data_recovers_from_invalid_test_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            yml = Path(td) / "bad.yaml"
            yml.write_text(
                textwrap.dedent(
                    """
                    data:
                      test_per_class: not_a_number
                      test_split_mode: dataset_weighted
                      seed: "42"
                    """
                ).strip(),
                encoding="utf-8",
            )
            cfg = PipelineConfig(yml)
            self.assertEqual(cfg.get("data.test_per_class"), 1000)
            self.assertEqual(cfg.get("data.test_split_mode"), DatasetSplitMode.DATASET_WEIGHTED.value)
            self.assertEqual(cfg.get("data.seed"), 42)

    def test_resolve_create_dataset_cli_merges_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            yml = Path(td) / "ds.yaml"
            yml.write_text(
                textwrap.dedent(
                    """
                    data:
                      test_per_class: 2500
                      test_split_mode: dataset-weighted
                      test_small_class_threshold: 800
                      seed: 99
                    """
                ).strip(),
                encoding="utf-8",
            )
            reload_pipeline_config(yml, force=True)
            args = Namespace(
                test_per_class=None,
                test_split_mode=None,
                test_small_class_threshold=None,
                seed=None,
            )
            out = resolve_create_dataset_cli(args)
            self.assertEqual(out["test_per_class"], 2500)
            self.assertEqual(out["test_split_mode"], DatasetSplitMode.DATASET_WEIGHTED)
            self.assertEqual(out["test_small_class_threshold"], 800)
            self.assertEqual(out["seed"], 99)

            args2 = Namespace(
                test_per_class=10,
                test_split_mode="fixed",
                test_small_class_threshold=None,
                seed=1,
            )
            out2 = resolve_create_dataset_cli(args2)
            self.assertEqual(out2["test_per_class"], 10)
            self.assertEqual(out2["test_split_mode"], DatasetSplitMode.FIXED)
            self.assertEqual(out2["seed"], 1)


if __name__ == "__main__":
    unittest.main()
