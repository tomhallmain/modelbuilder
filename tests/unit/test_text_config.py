"""Tests for ``mb.training.text_config`` (strict text_classification section parsing)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mb.models.types import TextBackendType, TextCalibrationMethod, TextThresholdPolicy
from mb.pipeline_config import PipelineConfig
from mb.training.text_config import (
    TEXT_CLASSIFICATION_DEFAULTS,
    TextConfigError,
    apply_text_overrides,
    dataset_file_options,
    parse_text_run_config,
    resolve_text_run_config,
)

from tests.test_utils import default_pipeline_config_path


def _defaults() -> dict:
    return copy.deepcopy(TEXT_CLASSIFICATION_DEFAULTS)


def test_defaults_parse() -> None:
    cfg = parse_text_run_config(_defaults())
    assert cfg.backend == TextBackendType.CHAR_NGRAM_LINEAR
    assert cfg.calibration == TextCalibrationMethod.AUTO
    assert cfg.threshold_policy.kind == TextThresholdPolicy.PRECISION_TARGET
    assert cfg.class_weight == "balanced"
    assert cfg.gold_tiers is None
    assert cfg.unreviewed_tier is None
    assert cfg.optim.lr is None


def test_shipped_yaml_section_matches_code_defaults() -> None:
    pc = PipelineConfig(default_pipeline_config_path())
    assert pc.get("text_classification") == TEXT_CLASSIFICATION_DEFAULTS
    resolve_text_run_config(pc)


def test_unknown_top_level_key_is_an_error() -> None:
    raw = _defaults()
    raw["tier_weigth"] = {"unreviewed": 0.5}
    with pytest.raises(TextConfigError) as exc:
        parse_text_run_config(raw)
    assert any("tier_weigth" in p for p in exc.value.problems)


def test_unknown_nested_key_is_an_error() -> None:
    raw = _defaults()
    raw["optim"]["learning_rate"] = 1e-3
    with pytest.raises(TextConfigError) as exc:
        parse_text_run_config(raw)
    assert any("learning_rate" in p for p in exc.value.problems)


def test_all_problems_reported_together() -> None:
    raw = _defaults()
    raw["backend"] = "nope"
    raw["seed"] = -1
    raw["threshold_policy"]["value"] = 1.5
    with pytest.raises(TextConfigError) as exc:
        parse_text_run_config(raw)
    assert len(exc.value.problems) == 3


def test_explicit_class_weight_mapping_accepts_string_keys() -> None:
    raw = _defaults()
    raw["class_weight"] = {"0": 1.0, 1: 4.0}
    cfg = parse_text_run_config(raw)
    assert cfg.class_weight == {0: 1.0, 1: 4.0}


def test_class_weight_mapping_needs_both_labels() -> None:
    raw = _defaults()
    raw["class_weight"] = {1: 4.0}
    with pytest.raises(TextConfigError):
        parse_text_run_config(raw)


def test_to_dict_round_trips() -> None:
    raw = _defaults()
    raw["class_weight"] = {0: 1.0, 1: 3.0}
    raw["tier_weight"] = {"unreviewed": 0.5}
    raw["unreviewed_tier"] = "unreviewed"
    raw["subsample_unreviewed"] = 4
    raw["gold_tiers"] = ["reviewed"]
    raw["reference_score_column"] = "prior_score"
    raw["unlabeled_file"] = "pool.tsv"
    cfg = parse_text_run_config(raw)
    assert parse_text_run_config(cfg.to_dict()) == cfg


def test_subsample_without_unreviewed_tier_is_an_error() -> None:
    raw = _defaults()
    raw["subsample_unreviewed"] = 4
    with pytest.raises(TextConfigError) as exc:
        parse_text_run_config(raw)
    assert any("unreviewed_tier" in p for p in exc.value.problems)


def test_dataset_file_options_default_and_override(tmp_path: Path) -> None:
    yml = tmp_path / "pipe.yaml"
    yml.write_text("text_classification:\n  unlabeled_file: pool.tsv\n", encoding="utf-8")
    assert dataset_file_options(PipelineConfig(yml)) == {
        "unlabeled_file": "pool.tsv",
        "reference_score_column": TEXT_CLASSIFICATION_DEFAULTS["reference_score_column"],
    }


def test_dataset_defaults_match_text_dataset_constants() -> None:
    from mb.data.text_dataset import COL_REFERENCE_SCORE, UNLABELED_FILE

    assert TEXT_CLASSIFICATION_DEFAULTS["reference_score_column"] == COL_REFERENCE_SCORE
    assert TEXT_CLASSIFICATION_DEFAULTS["unlabeled_file"] == UNLABELED_FILE


def test_overrides_skip_none_and_reach_nested_keys() -> None:
    out = apply_text_overrides(_defaults(), {"seed": 7, "optim.lr": 2e-5, "backend": None})
    assert out["seed"] == 7
    assert out["optim"]["lr"] == 2e-5
    assert out["optim"]["warmup_ratio"] == TEXT_CLASSIFICATION_DEFAULTS["optim"]["warmup_ratio"]
    assert out["backend"] == TEXT_CLASSIFICATION_DEFAULTS["backend"]


def test_with_backend_defaults_fills_only_nulls() -> None:
    raw = _defaults()
    raw["optim"]["epochs"] = 5
    cfg = parse_text_run_config(raw).with_backend_defaults(
        model_id="m", lr=1e-3, epochs=10, batch_size=64, backend_options={"x": 1}
    )
    assert (cfg.model_id, cfg.optim.lr, cfg.optim.epochs, cfg.optim.batch_size) == ("m", 1e-3, 5, 64)
    assert cfg.backend_options == {"x": 1}


def test_pipeline_yaml_section_merges_over_defaults(tmp_path: Path) -> None:
    yml = tmp_path / "pipe.yaml"
    yml.write_text(
        "model:\n  default_type: text_classification\n"
        "text_classification:\n  backend: encoder_finetune\n  optim:\n    epochs: 2\n",
        encoding="utf-8",
    )
    cfg = resolve_text_run_config(PipelineConfig(yml))
    assert cfg.backend == TextBackendType.ENCODER_FINETUNE
    assert cfg.optim.epochs == 2
    assert cfg.optim.warmup_ratio == TEXT_CLASSIFICATION_DEFAULTS["optim"]["warmup_ratio"]
