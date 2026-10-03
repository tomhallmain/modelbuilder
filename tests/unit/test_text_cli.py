"""CLI plumbing for text classification (``mb text`` and ``mb train --model-type text_classification``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mb.cli import main
from mb.models.types import ModelType, TextSubcommand
from mb.training.text_config import TEXT_CLASSIFICATION_DEFAULTS, parse_text_run_config
from mb.utils.constants import ModelBuilderTaskType

from tests.fixtures.text_dataset import write_text_dataset

TEXT = ModelBuilderTaskType.TEXT.value


def test_text_verify_exit_codes(tmp_path: Path) -> None:
    d = write_text_dataset(tmp_path / "ds")
    assert main([TEXT, TextSubcommand.VERIFY.value, "--data-dir", str(d)]) == 0
    with open(d / "dataset.tsv", "a", encoding="utf-8") as f:
        f.write("extra\t0\ttrain\n")
    assert main([TEXT, TextSubcommand.VERIFY.value, "--data-dir", str(d)]) == 1


def test_train_rejects_unknown_text_config_key(tmp_path: Path) -> None:
    d = write_text_dataset(tmp_path / "ds")
    yml = tmp_path / "pipe.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "model": {"default_type": ModelType.TEXT_CLASSIFICATION.value},
                "text_classification": {"data_dir": str(d), "bogus": 1},
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(yml), ModelBuilderTaskType.TRAIN.value]) == 1


def test_train_requires_dataset_file(tmp_path: Path) -> None:
    assert (
        main(
            [
                ModelBuilderTaskType.TRAIN.value,
                "--model-type",
                ModelType.TEXT_CLASSIFICATION.value,
                "--data-dir",
                str(tmp_path / "missing"),
            ]
        )
        == 1
    )


def test_create_dataset_refuses_text_model_type(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    code = main(
        [
            ModelBuilderTaskType.DATA.value,
            "create-dataset",
            "--raw-data-dir",
            str(raw),
            "--data-dir",
            str(tmp_path / "out"),
            "--model-type",
            ModelType.TEXT_CLASSIFICATION.value,
        ]
    )
    assert code == 1
    assert not (tmp_path / "out").exists()


def _fake_run(root: Path, name: str, gold_ap: float) -> Path:
    run = root / name
    run.mkdir(parents=True)
    cfg = parse_text_run_config(TEXT_CLASSIFICATION_DEFAULTS)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    metrics = {
        "test": {
            "gold": {"ap": gold_ap, "ece": 0.01, "fpr": 0.05, "recall_at_precision": {"0.90": 0.7}},
            "full": {"ap": gold_ap - 0.1},
        }
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run


def test_text_compare_tabulates_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_run(tmp_path / "runs", "run_a", 0.8)
    _fake_run(tmp_path / "runs", "run_b", 0.9)
    (tmp_path / "runs" / "not_a_run").mkdir()
    code = main([TEXT, TextSubcommand.COMPARE.value, str(tmp_path / "runs" / "*")])
    out = capsys.readouterr().out
    assert code == 0
    assert "run_a" in out and "run_b" in out and "0.9000" in out and "0.0500" in out
    assert "not_a_run" not in out


def test_text_compare_with_no_runs_fails(tmp_path: Path) -> None:
    assert main([TEXT, TextSubcommand.COMPARE.value, str(tmp_path / "nothing*")]) == 1


def test_text_score_missing_input(tmp_path: Path) -> None:
    code = main(
        [
            TEXT,
            TextSubcommand.SCORE.value,
            "--model",
            str(tmp_path),
            "--input",
            str(tmp_path / "nope.txt"),
            "--output",
            str(tmp_path / "out.tsv"),
        ]
    )
    assert code == 1
