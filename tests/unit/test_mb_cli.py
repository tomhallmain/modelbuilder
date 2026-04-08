"""CLI behavior: argparse exit codes, help text, ``train --train-args-json`` routing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mb.cli import create_parser, main
from mb.models.types import ArchitectureType, FrameworkType
from mb.training.run_args import TrainingRunArgs

from tests.test_utils import default_pipeline_config_path, repo_root


def test_main_no_command_is_nonzero() -> None:
    assert main([]) == 1


def test_main_invalid_top_level_command_is_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-valid-command"])
    assert exc.value.code != 0


def test_main_version_is_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_train_help_lists_expected_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--framework" in out
    assert "--train-args-json" in out
    assert "--frozen-epochs" in out


def test_create_parser_version_action() -> None:
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_main_top_level_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "data" in out
    assert "train" in out
    assert "convert" in out


def test_data_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["data", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "gather" in out
    assert "convert" in out
    assert "create-dataset" in out


def test_data_convert_help_lists_raw_data_dir(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["data", "convert", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--raw-data-dir" in out
    assert "--run-id" in out


def test_convert_command_help_lists_target_choices(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["convert", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--target" in out
    assert "--input" in out


def test_main_invalid_data_subcommand_fails_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["data", "not-a-real-subcommand"])
    assert exc.value.code != 0


def test_mb_cli_module_help_via_subprocess() -> None:
    root = repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "mb.cli", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0
    assert "train" in result.stdout


def test_train_train_args_json_invokes_trainer_with_stub_trainer(
    tmp_path: Path,
) -> None:
    """JSON path exercises ``handle_train``; real :class:`ModelTrainer` is patched out (no training)."""
    cfg = default_pipeline_config_path()
    assert cfg.is_file()

    data_dir = tmp_path / "data"
    (data_dir / "train" / "coherent").mkdir(parents=True)
    (data_dir / "test" / "coherent").mkdir(parents=True)
    out_dir = tmp_path / "models"
    out_dir.mkdir()

    json_path = tmp_path / "run.json"
    args = TrainingRunArgs(
        framework=FrameworkType.PYTORCH,
        architecture=ArchitectureType.RESNET18,
        data_dir=data_dir,
        output_dir=out_dir,
        resume_from=None,
        run_id=None,
        update_snapshot=False,
        cli_hyperparams={"frozen_epochs": 0, "unfrozen_epochs": 0, "batch_size": 1, "num_workers": 0},
    )
    json_path.write_text(json.dumps(args.to_json_dict()), encoding="utf-8")

    calls: list[Any] = []

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", kwargs))

        def get_supported_architectures(self) -> list[str]:
            return [ArchitectureType.RESNET18.value]

        def train(self, run_args: TrainingRunArgs) -> Path:
            calls.append(("train", run_args))
            return out_dir / "fake_model.pth"

    with patch("mb.training.trainer.ModelTrainer", FakeTrainer):
        rc = main(
            [
                "--config",
                str(cfg),
                "train",
                "--train-args-json",
                str(json_path),
            ]
        )

    assert rc == 0
    assert len(calls) == 2
    assert calls[0][0] == "init"
    assert calls[1][0] == "train"
    assert calls[1][1].architecture == ArchitectureType.RESNET18
