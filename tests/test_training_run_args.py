"""Tests for TrainingRunArgs JSON helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from mb.training.run_args import TrainingRunArgs, load_training_run_args_json


class TestTrainingRunArgsJson(unittest.TestCase):
    def test_roundtrip_to_from_dict(self) -> None:
        args = TrainingRunArgs(
            framework="pytorch",
            architecture="resnet34",
            data_dir=Path("data"),
            output_dir=Path("out/models"),
            resume_from=Path("ckpt.pth"),
            run_id="r1",
            update_snapshot=True,
            cli_hyperparams={"frozen_epochs": 1, "batch_size": 32},
        )
        d = args.to_json_dict()
        back = TrainingRunArgs.from_json_dict(d)
        self.assertEqual(back, args)

    def test_roundtrip_none_resume(self) -> None:
        args = TrainingRunArgs(
            framework="keras",
            architecture="resnet50",
            data_dir=Path("d"),
            output_dir=Path("o"),
            resume_from=None,
            run_id=None,
            update_snapshot=False,
            cli_hyperparams={},
        )
        back = TrainingRunArgs.from_json_dict(args.to_json_dict())
        self.assertEqual(back, args)

    def test_load_training_run_args_json_file(self) -> None:
        raw = {
            "framework": "pytorch",
            "architecture": "resnet34",
            "data_dir": "x/trainset",
            "output_dir": "y/models",
            "resume_from": None,
            "run_id": None,
            "update_snapshot": True,
            "cli_hyperparams": {},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(raw, f)
            path = Path(f.name)
        try:
            loaded = load_training_run_args_json(path)
            self.assertEqual(loaded.framework, "pytorch")
            self.assertEqual(loaded.data_dir, Path("x/trainset"))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
