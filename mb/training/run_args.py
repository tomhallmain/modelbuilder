"""
Structured inputs for a training run (paths, framework, CLI-style hyperparams).

Pass a :class:`TrainingRunArgs` instance as the first argument to
:meth:`~mb.training.trainer.ModelTrainer.train` (CLI, GUI, or tests).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from mb.models.types import ArchitectureType, FrameworkType


@dataclass(frozen=True)
class TrainingRunArgs:
    """Immutable parameters for :meth:`~mb.training.trainer.ModelTrainer.train`."""

    framework: FrameworkType
    architecture: ArchitectureType
    data_dir: Path
    output_dir: Path
    resume_from: Path | None
    run_id: str | None
    update_snapshot: bool
    cli_hyperparams: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict (paths as strings)."""
        return {
            "framework": self.framework.value,
            "architecture": self.architecture.value,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "resume_from": str(self.resume_from) if self.resume_from is not None else None,
            "run_id": self.run_id,
            "update_snapshot": self.update_snapshot,
            "cli_hyperparams": dict(self.cli_hyperparams),
        }

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> TrainingRunArgs:
        """Deserialize from :meth:`to_json_dict` output."""
        fw = FrameworkType.try_from(d.get("framework"))
        if fw is None:
            raise ValueError(f"Unsupported framework: {d.get('framework')!r}")
        arch = ArchitectureType.try_from(d.get("architecture"))
        if arch is None:
            raise ValueError(f"Unsupported architecture: {d.get('architecture')!r}")
        rf = d.get("resume_from")
        return cls(
            framework=fw,
            architecture=arch,
            data_dir=Path(d["data_dir"]),
            output_dir=Path(d["output_dir"]),
            resume_from=Path(rf) if rf else None,
            run_id=d.get("run_id"),
            update_snapshot=bool(d.get("update_snapshot", True)),
            cli_hyperparams=dict(d.get("cli_hyperparams") or {}),
        )


def load_training_run_args_json(path: Path) -> TrainingRunArgs:
    """Load :class:`TrainingRunArgs` from a JSON file (see :meth:`TrainingRunArgs.to_json_dict`)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Training args JSON must be an object at the top level")
    return TrainingRunArgs.from_json_dict(raw)
