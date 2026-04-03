"""Launch ``mb train --train-args-json`` in a detached OS process (GUI long runs)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from mb.training.run_args import TrainingRunArgs


def spawn_mb_train_subprocess(
    run_args: TrainingRunArgs,
    *,
    pipeline_config: Path | None,
    log_file: Path,
) -> Tuple[subprocess.Popen, Path]:
    """
    Start training in a subprocess that is not tied to the GUI process lifetime.

    Writes *run_args* to a temporary JSON file (returned as the second element).
    Stdout and stderr of the child are appended to *log_file*.

    Returns:
        ``(popen, json_path)`` — you may delete *json_path* after start if desired.
    """
    fd, tmp = tempfile.mkstemp(prefix="mb_train_args_", suffix=".json")
    json_path = Path(tmp)
    try:
        json_path.write_text(
            json.dumps(run_args.to_json_dict(), indent=2),
            encoding="utf-8",
        )
    finally:
        os.close(fd)

    cmd = [sys.executable, "-m", "mb.cli"]
    if pipeline_config is not None:
        cmd.extend(["--config", str(pipeline_config)])
    cmd.extend(["train", "--train-args-json", str(json_path)])

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_file, "w", encoding="utf-8")
    popen_kw: dict = {
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kw["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kw["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kw)
    finally:
        log_f.close()

    return proc, json_path
