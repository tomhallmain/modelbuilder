"""``mb.training.text_trainer.cli_progress_logger`` throttling."""

from __future__ import annotations

import pytest

import mb.training.text_trainer as text_trainer


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, fmt: str, *args) -> None:
        self.messages.append(fmt % args)


def test_cli_progress_throttles_steps_but_not_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    rec = _Recorder()
    monkeypatch.setattr(text_trainer.time, "monotonic", lambda: clock[0])
    # mb loggers do not propagate, so caplog cannot see them; record calls directly.
    monkeypatch.setattr(text_trainer, "logger", rec)
    report = text_trainer.cli_progress_logger(min_interval=30.0)
    report("step a", 0.1)  # first step line always logs
    clock[0] += 5
    report("step b", 0.2)  # throttled
    report("phase", None)  # phase changes always log
    clock[0] += 31
    report("step c", 0.3)
    report("done", 1.0)  # completion always logs
    joined = " | ".join(rec.messages)
    assert "step a" in joined and "step b" not in joined
    assert "phase" in joined and "step c" in joined and "done" in joined
