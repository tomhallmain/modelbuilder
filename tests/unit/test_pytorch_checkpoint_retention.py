"""training.store_checkpoints: keep every epoch's checkpoint, or only the latest."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for checkpoint tests")

pytestmark = pytest.mark.requires_torch


def _save(trainer, model, optimizer, out: Path, epoch: int, keep_previous: bool) -> None:
    trainer._save_checkpoint(
        model,
        optimizer,
        out,
        epoch=epoch,
        frozen_epochs_completed=epoch,
        unfrozen_epochs_completed=0,
        best_val_acc=0.5,
        phase="frozen",
        keep_previous=keep_previous,
    )


def _setup():
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    trainer = PyTorchTrainer(device="cpu")
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return trainer, model, optimizer


def _names(out: Path) -> list[str]:
    return sorted(p.name for p in out.glob("checkpoint_epoch_*"))


def test_latest_only_keeps_one_checkpoint(tmp_path: Path) -> None:
    trainer, model, optimizer = _setup()
    for epoch in (1, 2, 3):
        _save(trainer, model, optimizer, tmp_path, epoch, keep_previous=False)
    assert _names(tmp_path) == ["checkpoint_epoch_3.json", "checkpoint_epoch_3.pth"]


def test_store_all_keeps_every_checkpoint(tmp_path: Path) -> None:
    trainer, model, optimizer = _setup()
    for epoch in (1, 2):
        _save(trainer, model, optimizer, tmp_path, epoch, keep_previous=True)
    assert _names(tmp_path) == [
        "checkpoint_epoch_1.json",
        "checkpoint_epoch_1.pth",
        "checkpoint_epoch_2.json",
        "checkpoint_epoch_2.pth",
    ]


def test_latest_only_never_removes_files_this_run_did_not_write(tmp_path: Path) -> None:
    """A checkpoint resumed from, or left by an earlier run, survives pruning."""
    trainer, model, optimizer = _setup()
    earlier = tmp_path / "checkpoint_epoch_4.pth"
    earlier.write_bytes(b"from an earlier run")
    trainer._last_checkpoint_path = None  # what train() sets at the start of a run
    for epoch in (5, 6):
        _save(trainer, model, optimizer, tmp_path, epoch, keep_previous=False)
    assert earlier.is_file()
    assert not (tmp_path / "checkpoint_epoch_5.pth").exists()
    assert (tmp_path / "checkpoint_epoch_6.pth").is_file()


def test_store_checkpoints_reaches_the_trainer_from_the_pipeline() -> None:
    from mb.pipeline_config import PipelineConfig

    assert PipelineConfig(config_path=None).training_hyperparams()["store_checkpoints"] is False
