"""Regression: PyTorch checkpoint epoch numbers advance sequentially."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.requires_torch
def test_pytorch_trainer_checkpoint_epochs_are_sequential(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required for trainer regression")
    pytest.importorskip("torchvision", reason="PyTorch vision stack required")

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    trainer = PyTorchTrainer(device="cpu")
    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(self.backbone(x))

    model = _TinyModel()
    # Minimal iterable loaders; the training internals are monkeypatched below.
    train_loader = [0]
    val_loader = [0]

    def _fake_train_epoch(*args, **kwargs):
        return 0.1, 75.0

    def _fake_validate(*args, **kwargs):
        return 0.2, 70.0

    saved_epochs: list[int] = []

    def _fake_save_checkpoint(*args, **kwargs):
        saved_epochs.append(int(kwargs["epoch"]))

    monkeypatch.setattr(trainer, "_train_epoch", _fake_train_epoch)
    monkeypatch.setattr(trainer, "_validate", _fake_validate)
    monkeypatch.setattr(trainer, "_save_checkpoint", _fake_save_checkpoint)

    trainer.train(
        model=model,
        train_loader=train_loader,  # type: ignore[arg-type]
        val_loader=val_loader,  # type: ignore[arg-type]
        hyperparams={
            "frozen_epochs": 3,
            "unfrozen_epochs": 2,
            "frozen_lr": 0.001,
            "unfrozen_lr_max": 0.0003,
            "unfrozen_lr_min": 0.00001,
        },
        output_dir=tmp_path / "models",
    )

    assert saved_epochs == [1, 2, 3, 4, 5]


@pytest.mark.requires_torch
def test_pytorch_trainer_resume_from_epoch4_continues_expected_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Resume state like a completed frozen epoch 4 checkpoint:
    - one remaining frozen epoch (5/5), then unfrozen epochs (1..2)
    - saved global epoch numbers should be 5, 6, 7.
    """
    torch = pytest.importorskip("torch", reason="PyTorch required for trainer regression")
    pytest.importorskip("torchvision", reason="PyTorch vision stack required")

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    trainer = PyTorchTrainer(device="cpu")

    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(self.backbone(x))

    model = _TinyModel()
    train_loader = [0]
    val_loader = [0]

    def _fake_train_epoch(*args, **kwargs):
        return 0.1, 75.0

    def _fake_validate(*args, **kwargs):
        return 0.2, 70.0

    def _fake_load_checkpoint(*args, **kwargs):
        return {
            "epoch": 4,
            "frozen_epochs_completed": 4,
            "unfrozen_epochs_completed": 0,
            "best_val_acc": 61.6796,
        }

    save_events: list[tuple[int, int, int, str]] = []

    def _fake_save_checkpoint(*args, **kwargs):
        save_events.append(
            (
                int(kwargs["epoch"]),
                int(kwargs["frozen_epochs_completed"]),
                int(kwargs["unfrozen_epochs_completed"]),
                str(kwargs["phase"]),
            )
        )

    resume_ckpt = tmp_path / "checkpoint_epoch_4.pth"
    resume_ckpt.write_bytes(b"stub")

    monkeypatch.setattr(trainer, "_train_epoch", _fake_train_epoch)
    monkeypatch.setattr(trainer, "_validate", _fake_validate)
    monkeypatch.setattr(trainer, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(trainer, "_save_checkpoint", _fake_save_checkpoint)

    trainer.train(
        model=model,
        train_loader=train_loader,  # type: ignore[arg-type]
        val_loader=val_loader,  # type: ignore[arg-type]
        hyperparams={
            "frozen_epochs": 5,
            "unfrozen_epochs": 2,
            "frozen_lr": 0.001,
            "unfrozen_lr_max": 0.0003,
            "unfrozen_lr_min": 0.00001,
        },
        output_dir=tmp_path / "models",
        resume_from_checkpoint=resume_ckpt,
    )

    assert save_events == [
        (5, 5, 0, "frozen"),
        (6, 5, 1, "unfrozen"),
        (7, 5, 2, "unfrozen"),
    ]
