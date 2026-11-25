"""
The fine-tuning schedule's floor (unfrozen_lr_min), resuming with optimizer, scheduler and
scaler state, and the fp16 loss-scaler call pattern.

The central resume check is behavioral: a run interrupted after epoch k and resumed must end
with the same weights as the same run left uninterrupted. That only holds if the optimizer
moments and the learning-rate schedule both carry over.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")
nn = torch.nn

from mb.models.frameworks.pytorch.optimization import (  # noqa: E402
    OptimizationSettings,
    build_one_cycle,
    build_optimizer,
    load_optimizer_moments,
    scheduler_checkpoint_state,
)

pytestmark = pytest.mark.requires_torch


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(torch.relu(self.backbone(x)))


def _batches(n: int = 3):
    g = torch.Generator().manual_seed(0)
    return [
        (torch.randn(4, 4, generator=g), torch.randint(0, 2, (4,), generator=g))
        for _ in range(n)
    ]


def _trainer():
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    return PyTorchTrainer(device="cpu")


def _run(out: Path, hyperparams: dict, resume: Path | None = None, seed: int = 0) -> _Net:
    torch.manual_seed(seed)
    model = _Net()
    hp = {"frozen_lr": 0.01, "store_checkpoints": True, **hyperparams}
    _trainer().train(
        model=model,
        train_loader=_batches(),  # type: ignore[arg-type]
        val_loader=_batches(1),  # type: ignore[arg-type]
        hyperparams=hp,
        output_dir=out,
        resume_from_checkpoint=resume,
    )
    return model


def _same_weights(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.allclose(x, y, atol=1e-7) for x, y in zip(a.state_dict().values(),
                                                                 b.state_dict().values()))


# --- unfrozen_lr_min ---


def _final_lrs(optimizer, scheduler, total: int) -> list[float]:
    """The rates the last of *total* optimizer steps runs at (the schedule's step total-1)."""
    for _ in range(total - 1):
        optimizer.step()
        scheduler.step()
    return [g["lr"] for g in optimizer.param_groups]


def test_schedule_ends_at_unfrozen_lr_min() -> None:
    model = _Net()
    optimizer, lrs = build_optimizer(model, 3e-4, OptimizationSettings())
    scheduler = build_one_cycle(optimizer, lrs, 3e-4, 1e-5, total_steps=20, pct_start=0.05)
    # Warmup still starts at max / 25.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 25)
    assert _final_lrs(optimizer, scheduler, 20)[0] == pytest.approx(1e-5, rel=1e-6)


def test_schedule_floor_scales_with_layer_decay_groups() -> None:
    """Each group ends at min * (its peak / the top peak)."""
    model = _Net()
    optimizer = torch.optim.Adam(
        [{"params": model.backbone.parameters(), "lr": 0.5}, {"params": model.fc.parameters(), "lr": 1.0}]
    )
    scheduler = build_one_cycle(optimizer, [0.5, 1.0], 1.0, 0.01, total_steps=10, pct_start=0.1)
    low, top = _final_lrs(optimizer, scheduler, 10)
    assert top == pytest.approx(0.01, rel=1e-6)
    assert low == pytest.approx(0.005, rel=1e-6)


@pytest.mark.parametrize("bad_min", [None, 0.0, -1.0, 3e-4, 1.0])
def test_unusable_lr_min_falls_back_to_library_ending(bad_min) -> None:
    model = _Net()
    optimizer, lrs = build_optimizer(model, 3e-4, OptimizationSettings())
    scheduler = build_one_cycle(optimizer, lrs, 3e-4, bad_min, total_steps=10, pct_start=0.1)
    assert _final_lrs(optimizer, scheduler, 10)[0] == pytest.approx(3e-4 / 25 / 1e4, rel=1e-6)


@pytest.mark.parametrize("total, pct", [(20, 0.05), (10, 0.1), (100, 0.01)])
def test_warmup_ending_exactly_at_step_zero_does_not_divide_by_zero(total, pct) -> None:
    """OneCycleLR divides by zero when pct_start * total_steps == 1; the builder avoids it."""
    model = _Net()
    optimizer, lrs = build_optimizer(model, 3e-4, OptimizationSettings())
    scheduler = build_one_cycle(optimizer, lrs, 3e-4, 1e-5, total_steps=total, pct_start=pct)
    assert _final_lrs(optimizer, scheduler, total)[0] == pytest.approx(1e-5, rel=1e-4)


# --- Checkpoint state helpers ---


def test_scheduler_state_survives_weights_only_load(tmp_path: Path) -> None:
    model = _Net()
    optimizer, lrs = build_optimizer(model, 3e-4, OptimizationSettings())
    scheduler = build_one_cycle(optimizer, lrs, 3e-4, 1e-5, total_steps=10, pct_start=0.1)
    optimizer.step()
    scheduler.step()
    path = tmp_path / "s.pth"
    torch.save({"scheduler_state_dict": scheduler_checkpoint_state(scheduler)}, path)
    loaded = torch.load(path, weights_only=True)["scheduler_state_dict"]
    assert loaded["total_steps"] == 10
    assert loaded["last_epoch"] == 1


def test_load_optimizer_moments_keeps_this_runs_settings() -> None:
    model = _Net()
    old = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.5)
    model(torch.randn(2, 4)).sum().backward()
    old.step()
    new = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    load_optimizer_moments(new, old.state_dict())
    assert new.param_groups[0]["lr"] == 0.001
    assert new.param_groups[0]["weight_decay"] == 0.0
    first = next(iter(model.parameters()))
    assert torch.equal(new.state[first]["exp_avg"], old.state[first]["exp_avg"])


def test_load_optimizer_moments_rejects_mismatched_groups() -> None:
    model = _Net()
    two_groups = torch.optim.Adam(
        [{"params": model.backbone.parameters()}, {"params": model.fc.parameters()}]
    )
    one_group = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError):
        load_optimizer_moments(one_group, two_groups.state_dict())


# --- Resume equals an uninterrupted run ---


def test_resumed_fine_tuning_matches_uninterrupted_run(tmp_path: Path) -> None:
    hp = {"frozen_epochs": 0, "unfrozen_epochs": 4, "unfrozen_lr_max": 0.01, "unfrozen_lr_min": 1e-4}
    full = _run(tmp_path / "full", hp)
    resumed = _run(
        tmp_path / "resumed", hp, resume=tmp_path / "full" / "checkpoint_epoch_2.pth", seed=123
    )
    assert _same_weights(full, resumed)


def test_resumed_frozen_phase_matches_uninterrupted_run(tmp_path: Path) -> None:
    hp = {"frozen_epochs": 3, "unfrozen_epochs": 0}
    full = _run(tmp_path / "full", hp)
    resumed = _run(
        tmp_path / "resumed", hp, resume=tmp_path / "full" / "checkpoint_epoch_1.pth", seed=123
    )
    assert _same_weights(full, resumed)


def test_resume_across_phases_matches_uninterrupted_run(tmp_path: Path) -> None:
    """Resuming from the last frozen checkpoint starts fine-tuning fresh, exactly as before."""
    hp = {"frozen_epochs": 2, "unfrozen_epochs": 2, "unfrozen_lr_max": 0.01}
    full = _run(tmp_path / "full", hp)
    resumed = _run(
        tmp_path / "resumed", hp, resume=tmp_path / "full" / "checkpoint_epoch_2.pth", seed=123
    )
    assert _same_weights(full, resumed)


def test_resume_from_checkpoint_without_saved_state_still_trains(tmp_path: Path) -> None:
    """Checkpoints written before this state was saved resume with a fresh optimizer."""
    hp = {"frozen_epochs": 0, "unfrozen_epochs": 3, "unfrozen_lr_max": 0.01}
    _run(tmp_path / "full", hp)
    old_style = tmp_path / "old.pth"
    state = torch.load(tmp_path / "full" / "checkpoint_epoch_1.pth", weights_only=True)
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"):
        state.pop(key, None)
    torch.save(state, old_style)
    _run(tmp_path / "resumed", hp, resume=old_style)
    assert (tmp_path / "resumed" / "checkpoint_epoch_3.pth").is_file()


def test_resume_with_a_changed_plan_restarts_the_schedule(tmp_path: Path) -> None:
    """A different epoch count cannot continue the saved schedule; it must still run."""
    _run(tmp_path / "full", {"frozen_epochs": 0, "unfrozen_epochs": 3, "unfrozen_lr_max": 0.01})
    _run(
        tmp_path / "resumed",
        {"frozen_epochs": 0, "unfrozen_epochs": 5, "unfrozen_lr_max": 0.01},
        resume=tmp_path / "full" / "checkpoint_epoch_2.pth",
    )
    assert (tmp_path / "resumed" / "checkpoint_epoch_5.pth").is_file()


def test_resume_with_changed_optimizer_groups_continues_fresh(tmp_path: Path) -> None:
    """Weight decay splits the parameter groups, so the saved optimizer no longer fits."""
    base = {"frozen_epochs": 0, "unfrozen_epochs": 3, "unfrozen_lr_max": 0.01}
    _run(tmp_path / "full", base)
    _run(
        tmp_path / "resumed",
        {**base, "optimizer": "adamw", "weight_decay": 0.05},
        resume=tmp_path / "full" / "checkpoint_epoch_1.pth",
    )
    assert (tmp_path / "resumed" / "checkpoint_epoch_3.pth").is_file()


# --- fp16 loss scaler ---


class _RecordingScaler:
    """Stands in for GradScaler on CPU, recording the calls the training step makes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def step(self, optimizer):
        self.calls.append("step")
        optimizer.step()

    def update(self):
        self.calls.append("update")


def test_scaler_scales_every_batch_and_steps_once_per_group() -> None:
    model = _Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _RecordingScaler()
    _trainer()._train_epoch(
        model, _batches(3), nn.CrossEntropyLoss(), optimizer, 1,
        grad_accum_steps=2, scaler=scaler,
    )
    assert scaler.calls == ["scale", "scale", "step", "update", "scale", "step", "update"]


def test_scaler_state_is_saved_and_restored(tmp_path: Path) -> None:
    trainer = _trainer()
    model = _Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    class _StatefulScaler(_RecordingScaler):
        def __init__(self, scale_value: float) -> None:
            super().__init__()
            self.value = scale_value

        def state_dict(self):
            return {"scale": self.value}

        def load_state_dict(self, state):
            self.value = state["scale"]

    trainer._save_checkpoint(
        model, optimizer, tmp_path, epoch=1, frozen_epochs_completed=1,
        unfrozen_epochs_completed=0, best_val_acc=0.0, phase="frozen",
        scaler=_StatefulScaler(1024.0),
    )
    state = torch.load(tmp_path / "checkpoint_epoch_1.pth", weights_only=True)
    fresh = _StatefulScaler(65536.0)
    assert trainer._restore_training_state(optimizer, fresh, state) is True
    assert fresh.value == 1024.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs CUDA")
def test_fp16_epoch_on_cuda() -> None:
    from mb.models.frameworks.pytorch.optimization import make_grad_scaler
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    trainer = PyTorchTrainer(device="cuda")
    model = _Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    loss, _acc = trainer._train_epoch(
        model, _batches(3), nn.CrossEntropyLoss(), optimizer, 1,
        autocast_dtype=torch.float16, scaler=make_grad_scaler(torch.float16),
        grad_accum_steps=2,
    )
    assert loss == loss  # not NaN
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
