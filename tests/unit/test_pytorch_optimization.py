"""
Optimizer, layer-wise decay, gradient accumulation, precision, and best-epoch restoration.

The central guarantee is that the defaults reproduce the trainer's original behavior, so
the first tests pin that down before the new options are exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for optimizer tests")
nn = torch.nn

from mb.models.frameworks.pytorch.optimization import (  # noqa: E402
    OptimizationSettings,
    build_optimizer,
    make_grad_scaler,
    optimizer_steps_per_epoch,
    resolve_autocast_dtype,
    resolve_optimization_settings,
)
from mb.models.types import MixedPrecisionMode, OptimizerType  # noqa: E402

pytestmark = pytest.mark.requires_torch


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


class _Encoder(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(n))


class _TinyTransformer(nn.Module):
    """Parameter names shaped like a SigLIP vision tower: embeddings, encoder.layers.N, head."""

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.embeddings = nn.Linear(4, 4)
        self.encoder = _Encoder(n_layers)
        self.post_layernorm = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, 2)

    def head_parameters(self):
        return self.classifier.parameters()

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.encoder.layers:
            x = block.norm(block.proj(x))
        return self.classifier(self.post_layernorm(x))


# --- Defaults reproduce the original trainer ---


def test_empty_hyperparams_resolve_to_original_behavior() -> None:
    assert resolve_optimization_settings({}) == OptimizationSettings()
    settings = OptimizationSettings()
    assert settings.optimizer == OptimizerType.ADAM
    assert settings.weight_decay == 0.0
    assert settings.layer_decay == 1.0
    assert settings.grad_accum_steps == 1
    assert settings.mixed_precision == MixedPrecisionMode.NONE
    assert settings.restore_best is False


def test_default_optimizer_is_single_group_adam_over_the_same_parameters() -> None:
    model = _TinyTransformer()
    optimizer, lrs = build_optimizer(
        model, 3e-4, OptimizationSettings(), apply_layer_decay=True
    )
    assert type(optimizer) is torch.optim.Adam
    assert lrs == [3e-4]
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 3e-4
    assert group["weight_decay"] == 0
    assert [id(p) for p in group["params"]] == [id(p) for p in model.parameters()]


def test_trainable_only_matches_the_frozen_phase_filter() -> None:
    model = _Head()
    model.fc.bias.requires_grad = False
    optimizer, _ = build_optimizer(model, 1e-3, OptimizationSettings(), trainable_only=True)
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [id(model.fc.weight)]


def test_adamw_without_weight_decay_does_not_inherit_its_library_default() -> None:
    """torch's AdamW defaults weight_decay to 0.01; an unset option must mean 0."""
    optimizer, _ = build_optimizer(
        _Head(), 1e-3, OptimizationSettings(optimizer=OptimizerType.ADAMW)
    )
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


# --- Weight decay and layer-wise decay ---


def test_weight_decay_skips_biases_and_norms() -> None:
    model = _TinyTransformer(n_layers=1)
    settings = OptimizationSettings(optimizer=OptimizerType.ADAMW, weight_decay=0.05)
    optimizer, _ = build_optimizer(model, 1e-3, settings)
    decay_of = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        expected = 0.05 if p.ndim > 1 and not name.endswith(".bias") else 0.0
        assert decay_of[id(p)] == expected, name
    # Every parameter is in exactly one group.
    assert sorted(decay_of) == sorted(id(p) for p in model.parameters())


def test_layer_decay_scales_by_depth() -> None:
    model = _TinyTransformer(n_layers=3)
    settings = OptimizationSettings(layer_decay=0.5)
    optimizer, lrs = build_optimizer(model, 1.0, settings, apply_layer_decay=True)
    lr_of = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    # Depths: embeddings 0, blocks 1..3, post-encoder and head 4. Scale = 0.5 ** (4 - depth).
    assert lr_of[id(model.embeddings.weight)] == pytest.approx(0.5 ** 4)
    for i, block in enumerate(model.encoder.layers):
        assert lr_of[id(block.proj.weight)] == pytest.approx(0.5 ** (3 - i))
    assert lr_of[id(model.post_layernorm.weight)] == pytest.approx(1.0)
    assert lr_of[id(model.classifier.weight)] == pytest.approx(1.0)
    assert lrs == [g["lr"] for g in optimizer.param_groups]


def test_layer_decay_is_not_applied_without_encoder_blocks() -> None:
    optimizer, lrs = build_optimizer(
        _Head(), 1e-3, OptimizationSettings(layer_decay=0.5), apply_layer_decay=True
    )
    assert lrs == [1e-3]
    assert len(optimizer.param_groups) == 1


def test_layer_decay_lists_drive_one_cycle_per_group() -> None:
    model = _TinyTransformer(n_layers=2)
    optimizer, lrs = build_optimizer(
        model, 1.0, OptimizationSettings(layer_decay=0.5), apply_layer_decay=True
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lrs, total_steps=10)
    peaks = [g["max_lr"] for g in optimizer.param_groups]
    assert peaks == lrs
    sched.step()


# --- Validation of malformed values ---


@pytest.mark.parametrize(
    "hp, field, expected",
    [
        ({"optimizer": "sgd"}, "optimizer", OptimizerType.ADAM),
        ({"mixed_precision": "int4"}, "mixed_precision", MixedPrecisionMode.NONE),
        ({"weight_decay": -1}, "weight_decay", 0.0),
        ({"weight_decay": "abc"}, "weight_decay", 0.0),
        ({"layer_decay": 0}, "layer_decay", 1.0),
        ({"layer_decay": 1.5}, "layer_decay", 1.0),
        ({"grad_accum_steps": 0}, "grad_accum_steps", 1),
        ({"grad_accum_steps": "x"}, "grad_accum_steps", 1),
        ({"optimizer": "AdamW"}, "optimizer", OptimizerType.ADAMW),
        ({"grad_accum_steps": "4"}, "grad_accum_steps", 4),
    ],
)
def test_malformed_values_fall_back(hp, field, expected) -> None:
    assert getattr(resolve_optimization_settings(hp), field) == expected


def test_optimizer_steps_count_the_trailing_partial_group() -> None:
    assert optimizer_steps_per_epoch(10, 1) == 10
    assert optimizer_steps_per_epoch(10, 4) == 3
    assert optimizer_steps_per_epoch(8, 4) == 2


# --- Precision ---


def test_precision_fallbacks_on_cpu() -> None:
    cpu = torch.device("cpu")
    assert resolve_autocast_dtype(MixedPrecisionMode.NONE, cpu) is None
    assert resolve_autocast_dtype(MixedPrecisionMode.FP16, cpu) is None
    assert resolve_autocast_dtype(MixedPrecisionMode.BF16, cpu) is torch.bfloat16
    assert make_grad_scaler(None) is None
    assert make_grad_scaler(torch.bfloat16) is None


# --- The training step ---


class _Loader(list):
    """A list of (inputs, targets) batches; len() and iteration are all _train_epoch needs."""


def _trainer():
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    return PyTorchTrainer(device="cpu")


def test_gradient_accumulation_matches_one_combined_batch() -> None:
    """Two accumulated batches of 2 produce the same update as one batch of 4."""
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 1, 0])
    reference = _Head()
    accumulated = _Head()
    accumulated.load_state_dict(reference.state_dict())
    criterion = nn.CrossEntropyLoss()
    trainer = _trainer()

    opt_ref = torch.optim.SGD(reference.parameters(), lr=0.1)
    trainer._train_epoch(reference, _Loader([(x, y)]), criterion, opt_ref, 1)

    opt_acc = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    trainer._train_epoch(
        accumulated,
        _Loader([(x[:2], y[:2]), (x[2:], y[2:])]),
        criterion,
        opt_acc,
        1,
        grad_accum_steps=2,
    )

    for a, b in zip(reference.parameters(), accumulated.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_accumulation_steps_the_scheduler_once_per_optimizer_step() -> None:
    model = _Head()
    batches = _Loader([(torch.randn(2, 4), torch.tensor([0, 1]))] * 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = optimizer_steps_per_epoch(len(batches), 2)
    # OneCycleLR raises if stepped past total_steps, so an over-count fails here.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=steps)
    _trainer()._train_epoch(
        model, batches, nn.CrossEntropyLoss(), optimizer, 1, scheduler, grad_accum_steps=2
    )
    assert scheduler.last_epoch == steps


def test_bf16_autocast_epoch_runs_on_cpu() -> None:
    model = _Head()
    batches = _Loader([(torch.randn(2, 4), torch.tensor([0, 1]))])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, _acc = _trainer()._train_epoch(
        model, batches, nn.CrossEntropyLoss(), optimizer, 1, autocast_dtype=torch.bfloat16
    )
    assert loss == loss  # not NaN


# --- Best-epoch weights ---


def test_restore_best_loads_the_matching_weights(tmp_path: Path) -> None:
    trainer = _trainer()
    model = _Head()
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "best.pth"
    trainer._save_best_weights(model, path, 0.9, epoch=3)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    trainer._restore_best_weights(model, path, 0.9)
    for k, v in model.state_dict().items():
        assert torch.equal(v, best_state[k])


def test_restore_best_ignores_weights_from_another_run(tmp_path: Path) -> None:
    trainer = _trainer()
    model = _Head()
    path = tmp_path / "best.pth"
    trainer._save_best_weights(model, path, 0.5, epoch=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    final_state = {k: v.clone() for k, v in model.state_dict().items()}
    trainer._restore_best_weights(model, path, 0.8)
    for k, v in model.state_dict().items():
        assert torch.equal(v, final_state[k])


def test_restore_best_without_a_file_keeps_final_weights(tmp_path: Path) -> None:
    model = _Head()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _trainer()._restore_best_weights(model, tmp_path / "missing.pth", 0.8)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_train_restores_best_epoch_when_enabled(monkeypatch, tmp_path: Path) -> None:
    """End to end through train(): the epoch with the highest metric supplies the weights."""
    from mb.models.frameworks.pytorch.trainer import BEST_WEIGHTS_FILENAME

    trainer = _trainer()
    model = _Head()
    metrics = iter([0.6, 0.9, 0.7])
    snapshots = {}

    def _fake_train_epoch(model, *args, **kwargs):
        with torch.no_grad():
            model.fc.weight.add_(1.0)
        return 0.1, 0.5

    def _fake_validate(model, *args, **kwargs):
        value = next(metrics)
        snapshots[value] = model.fc.weight.detach().clone()
        return 0.2, value

    monkeypatch.setattr(trainer, "_train_epoch", _fake_train_epoch)
    monkeypatch.setattr(trainer, "_validate", _fake_validate)
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda *a, **k: None)

    trainer.train(
        model=model,
        train_loader=[0],  # type: ignore[arg-type]
        val_loader=[0],  # type: ignore[arg-type]
        hyperparams={
            "frozen_epochs": 1,
            "unfrozen_epochs": 2,
            "frozen_lr": 0.001,
            "unfrozen_lr_max": 0.0003,
            "restore_best": True,
        },
        output_dir=tmp_path,
    )
    assert (tmp_path / BEST_WEIGHTS_FILENAME).is_file()
    assert torch.equal(model.fc.weight, snapshots[0.9])


def test_train_default_does_not_write_best_weights(monkeypatch, tmp_path: Path) -> None:
    from mb.models.frameworks.pytorch.trainer import BEST_WEIGHTS_FILENAME

    trainer = _trainer()
    monkeypatch.setattr(trainer, "_train_epoch", lambda *a, **k: (0.1, 0.5))
    monkeypatch.setattr(trainer, "_validate", lambda *a, **k: (0.2, 0.9))
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda *a, **k: None)
    trainer.train(
        model=_Head(),
        train_loader=[0],  # type: ignore[arg-type]
        val_loader=[0],  # type: ignore[arg-type]
        hyperparams={"frozen_epochs": 1, "unfrozen_epochs": 1, "frozen_lr": 0.001},
        output_dir=tmp_path,
    )
    assert not (tmp_path / BEST_WEIGHTS_FILENAME).exists()
