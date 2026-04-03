"""Training orchestration and hyperparameter management."""

from mb.training.hyperparams import HyperparameterManager, get_training_hyperparams
from mb.training.run_args import TrainingRunArgs, load_training_run_args_json

__all__ = [
    "ModelTrainer",
    "HyperparameterManager",
    "get_training_hyperparams",
    "TrainingRunArgs",
    "load_training_run_args_json",
]


def __getattr__(name: str):
    # Lazy import avoids a cycle: mb.training.gui_progress → mb.training package
    # → (eager ModelTrainer) → pytorch.trainer → gui_progress while pytorch.trainer
    # is still loading. That breaks DataLoader workers (multiprocessing spawn).
    if name == "ModelTrainer":
        from mb.training.trainer import ModelTrainer

        return ModelTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
