"""
Multi-label support must be inert unless explicitly asked for.

Every default resolves to single-label, and the single-label code paths behave exactly as
they did before label mode existed. These are regression guards, not feature tests: the
multi-label capability was added ahead of any dataset that uses it, so the failure mode to
protect against is it leaking into existing runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.data.label_schema import LABEL_SCHEMA_FILENAME
from mb.models.types import (
    ImageClassificationHandler,
    LabelMode,
    ModelType,
    MultiLabelImageClassificationHandler,
    get_classification_handler,
    get_model_type_handler,
)
from mb.training.run_args import TrainingRunArgs


def test_label_mode_defaults_to_single_label() -> None:
    assert LabelMode.get_default() == LabelMode.SINGLE_LABEL
    assert LabelMode.try_from(None) is None
    assert LabelMode.try_from("multi_label") == LabelMode.MULTI_LABEL
    assert LabelMode.try_from("nonsense") is None


def test_pipeline_default_config_ships_single_label() -> None:
    from mb.pipeline_config import PipelineConfig

    assert PipelineConfig().get("data.label_mode") == LabelMode.SINGLE_LABEL.value


def test_handler_resolution_is_unchanged_without_multi_label() -> None:
    """Every mode but multi-label resolves to exactly the handler callers had before."""
    default = get_model_type_handler(ModelType.IMAGE_CLASSIFICATION)

    assert get_classification_handler(ModelType.IMAGE_CLASSIFICATION) is default
    assert (
        get_classification_handler(ModelType.IMAGE_CLASSIFICATION, LabelMode.SINGLE_LABEL)
        is default
    )
    assert isinstance(default, ImageClassificationHandler)
    assert not isinstance(default, MultiLabelImageClassificationHandler)


def test_multi_label_resolves_to_its_own_handler() -> None:
    handler = get_classification_handler(
        ModelType.IMAGE_CLASSIFICATION, LabelMode.MULTI_LABEL
    )
    assert isinstance(handler, MultiLabelImageClassificationHandler)


def test_label_mode_does_not_make_other_model_types_resolvable() -> None:
    """
    Label mode is a classification concept and does not widen what else is supported.

    A model type with no handler still raises, rather than falling through to the
    multi-label one because a label mode happened to be supplied.
    """
    with pytest.raises(ValueError):
        get_classification_handler(ModelType.IMAGE_GENERATION_LORA, LabelMode.MULTI_LABEL)


def _run_args(**overrides) -> TrainingRunArgs:
    from mb.models.types import ArchitectureType, FrameworkType

    base = dict(
        framework=FrameworkType.PYTORCH,
        architecture=ArchitectureType.RESNET34,
        data_dir=Path("data"),
        output_dir=Path("out"),
        resume_from=None,
        run_id=None,
        update_snapshot=True,
        cli_hyperparams={},
    )
    base.update(overrides)
    return TrainingRunArgs(**base)


def test_training_run_args_defaults_to_single_label() -> None:
    args = _run_args()
    assert args.label_mode == LabelMode.SINGLE_LABEL
    assert args.to_json_dict()["label_mode"] == "single_label"


def test_run_args_json_without_label_mode_loads_as_single_label() -> None:
    """Args JSON written before label mode existed must still deserialize."""
    legacy = {
        "framework": "pytorch",
        "architecture": "resnet34",
        "data_dir": "data",
        "output_dir": "out",
        "resume_from": None,
        "run_id": None,
        "update_snapshot": True,
        "cli_hyperparams": {},
    }
    assert TrainingRunArgs.from_json_dict(legacy).label_mode == LabelMode.SINGLE_LABEL


def test_run_args_json_round_trips_multi_label() -> None:
    """The GUI trains in a subprocess via this JSON, so label mode has to survive it."""
    args = _run_args(label_mode=LabelMode.MULTI_LABEL)
    restored = TrainingRunArgs.from_json_dict(json.loads(json.dumps(args.to_json_dict())))
    assert restored.label_mode == LabelMode.MULTI_LABEL


def test_metrics_request_defaults_to_single_label() -> None:
    from mb.evaluate._contracts import MetricsRequest

    req = MetricsRequest(
        model_path=Path("m.pth"),
        data_dir=Path("d"),
        model_type=ModelType.IMAGE_CLASSIFICATION,
    )
    assert req.label_mode == LabelMode.SINGLE_LABEL
    assert req.tune_thresholds is False


def test_multi_label_handler_rejects_a_dataset_without_a_schema(tmp_path: Path) -> None:
    """A single-label dataset is not silently treated as multi-label."""
    for split in ("train", "test"):
        (tmp_path / split / "a").mkdir(parents=True)
    handler = get_classification_handler(
        ModelType.IMAGE_CLASSIFICATION, LabelMode.MULTI_LABEL
    )
    assert handler.validate_data(tmp_path) is False


def test_multi_label_handler_counts_labels_from_the_schema(tmp_path: Path) -> None:
    """
    Label count comes from the schema, not the folder count.

    A label that only ever appears in the manifest has no folder of its own, so counting
    directories would size the output layer too small.
    """
    for split in ("train", "test"):
        (tmp_path / split / "neutral").mkdir(parents=True)
    (tmp_path / LABEL_SCHEMA_FILENAME).write_text(
        json.dumps({"labels": ["neutral", "sexy", "gore"]}), encoding="utf-8"
    )
    handler = get_classification_handler(
        ModelType.IMAGE_CLASSIFICATION, LabelMode.MULTI_LABEL
    )

    assert handler.validate_data(tmp_path) is True
    assert handler.get_num_classes(tmp_path) == 3
    # The single-label handler still counts folders, unchanged.
    assert get_model_type_handler(ModelType.IMAGE_CLASSIFICATION).get_num_classes(tmp_path) == 1


def test_multi_label_handler_rejects_a_folder_that_is_not_a_label(tmp_path: Path) -> None:
    for split in ("train", "test"):
        (tmp_path / split / "unlisted").mkdir(parents=True)
    (tmp_path / LABEL_SCHEMA_FILENAME).write_text(
        json.dumps({"labels": ["neutral"]}), encoding="utf-8"
    )
    handler = get_classification_handler(
        ModelType.IMAGE_CLASSIFICATION, LabelMode.MULTI_LABEL
    )
    assert handler.validate_data(tmp_path) is False


def test_export_contract_stays_single_label_without_a_schema(tmp_path: Path) -> None:
    from mb.export.bundle import _resolve_output_contract

    contract = _resolve_output_contract(tmp_path, ["a", "b"])
    assert contract.label_mode == "single_label"
    assert contract.activation == "softmax"
    assert contract.thresholds is None


def test_export_contract_follows_a_schema_when_present(tmp_path: Path) -> None:
    """A model trained multi-label must not be exported as if it were softmax."""
    from mb.export.bundle import _resolve_output_contract

    (tmp_path / LABEL_SCHEMA_FILENAME).write_text(
        json.dumps(
            {
                "labels": ["neutral", "gore"],
                "default_thresholds": {"gore": 0.35},
                "axes": {"violence": {"ordered": False, "labels": ["gore"]}},
            }
        ),
        encoding="utf-8",
    )
    contract = _resolve_output_contract(tmp_path, ["neutral", "gore"])

    assert contract.label_mode == "multi_label"
    assert contract.activation == "sigmoid"
    # Every label carries an explicit threshold, including the one using the default.
    assert contract.thresholds == {"neutral": 0.5, "gore": 0.35}
    assert contract.axes is not None


def test_unreadable_schema_fails_rather_than_downgrading(tmp_path: Path) -> None:
    """Silently exporting as single-label would misdescribe the model."""
    from mb.export.bundle import _resolve_output_contract

    (tmp_path / LABEL_SCHEMA_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _resolve_output_contract(tmp_path, ["a"])
