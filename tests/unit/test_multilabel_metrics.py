"""
Multi-label counting, averaging, positive weighting, and threshold tuning.

Pure arithmetic — no torch, no inference. The single-label conventions are mirrored
deliberately: a label with no positives is excluded from macro averages, and a zero
denominator yields 0.0 rather than NaN.
"""

from __future__ import annotations

import pytest

from mb.evaluate._multilabel import (
    MultiLabelCounter,
    f1_from_precision_recall,
    positive_weights,
    thresholds_for_labels,
)

# Three labels over four samples; the third label never occurs.
_TRUTH = [
    [True, False, False],
    [True, True, False],
    [False, True, False],
    [True, False, False],
]
_PRED = [
    [True, False, False],
    [True, False, False],
    [False, True, False],
    [False, False, False],
]


def _counter() -> MultiLabelCounter:
    counter = MultiLabelCounter(3)
    counter.add_batch(_PRED, _TRUTH)
    return counter


def test_counts_are_a_two_by_two_per_label() -> None:
    counts = _counter().counts
    assert (counts[0].true_positives, counts[0].false_positives) == (2, 0)
    assert (counts[0].false_negatives, counts[0].true_negatives) == (1, 1)
    assert counts[0].support == 3
    assert counts[0].predicted == 2


def test_per_label_rates() -> None:
    counts = _counter().counts
    assert counts[0].precision == 1.0
    assert counts[0].recall == pytest.approx(2 / 3)
    assert counts[1].precision == 1.0
    assert counts[1].recall == 0.5


def test_label_with_no_positives_is_zero_not_nan() -> None:
    counts = _counter().counts[2]
    assert counts.support == 0
    assert counts.predicted == 0
    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0


def test_macro_excludes_unsupported_labels_micro_pools_them() -> None:
    counter = _counter()
    assert counter.scored_label_count() == 2

    scored = [c for c in counter.counts if c.support > 0]
    assert counter.macro_f1() == pytest.approx(sum(c.f1 for c in scored) / 2)
    # Micro pools counts, so it leans on the label with more support.
    assert counter.micro_f1() == pytest.approx(0.75)


def test_axis_macro_restricts_to_its_members() -> None:
    counter = _counter()
    names = ["a", "b", "c"]
    only_a = counter.axis_macro_f1(names, ["a"])
    assert only_a == pytest.approx(counter.counts[0].f1)
    # An axis whose labels have no support has nothing to report.
    assert counter.axis_macro_f1(names, ["c"]) is None


def test_add_label_increments_matches_add_batch() -> None:
    """The framework-reduced path must agree with the per-row one."""
    by_row = _counter()

    tp = [2, 1, 0]
    fp = [0, 0, 0]
    fn = [1, 1, 0]
    tn = [1, 2, 4]
    reduced = MultiLabelCounter(3)
    reduced.add_label_increments(tp, fp, fn, tn, n_samples=4)

    assert reduced.micro_f1() == pytest.approx(by_row.micro_f1())
    assert reduced.macro_f1() == pytest.approx(by_row.macro_f1())
    assert reduced.n_samples == by_row.n_samples


def test_empty_counter_does_not_divide_by_zero() -> None:
    empty = MultiLabelCounter(2)
    assert empty.micro_f1() == 0.0
    assert empty.macro_f1() == 0.0
    assert empty.scored_label_count() == 0


def test_f1_helper() -> None:
    assert f1_from_precision_recall(1.0, 1.0) == 1.0
    assert f1_from_precision_recall(0.0, 0.0) == 0.0
    assert f1_from_precision_recall(1.0, 0.0) == 0.0


def test_positive_weights_favour_rare_labels() -> None:
    """
    Per-label negative-to-positive ratio, unlike the shared-softmax class weighting.

    Each label is its own binary problem, so a label present on most images is weighted
    *down* rather than being scaled against the other labels.
    """
    # Ceiling raised past the rarest label's ratio so the formula itself is what is checked;
    # the ceiling gets its own test below.
    weights = positive_weights([10, 500], n_samples=1000, max_weight=1000.0)
    assert weights[0] == pytest.approx((1000 - 10) / 10)
    assert weights[1] == pytest.approx((1000 - 500) / 500)
    assert weights[0] > weights[1]


def test_positive_weight_ceiling_is_absolute() -> None:
    weights = positive_weights([1], n_samples=10_000, max_weight=50.0)
    assert weights[0] == 50.0


def test_positive_weight_edge_cases() -> None:
    # No positives, and all-positive: neither has a meaningful ratio.
    assert positive_weights([0], n_samples=100, max_weight=50.0) == [1.0]
    assert positive_weights([100], n_samples=100, max_weight=50.0) == [1.0]


def test_thresholds_for_labels_falls_back() -> None:
    assert thresholds_for_labels(["a", "b"], {"a": 0.3}, 0.5) == [0.3, 0.5]


def test_threshold_tuning_picks_a_better_threshold_than_the_default() -> None:
    """
    A label whose scores sit below 0.5 is invisible at the default threshold.

    This is the case tuning exists for: a rare label the model ranks correctly but scores
    conservatively would otherwise be reported as never predicted.
    """
    from mb.data.label_schema import parse_label_schema
    from mb.evaluate.classification.image_multilabel_metrics import tune_thresholds

    schema = parse_label_schema({"labels": ["rare"], "default_threshold": 0.5})
    # Positives score 0.4, negatives 0.1 — separable, but not above 0.5.
    scores = [[0.4], [0.4], [0.1], [0.1]]
    truths = [[True], [True], [False], [False]]

    tuned = tune_thresholds(scores, truths, schema)
    assert tuned[0] <= 0.4
    assert tuned[0] > 0.1


def test_threshold_tuning_leaves_unsupported_labels_alone() -> None:
    from mb.data.label_schema import parse_label_schema
    from mb.evaluate.classification.image_multilabel_metrics import tune_thresholds

    schema = parse_label_schema(
        {"labels": ["absent"], "default_thresholds": {"absent": 0.42}}
    )
    tuned = tune_thresholds([[0.9], [0.8]], [[False], [False]], schema)
    assert tuned[0] == 0.42


@pytest.mark.requires_torch
def test_run_multilabel_metrics_end_to_end(
    multi_label_classification_data_dir, tmp_path
) -> None:
    """
    Real (untrained) inference through the multi-label metrics entry point.

    Covers the chain the pure-arithmetic tests above cannot: schema and manifest loading
    relative to the split, multi-hot truth assembly, per-label counting, and report
    construction — including the axis grouping.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from mb.evaluate._contracts import MetricsRequest
    from mb.evaluate.classification.image_multilabel_metrics import run_multilabel_metrics
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer
    from mb.models.types import LabelMode, ModelType

    torch.manual_seed(0)
    trainer = PyTorchTrainer(device="cpu")
    model = trainer.create_model("resnet18", num_classes=3, pretrained=False)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    report = run_multilabel_metrics(
        MetricsRequest(
            model_path=model_path,
            data_dir=multi_label_classification_data_dir / "test",
            model_type=ModelType.IMAGE_CLASSIFICATION,
            architecture="resnet18",
            image_size=64,
            batch_size=2,
            num_workers=0,
            device="cpu",
            label_mode=LabelMode.MULTI_LABEL,
        )
    )

    assert report.n_samples == 4
    assert report.label_names == ["class_a", "class_b", "extra"]
    assert len(report.per_label) == 3
    # Truth comes from folders plus the manifest: two per folder label, one from `extra`.
    supports = {m.name: m.support for m in report.per_label}
    assert supports == {"class_a": 2, "class_b": 2, "extra": 1}
    # `extra`'s threshold comes from the schema, not the global default.
    assert next(m.threshold for m in report.per_label if m.name == "extra") == 0.4
    assert report.n_labels_with_support == 3
    assert "grouped" in report.axis_macro_f1
    assert 0.0 <= report.micro_f1 <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0


@pytest.mark.requires_torch
def test_tune_thresholds_writes_back_to_the_schema(
    multi_label_classification_data_dir, tmp_path
) -> None:
    """Tuning is a calibration step that persists — the schema on disk changes."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from mb.data.label_schema import load_label_schema
    from mb.evaluate._contracts import MetricsRequest
    from mb.evaluate.classification.image_multilabel_metrics import run_multilabel_metrics
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer
    from mb.models.types import LabelMode, ModelType

    torch.manual_seed(0)
    trainer = PyTorchTrainer(device="cpu")
    model = trainer.create_model("resnet18", num_classes=3, pretrained=False)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    before = load_label_schema(multi_label_classification_data_dir)
    run_multilabel_metrics(
        MetricsRequest(
            model_path=model_path,
            data_dir=multi_label_classification_data_dir / "test",
            model_type=ModelType.IMAGE_CLASSIFICATION,
            architecture="resnet18",
            image_size=64,
            batch_size=2,
            num_workers=0,
            device="cpu",
            label_mode=LabelMode.MULTI_LABEL,
            tune_thresholds=True,
        )
    )
    after = load_label_schema(multi_label_classification_data_dir)

    # Every label gains an explicit threshold, and the label set is untouched.
    assert after.labels == before.labels
    assert set(after.default_thresholds) == set(after.labels)
    assert set(after.axes) == set(before.axes)
