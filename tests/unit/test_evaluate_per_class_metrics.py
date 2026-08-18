"""
Per-class precision/recall/F1 derived from a confusion matrix.

Covers the zero-denominator conventions specifically: a class absent from the evaluation
split and a class the model never predicts are both real cases for an imbalanced
multi-harm taxonomy, and neither may raise or produce NaN.
"""

from __future__ import annotations

from pathlib import Path

from mb.evaluate._contracts import ClassificationMetricsReport
from mb.models.types import FrameworkType, ModelType


def _report(class_names: list[str], cm: list[list[int]]) -> ClassificationMetricsReport:
    n_samples = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    return ClassificationMetricsReport(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.PYTORCH,
        model_path=Path("model.pth"),
        data_dir=Path("data"),
        n_samples=n_samples,
        class_names=list(class_names),
        accuracy_percent=100.0 * correct / max(n_samples, 1),
        per_class_correct=[cm[i][i] for i in range(len(cm))],
        per_class_total=[sum(row) for row in cm],
        confusion_matrix=cm,
    )


def test_per_class_metrics_hand_computed() -> None:
    """Two classes, known confusion matrix, values checked against hand arithmetic."""
    # a: 8 correct, 2 predicted as b. b: 9 correct, 1 predicted as a.
    report = _report(["a", "b"], [[8, 2], [1, 9]])
    by_name = {m.name: m for m in report.per_class_metrics()}

    a = by_name["a"]
    assert a.support == 10
    assert a.predicted == 9  # 8 true a + 1 b misread as a
    assert a.true_positives == 8
    assert a.precision == 8 / 9
    assert a.recall == 8 / 10
    assert a.f1 == 2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8)

    b = by_name["b"]
    assert b.support == 10
    assert b.predicted == 11
    assert b.precision == 9 / 11
    assert b.recall == 9 / 10


def test_zero_support_class_is_zero_not_nan_and_excluded_from_averages() -> None:
    """
    A class with no samples in the split scores 0.0 and does not drag the averages down.

    This is the ``gore``-style case: the class exists in the taxonomy but the evaluation
    split may not contain any of it. Averaging a 0.0 in would understate the model by an
    amount that depends only on how the split was built.
    """
    report = _report(["a", "b", "gore"], [[8, 2, 0], [1, 9, 0], [0, 0, 0]])
    metrics = report.per_class_metrics()
    gore = next(m for m in metrics if m.name == "gore")

    assert gore.support == 0
    assert gore.predicted == 0
    assert gore.precision == 0.0
    assert gore.recall == 0.0
    assert gore.f1 == 0.0

    macro = report.macro_averages()
    assert macro is not None
    assert macro.n_classes == 2  # gore excluded
    scored = [m for m in metrics if m.support > 0]
    assert macro.f1 == sum(m.f1 for m in scored) / 2


def test_class_never_predicted_has_zero_precision_without_dividing_by_zero() -> None:
    """Support > 0 but predicted == 0 — distinguishable from the zero-support case."""
    # Every true "rare" sample is predicted as "common"; nothing is ever predicted "rare".
    report = _report(["common", "rare"], [[10, 0], [5, 0]])
    by_name = {m.name: m for m in report.per_class_metrics()}

    rare = by_name["rare"]
    assert rare.support == 5
    assert rare.predicted == 0
    assert rare.precision == 0.0
    assert rare.recall == 0.0
    assert rare.f1 == 0.0

    # Still averaged over, unlike a zero-support class: the model was measured on it.
    macro = report.macro_averages()
    assert macro is not None
    assert macro.n_classes == 2


def test_weighted_average_favours_the_larger_class() -> None:
    report = _report(["big", "small"], [[90, 10], [0, 10]])
    macro = report.macro_averages()
    weighted = report.weighted_averages()
    assert macro is not None and weighted is not None

    by_name = {m.name: m for m in report.per_class_metrics()}
    expected = (by_name["big"].f1 * 100 + by_name["small"].f1 * 10) / 110
    assert weighted.f1 == expected
    assert weighted.f1 > macro.f1  # the big class is the one the model does well on


def test_empty_confusion_matrix_yields_no_metrics() -> None:
    report = _report(["a"], [])
    assert report.per_class_metrics() == []
    assert report.macro_averages() is None
    assert report.weighted_averages() is None


def test_jsonable_includes_derived_metrics() -> None:
    report = _report(["a", "b"], [[8, 2], [1, 9]])
    payload = report.to_jsonable()

    assert [m["name"] for m in payload["per_class_metrics"]] == ["a", "b"]
    assert payload["macro_avg"]["n_classes"] == 2
    assert payload["weighted_avg"] is not None
    # Pre-existing keys are untouched so anything already reading this stays working.
    assert payload["confusion_matrix"] == [[8, 2], [1, 9]]
    assert payload["per_class_total"] == [10, 10]
