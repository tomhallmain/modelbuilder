"""Tests for ``mb.evaluate.classification.text_metrics`` against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from mb.evaluate.classification.text_metrics import (
    average_precision,
    binary_metrics,
    expected_calibration_error,
    length_bucket_codes,
    recall_slices,
    roc_auc,
    spearman,
    threshold_for_precision,
    threshold_for_recall,
)

Y = np.array([1, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.1])


def test_average_precision_step_definition() -> None:
    # Thresholds 0.9/0.8/0.7/0.1 give (P, R) = (1, .5), (.5, .5), (2/3, 1), (.5, 1).
    assert average_precision(Y, S) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_average_precision_ties_form_one_threshold() -> None:
    assert average_precision(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_roc_auc_counts_pairs_and_ties() -> None:
    assert roc_auc(Y, S) == pytest.approx(0.75)
    assert roc_auc(np.array([1, 0]), np.array([0.3, 0.3])) == pytest.approx(0.5)
    assert roc_auc(np.array([1, 1]), np.array([0.3, 0.4])) is None


def test_threshold_policies() -> None:
    # precision >= 0.6 holds at 0.9 and 0.7; the lowest of those is 0.7.
    assert threshold_for_precision(Y, S, 0.6) == pytest.approx(0.7)
    assert threshold_for_precision(Y, S, 1.01) is None
    assert threshold_for_recall(Y, S, 0.5) == pytest.approx(0.9)
    assert threshold_for_recall(Y, S, 0.9) == pytest.approx(0.7)


def test_binary_metrics_confusion_and_levels() -> None:
    m = binary_metrics(Y, S, 0.75)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["recall_at_precision"]["0.90"] == pytest.approx(0.5)
    assert m["precision_at_recall"]["0.90"] == pytest.approx(2 / 3)
    assert m["brier"] == pytest.approx(np.mean((S - Y) ** 2))


def test_single_label_slice_reports_rates_only() -> None:
    m = binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.6, 0.9]), 0.5)
    assert "ap" not in m
    assert m["fpr"] == pytest.approx(2 / 3)
    assert m["recall"] is None


def test_ece_zero_when_bins_are_calibrated() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert expected_calibration_error(y, p) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([1, 1]), np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_length_buckets() -> None:
    codes = length_bucket_codes(["", "abcde", "abcdef", "x" * 30, "x" * 31])
    assert codes.tolist() == [0, 0, 1, 2, 3]


def test_recall_slices_skip_empty_keys_and_negatives() -> None:
    out = recall_slices(np.array([1, 1, 0, 1]), np.array([0.9, 0.1, 0.9, 0.8]), 0.5, ["topic_a", "topic_a", "topic_a", ""])
    assert out == {"topic_a": {"n_pos": 2, "recall": 0.5}}


def test_spearman_ignores_nan() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0, np.nan]), np.array([10.0, 20.0, 30.0, 1.0])) == pytest.approx(1.0)
