"""
Binary-classifier metrics, slices and threshold selection for text classification.

Plain numpy so every backend is scored by identical code. Definitions:

* Average precision: ``sum_n (R_n - R_{n-1}) * P_n`` over distinct thresholds, highest
  first (the step-wise definition scikit-learn uses; no interpolation).
* ROC-AUC: Mann-Whitney U with tie-averaged ranks.
* A row is flagged at threshold ``t`` when ``score >= t``.
* ECE: 15 equal-width bins on [0, 1], weighted by bin size.

Values that are undefined (e.g. AP with no positives) are ``None`` so the result is valid
JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PRECISION_LEVELS = (0.90, 0.95)
RECALL_LEVELS = (0.90, 0.95)
ECE_BINS = 15
LENGTH_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("1-5", 0, 5),
    ("6-15", 6, 15),
    ("16-30", 16, 30),
    (">30", 31, None),
)


def _level_key(x: float) -> str:
    return f"{x:.2f}"


def _clean(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


@dataclass
class PrCurve:
    """Precision/recall at each distinct threshold, highest threshold first."""

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    n_pos: int


def pr_curve(labels: np.ndarray, scores: np.ndarray) -> PrCurve:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp_all = np.cumsum(y)
    fp_all = np.cumsum(1 - y)
    # Last index of each run of equal scores: everything up to it is flagged at that score.
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1] if len(s) else np.zeros(0, dtype=np.int64)
    tp, fp = tp_all[ends], fp_all[ends]
    n_pos = int(y.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos if n_pos else np.zeros(len(tp), dtype=np.float64)
    return PrCurve(s[ends], precision, recall, tp, fp, n_pos)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    c = pr_curve(labels, scores)
    if c.n_pos == 0:
        return None
    return float(np.sum(np.diff(c.recall, prepend=0.0) * c.precision))


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing their average rank."""
    x = np.asarray(x)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1] if n else np.zeros(0, dtype=np.int64)
    ends = np.r_[starts[1:], n] if n else np.zeros(0, dtype=np.int64)
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = np.asarray(labels, dtype=np.int64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = average_ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < 2:
        return None
    ra, rb = average_ranks(a[keep]), average_ranks(b[keep])
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return None if denom == 0 else float((ra * rb).sum() / denom)


def recall_at_precision(c: PrCurve, target: float) -> float:
    ok = c.precision >= target
    return float(c.recall[ok].max()) if ok.any() else 0.0


def precision_at_recall(c: PrCurve, target: float) -> float:
    ok = c.recall >= target
    return float(c.precision[ok].max()) if ok.any() else 0.0


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    d = np.asarray(probs, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    return float(np.mean(d * d))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    bins = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    nz = count > 0
    gap = np.abs(sum_y[nz] / count[nz] - sum_p[nz] / count[nz])
    return float(np.sum(gap * count[nz]) / len(p))


def confusion_at(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, int]:
    y = np.asarray(labels, dtype=bool)
    flag = np.asarray(probs, dtype=np.float64) >= threshold
    return {
        "tn": int(np.sum(~y & ~flag)),
        "fp": int(np.sum(~y & flag)),
        "fn": int(np.sum(y & ~flag)),
        "tp": int(np.sum(y & flag)),
    }


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """
    Ranking, calibration and threshold metrics for one evaluation set.

    Ranking/calibration metrics need both labels; with only one label present the entry
    still carries counts, ``flag_rate`` and whichever of ``recall``/``fpr`` is defined.
    """
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    cm = confusion_at(y, p, threshold)
    out: Dict[str, Any] = {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(threshold),
        "flag_rate": _clean(np.mean(p >= threshold)) if n else None,
        "recall": _clean(cm["tp"] / n_pos) if n_pos else None,
        "fpr": _clean(cm["fp"] / n_neg) if n_neg else None,
        "precision": _clean(cm["tp"] / (cm["tp"] + cm["fp"])) if (cm["tp"] + cm["fp"]) else None,
        "confusion": cm,
    }
    if n_pos and n_neg:
        c = pr_curve(y, p)
        out.update(
            {
                "ap": _clean(average_precision(y, p)),
                "roc_auc": _clean(roc_auc(y, p)),
                "recall_at_precision": {
                    _level_key(t): recall_at_precision(c, t) for t in PRECISION_LEVELS
                },
                "precision_at_recall": {
                    _level_key(t): precision_at_recall(c, t) for t in RECALL_LEVELS
                },
                "brier": brier_score(y, p),
                "ece": expected_calibration_error(y, p),
            }
        )
    return out


def length_bucket_codes(texts: Sequence[str]) -> np.ndarray:
    lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts))
    codes = np.zeros(len(lengths), dtype=np.int64)
    for i, (_name, lo, hi) in enumerate(LENGTH_BUCKETS):
        sel = lengths >= lo if hi is None else (lengths >= lo) & (lengths <= hi)
        codes[sel] = i
    return codes


def slice_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """:func:`binary_metrics` per distinct value of *keys* (same length as *labels*)."""
    keys_arr = np.asarray(keys, dtype=object)
    out: Dict[str, Dict[str, Any]] = {}
    for k in sorted(set(keys_arr.tolist())):
        sel = keys_arr == k
        out[str(k)] = binary_metrics(labels[sel], probs[sel], threshold)
    return out


def recall_slices(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Recall per non-empty key among positive rows (for noisy positive-only tags)."""
    keys_arr = np.asarray(keys, dtype=object)
    pos = np.asarray(labels) == 1
    flag = np.asarray(probs) >= threshold
    out: Dict[str, Dict[str, Any]] = {}
    for k in sorted(set(keys_arr[pos].tolist())):
        if k == "":
            continue
        sel = pos & (keys_arr == k)
        n = int(sel.sum())
        out[str(k)] = {"n_pos": n, "recall": _clean(flag[sel].mean()) if n else None}
    return out


def threshold_for_precision(labels: np.ndarray, probs: np.ndarray, target: float) -> Optional[float]:
    """Lowest threshold whose precision is at least *target*; ``None`` when none reaches it."""
    c = pr_curve(labels, probs)
    if c.n_pos == 0:
        return None
    ok = np.flatnonzero(c.precision >= target)
    return float(c.thresholds[ok[-1]]) if len(ok) else None


def threshold_for_recall(labels: np.ndarray, probs: np.ndarray, target: float) -> Optional[float]:
    """Highest threshold whose recall is at least *target*; ``None`` with no positives."""
    c = pr_curve(labels, probs)
    if c.n_pos == 0:
        return None
    ok = np.flatnonzero(c.recall >= target)
    return float(c.thresholds[ok[0]]) if len(ok) else None


def histogram(probs: np.ndarray, n_bins: int = 20) -> List[Dict[str, Any]]:
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    counts, edges = np.histogram(p, bins=n_bins, range=(0.0, 1.0))
    return [
        {"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(n_bins)
    ]
