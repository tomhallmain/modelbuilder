"""
Post-hoc calibration of a text classifier's scores, fitted on ``val``.

Every calibrator maps an uncalibrated probability ``p`` (from a backend's
``predict_proba``) to a calibrated one and round-trips through ``calibrator.json``.
Temperature and Platt work on ``z = logit(p)``: temperature scaling is ``sigmoid(z / T)``,
Platt scaling ``sigmoid(a·z + b)``. Isotonic regression is a monotone step fit on ``p``,
interpolated linearly between steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np

from mb.models.types import TextCalibrationMethod
from mb.utils.translations import _

_EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _nll_logits(u: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy for logits *u*, computed without overflow."""
    return float(np.mean(np.logaddexp(0.0, u) - y * u))


class Calibrator:
    method: TextCalibrationMethod

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Calibrator":
        raise NotImplementedError

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "Calibrator":
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        return {"method": self.method.value, **self.params()}


class TemperatureCalibrator(Calibrator):
    method = TextCalibrationMethod.TEMPERATURE

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        z = logit(probs)
        y = np.asarray(labels, dtype=np.float64)
        # NLL is convex in a = 1/T; golden-section search over a bracket wide enough for
        # T in [0.02, 50].
        lo, hi = 0.02, 50.0
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - ratio * (hi - lo)
        d = lo + ratio * (hi - lo)
        fc, fd = _nll_logits(c * z, y), _nll_logits(d * z, y)
        for _ in range(200):
            if hi - lo < 1e-9:
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - ratio * (hi - lo)
                fc = _nll_logits(c * z, y)
            else:
                lo, c, fc = c, d, fd
                d = lo + ratio * (hi - lo)
                fd = _nll_logits(d * z, y)
        self.temperature = float(1.0 / ((lo + hi) / 2.0))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return sigmoid(logit(probs) / self.temperature)

    def params(self) -> Dict[str, Any]:
        return {"temperature": self.temperature}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "TemperatureCalibrator":
        return cls(float(params["temperature"]))


class PlattCalibrator(Calibrator):
    method = TextCalibrationMethod.PLATT

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        z = logit(probs)
        y = np.asarray(labels, dtype=np.float64)
        lam = 1e-6
        theta = np.array([1.0, 0.0])

        def loss(t: np.ndarray) -> float:
            return _nll_logits(t[0] * z + t[1], y) + 0.5 * lam * float(t @ t)

        current = loss(theta)
        n = len(z)
        for _ in range(100):
            q = sigmoid(theta[0] * z + theta[1])
            r = q - y
            w = q * (1.0 - q)
            g = np.array([np.dot(r, z), r.sum()]) / n + lam * theta
            h = np.array(
                [[np.dot(w, z * z), np.dot(w, z)], [np.dot(w, z), w.sum()]]
            ) / n + lam * np.eye(2)
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            scale = 1.0
            while scale > 1e-6:
                cand = theta - scale * step
                cand_loss = loss(cand)
                if cand_loss <= current:
                    break
                scale *= 0.5
            else:
                break
            converged = current - cand_loss < 1e-12
            theta, current = cand, cand_loss
            if converged:
                break
        self.a, self.b = float(theta[0]), float(theta[1])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * logit(probs) + self.b)

    def params(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "PlattCalibrator":
        return cls(float(params["a"]), float(params["b"]))


class IsotonicCalibrator(Calibrator):
    method = TextCalibrationMethod.ISOTONIC

    def __init__(self, x: Any = None, y: Any = None) -> None:
        self.x = np.asarray(x if x is not None else [0.0, 1.0], dtype=np.float64)
        self.y = np.asarray(y if y is not None else [0.0, 1.0], dtype=np.float64)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        p = np.asarray(probs, dtype=np.float64)
        lab = np.asarray(labels, dtype=np.float64)
        ux, inv = np.unique(p, return_inverse=True)
        weight = np.bincount(inv).astype(np.float64)
        mean = np.bincount(inv, weights=lab) / weight
        # Pool adjacent violators over the distinct scores.
        vals: list = []
        wts: list = []
        lo_idx: list = []
        hi_idx: list = []
        for i in range(len(ux)):
            vals.append(mean[i])
            wts.append(weight[i])
            lo_idx.append(i)
            hi_idx.append(i)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                hi = hi_idx[-1]
                vals.pop(); wts.pop(); lo_idx.pop(); hi_idx.pop()
                vals[-1], wts[-1], hi_idx[-1] = v, w, hi
        # Each block contributes its first and last score at the block value, which is all
        # linear interpolation needs.
        xs, ys = [], []
        for v, lo, hi in zip(vals, lo_idx, hi_idx):
            xs.append(ux[lo])
            ys.append(v)
            if hi != lo:
                xs.append(ux[hi])
                ys.append(v)
        self.x = np.asarray(xs, dtype=np.float64)
        self.y = np.asarray(ys, dtype=np.float64)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(probs, dtype=np.float64), self.x, self.y)

    def params(self) -> Dict[str, Any]:
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "IsotonicCalibrator":
        return cls(params["x"], params["y"])


_CALIBRATORS: Dict[TextCalibrationMethod, Type[Calibrator]] = {
    TextCalibrationMethod.TEMPERATURE: TemperatureCalibrator,
    TextCalibrationMethod.PLATT: PlattCalibrator,
    TextCalibrationMethod.ISOTONIC: IsotonicCalibrator,
}


def fit_calibrator(method: TextCalibrationMethod, probs: np.ndarray, labels: np.ndarray) -> Calibrator:
    if method not in _CALIBRATORS:
        raise ValueError(_("No calibrator for method '{m}'").format(m=method.value))
    labels = np.asarray(labels)
    if labels.min(initial=1) == labels.max(initial=0):
        raise ValueError(_("Calibration needs both labels in the val split."))
    return _CALIBRATORS[method]().fit(probs, labels)


def save_calibrator(cal: Calibrator, path: Path) -> None:
    Path(path).write_text(json.dumps(cal.to_json(), indent=2) + "\n", encoding="utf-8")


def load_calibrator(path: Path) -> Calibrator:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    method = TextCalibrationMethod.try_from(data.get("method"))
    if method is None or method not in _CALIBRATORS:
        raise ValueError(_("Unknown calibrator in {path}: {m!r}").format(path=path, m=data.get("method")))
    return _CALIBRATORS[method].from_params(data)
