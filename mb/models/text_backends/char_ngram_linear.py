"""
``char_ngram_linear``: TF-IDF over character n-grams + logistic regression (scikit-learn).

Character n-grams with ``analyzer="char_wb"`` see misspellings and glued stems that word
tokenizers miss. Case is kept (``lowercase=False``) because it carries signal.

TF-IDF is assembled from :class:`~sklearn.feature_extraction.text.CountVectorizer` plus an
explicit idf/normalize step (smooth idf, optional sublinear tf, L2 rows: the same maths as
``TfidfVectorizer``). Training and loading then share one code path, and the model is saved
as plain JSON/npz instead of a pickle tied to one scikit-learn version.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mb.cancellation import check_cancel_event
from mb.models.text_backends.base import ProgressFn, TextBackend, TextFitContext, TextFitData
from mb.models.types import TextBackendType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

_VOCAB_FILE = "vocabulary.json"
_IDF_FILE = "idf.npy"
_LINEAR_FILE = "linear.npz"
_PREDICT_CHUNK = 100_000


def _import_sklearn():
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import normalize
    except ImportError as e:
        raise ImportError(
            _(
                "The char_ngram_linear backend requires scikit-learn: "
                "pip install -e \".[text]\". Underlying error: {err}"
            ).format(err=e)
        ) from e
    return CountVectorizer, LogisticRegression, normalize


class CharNgramLinearBackend(TextBackend):
    name = TextBackendType.CHAR_NGRAM_LINEAR
    neural = False
    option_defaults = {
        "ngram_min": 1,
        "ngram_max": 5,
        # Ignore n-grams in fewer than this many training rows.
        "min_df": 2,
        # Keep at most this many n-grams (most frequent first); null = no cap.
        "max_features": 2_000_000,
        "sublinear_tf": True,
        # Inverse regularization strength; a list fits each value and keeps the best on val.
        "C": 1.0,
        "max_iter": 1000,
        "solver": "liblinear",
    }

    def __init__(self) -> None:
        self.options: Dict[str, Any] = dict(self.option_defaults)
        self.vocabulary: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self._counter = None

    def _make_counter(self, vocabulary: Optional[Dict[str, int]] = None):
        CountVectorizer, _lr, _norm = _import_sklearn()
        o = self.options
        return CountVectorizer(
            analyzer="char_wb",
            ngram_range=(int(o["ngram_min"]), int(o["ngram_max"])),
            lowercase=False,
            min_df=1 if vocabulary is not None else int(o["min_df"]),
            max_features=None if vocabulary is not None else o["max_features"],
            vocabulary=vocabulary,
            dtype=np.float32,
        )

    def _tfidf(self, counts):
        import scipy.sparse as sp

        _cv, _lr, normalize = _import_sklearn()
        x = counts.tocsr().astype(np.float32)
        if self.options["sublinear_tf"]:
            np.log(x.data, out=x.data)
            x.data += 1.0
        x = x @ sp.diags(self.idf.astype(np.float32))
        return normalize(x, norm="l2", copy=False)

    def _features(self, texts: Sequence[str]):
        return self._tfidf(self._counter.transform(texts))

    def fit(self, train: TextFitData, val: TextFitData, ctx: TextFitContext) -> Dict[str, Any]:
        _cv, LogisticRegression, _norm = _import_sklearn()
        self.options = dict(ctx.options)
        o = self.options
        if int(o["ngram_min"]) < 1 or int(o["ngram_max"]) < int(o["ngram_min"]):
            raise ValueError(_("backend_options: need 1 <= ngram_min <= ngram_max"))

        ctx.report(_("Counting character n-grams…"), 0.0)
        self._counter = self._make_counter()
        counts = self._counter.fit_transform(train.texts)
        check_cancel_event(ctx.cancel_event)
        self.vocabulary = {str(k): int(v) for k, v in self._counter.vocabulary_.items()}
        n_docs = counts.shape[0]
        df = np.bincount(counts.tocsr().indices, minlength=counts.shape[1]).astype(np.float64)
        self.idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)
        x_train = self._tfidf(counts)
        del counts
        ctx.report(_("Vectorizing val…"), 0.1)
        x_val = self._features(val.texts)
        logger.info("char_ngram_linear: %d features, %d train rows", len(self.vocabulary), n_docs)

        c_values = o["C"] if isinstance(o["C"], list) else [o["C"]]
        history: List[Dict[str, Any]] = []
        best = None
        for i, c in enumerate(c_values):
            check_cancel_event(ctx.cancel_event)
            ctx.report(
                _("Fitting logistic regression (C={c}, {i}/{n})…").format(c=c, i=i + 1, n=len(c_values)),
                0.2 + 0.7 * i / len(c_values),
            )
            clf = LogisticRegression(
                C=float(c),
                solver=str(o["solver"]),
                max_iter=int(o["max_iter"]),
                random_state=ctx.seed,
            )
            clf.fit(x_train, train.labels, sample_weight=train.weights)
            coef = clf.coef_.ravel().astype(np.float64)
            intercept = float(clf.intercept_[0])
            score = ctx.selection_metric(_sigmoid(x_val @ coef + intercept))
            history.append({"C": float(c), "val_metric": score})
            logger.info("char_ngram_linear: C=%s val metric %.6f", c, score)
            if best is None or score > best[0]:
                best = (score, float(c), coef, intercept)
        assert best is not None
        _score, best_c, self.coef, self.intercept = best
        ctx.report(_("Linear model fitted."), 1.0)
        return {"n_features": len(self.vocabulary), "selection": history, "chosen_C": best_c}

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[ProgressFn] = None,
    ) -> np.ndarray:
        out = np.empty(len(texts), dtype=np.float64)
        for start in range(0, len(texts), _PREDICT_CHUNK):
            check_cancel_event(cancel_event)
            chunk = texts[start : start + _PREDICT_CHUNK]
            out[start : start + len(chunk)] = _sigmoid(self._features(chunk) @ self.coef + self.intercept)
            if progress is not None:
                progress(_("Scoring…"), min(1.0, (start + len(chunk)) / max(len(texts), 1)))
        return out

    def metadata(self) -> Dict[str, Any]:
        return {"options": self.options, "n_features": len(self.vocabulary)}

    def save_weights(self, model_dir: Path) -> None:
        terms = [""] * len(self.vocabulary)
        for term, i in self.vocabulary.items():
            terms[i] = term
        (model_dir / _VOCAB_FILE).write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
        np.save(model_dir / _IDF_FILE, self.idf)
        np.savez(model_dir / _LINEAR_FILE, coef=self.coef, intercept=np.array([self.intercept]))

    @classmethod
    def load_weights(cls, model_dir: Path, meta: Dict[str, Any], device: str) -> CharNgramLinearBackend:
        b = cls()
        b.options = {**cls.option_defaults, **(meta.get("options") or {})}
        terms = json.loads((model_dir / _VOCAB_FILE).read_text(encoding="utf-8"))
        b.vocabulary = {t: i for i, t in enumerate(terms)}
        b.idf = np.load(model_dir / _IDF_FILE)
        lin = np.load(model_dir / _LINEAR_FILE)
        b.coef = lin["coef"].astype(np.float64)
        b.intercept = float(lin["intercept"][0])
        b._counter = b._make_counter(b.vocabulary)
        return b


def _sigmoid(x) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


BACKEND = CharNgramLinearBackend
