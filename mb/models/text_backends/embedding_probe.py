"""
``embedding_probe``: a frozen sentence-embedding model plus a trained linear or MLP head.

Each text is wrapped in ``prompt_format`` (the default is the Qwen3-Embedding instruction
format) when ``instruction`` is non-empty, embedded with ``AutoModel``, pooled
(``last_token`` / ``mean`` / ``cls``) and optionally L2-normalized. The head is trained in
torch with the per-row class × tier weights and val-metric early stopping.

Embedding the whole training split is the expensive step, so embeddings are cached as
float16 ``.npy`` files under ``<runs_dir>/.embedding_cache/``, keyed by everything that
changes them (model, revision, prompt, pooling, normalization, length limit, and a hash of
the texts). Runs that only change the head reuse them.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from mb.cancellation import check_cancel_event
from mb.models.text_backends._torch_common import (
    EarlyStopper,
    import_torch,
    import_transformers,
    seed_everything,
    weighted_bce,
)
from mb.models.text_backends.base import ProgressFn, TextBackend, TextFitContext, TextFitData
from mb.models.types import TextBackendType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

_HEAD_FILE = "head.safetensors"
_CACHE_DIR = ".embedding_cache"
_POOLINGS = ("last_token", "mean", "cls")
_HEADS = ("linear", "mlp")


class EmbeddingProbeBackend(TextBackend):
    name = TextBackendType.EMBEDDING_PROBE
    neural = True
    default_model_id = "Qwen/Qwen3-Embedding-0.6B"
    default_lr = 1.0e-3
    default_epochs = 10
    default_batch_size = 1024
    option_defaults = {
        # Task description inserted into prompt_format; "" = embed the bare text.
        "instruction": "",
        "prompt_format": "Instruct: {instruction}\nQuery:{text}",
        "pooling": "last_token",
        "normalize": True,
        "head": "linear",
        "mlp_hidden": 256,
        "mlp_dropout": 0.1,
        "embed_batch_size": 256,
        # Load the embedding model in float16 on CUDA.
        "half_precision": True,
        "cache_embeddings": True,
    }

    def __init__(self) -> None:
        self.options: Dict[str, Any] = dict(self.option_defaults)
        self.model_id: Optional[str] = None
        self.revision: Optional[str] = None
        self.max_length = 64
        self.device = "cpu"
        self.encoder = None
        self.tokenizer = None
        self.head = None
        self.dim: Optional[int] = None
        self._prefix_tokens = 0

    # --- embedding model ---

    def _prefix(self) -> str:
        instr = str(self.options["instruction"])
        if not instr:
            return "{text}"
        return str(self.options["prompt_format"]).replace("{instruction}", instr)

    def _prompts(self, texts: Sequence[str]) -> list:
        prefix = self._prefix()
        return [prefix.replace("{text}", t) for t in texts]

    def _load_encoder(self) -> None:
        torch = import_torch()
        transformers = import_transformers()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if (self.device.startswith("cuda") and self.options["half_precision"]) else None
        # transformers renamed ``torch_dtype`` to ``dtype`` in 4.56; older releases need the old name.
        kwargs = {}
        if dtype is not None:
            major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
            kwargs = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype}
        self.encoder = transformers.AutoModel.from_pretrained(self.model_id, **kwargs).to(self.device)
        self.encoder.eval()
        self.revision = getattr(self.encoder.config, "_commit_hash", None)
        empty = self._prefix().replace("{text}", "")
        self._prefix_tokens = len(self.tokenizer(empty, add_special_tokens=False)["input_ids"])

    @property
    def _token_limit(self) -> int:
        return self.max_length + self._prefix_tokens

    def count_truncated(self, texts: Sequence[str]) -> Optional[int]:
        n = 0
        for start in range(0, len(texts), 10_000):
            enc = self.tokenizer(self._prompts(texts[start : start + 10_000]), truncation=False)
            n += sum(1 for ids in enc["input_ids"] if len(ids) > self._token_limit)
        return n

    def _pool(self, hidden, mask):
        torch = import_torch()
        pooling = self.options["pooling"]
        if pooling == "cls":
            return hidden[:, 0]
        if pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
        # last_token: the last attended position, whichever side the tokenizer pads on.
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        last = (mask * positions).argmax(dim=1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]

    def _embed_into(
        self,
        texts: Sequence[str],
        out: np.ndarray,
        cancel_event: Optional[threading.Event],
        progress: Optional[ProgressFn],
        label: str,
    ) -> None:
        torch = import_torch()
        bs = int(self.options["embed_batch_size"])
        with torch.inference_mode():
            for start in range(0, len(texts), bs):
                check_cancel_event(cancel_event)
                enc = self.tokenizer(
                    self._prompts(texts[start : start + bs]),
                    padding=True,
                    truncation=True,
                    max_length=self._token_limit,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.encoder(**enc).last_hidden_state
                vec = self._pool(hidden, enc["attention_mask"]).float()
                if self.options["normalize"]:
                    vec = torch.nn.functional.normalize(vec, dim=-1)
                out[start : start + vec.shape[0]] = vec.cpu().numpy()
                if progress is not None and (start // bs) % 50 == 0:
                    progress(
                        _("Embedding {what}: {i}/{n}").format(what=label, i=start, n=len(texts)),
                        start / max(len(texts), 1),
                    )

    def _cache_key(self, texts: Sequence[str]) -> str:
        h = hashlib.sha256()
        meta = {
            "model_id": self.model_id,
            "revision": self.revision,
            "prefix": self._prefix(),
            "pooling": self.options["pooling"],
            "normalize": bool(self.options["normalize"]),
            "half_precision": bool(self.options["half_precision"]),
            "token_limit": self._token_limit,
            "n": len(texts),
        }
        h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:32]

    def _embeddings(
        self,
        texts: Sequence[str],
        cache_dir: Optional[Path],
        cancel_event: Optional[threading.Event],
        progress: Optional[ProgressFn],
        label: str,
    ) -> np.ndarray:
        if self.dim is None:
            self.dim = int(self.encoder.config.hidden_size)
        if cache_dir is None:
            out = np.empty((len(texts), self.dim), dtype=np.float16)
            self._embed_into(texts, out, cancel_event, progress, label)
            return out
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(texts)
        path = cache_dir / f"{key}.npy"
        done = cache_dir / f"{key}.done"
        if path.is_file() and done.is_file():
            logger.info("embedding_probe: reusing cached %s embeddings %s", label, path)
            return np.load(path, mmap_mode="r")
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), self.dim))
        self._embed_into(texts, out, cancel_event, progress, label)
        out.flush()
        done.write_text(self.model_id or "", encoding="utf-8")
        return np.load(path, mmap_mode="r")

    # --- head ---

    def _build_head(self):
        torch = import_torch()
        nn = torch.nn
        if self.options["head"] == "mlp":
            hidden = int(self.options["mlp_hidden"])
            return nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.GELU(),
                nn.Dropout(float(self.options["mlp_dropout"])),
                nn.Linear(hidden, 1),
            )
        return nn.Sequential(nn.Linear(self.dim, 1))

    def _head_proba(self, emb: np.ndarray, cancel_event: Optional[threading.Event] = None) -> np.ndarray:
        torch = import_torch()
        out = np.empty(len(emb), dtype=np.float64)
        self.head.eval()
        with torch.inference_mode():
            for start in range(0, len(emb), 65_536):
                check_cancel_event(cancel_event)
                x = torch.as_tensor(np.asarray(emb[start : start + 65_536], dtype=np.float32), device=self.device)
                out[start : start + len(x)] = torch.sigmoid(self.head(x)[:, 0]).cpu().numpy()
        return out

    def fit(self, train: TextFitData, val: TextFitData, ctx: TextFitContext) -> Dict[str, Any]:
        torch = import_torch()
        self.options = dict(ctx.options)
        if self.options["pooling"] not in _POOLINGS:
            raise ValueError(_("backend_options.pooling must be one of: {c}").format(c=", ".join(_POOLINGS)))
        if self.options["head"] not in _HEADS:
            raise ValueError(_("backend_options.head must be one of: {c}").format(c=", ".join(_HEADS)))
        seed_everything(ctx.seed)
        self.model_id = ctx.model_id
        self.max_length = ctx.max_length
        self.device = ctx.device
        ctx.report(_("Loading {model}…").format(model=self.model_id), None)
        self._load_encoder()
        cache_dir = ctx.run_dir.parent / _CACHE_DIR if self.options["cache_embeddings"] else None

        def sub(lo: float, hi: float) -> ProgressFn:
            return lambda m, f: ctx.report(m, None if f is None else lo + (hi - lo) * f)

        x_train = self._embeddings(train.texts, cache_dir, ctx.cancel_event, sub(0.0, 0.6), _("train"))
        x_val = self._embeddings(val.texts, cache_dir, ctx.cancel_event, sub(0.6, 0.7), _("val"))

        self.head = self._build_head().to(self.device)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=float(ctx.lr), weight_decay=ctx.weight_decay)
        labels = torch.as_tensor(train.labels, dtype=torch.float32)
        weights = torch.as_tensor(train.weights, dtype=torch.float32)
        rng = np.random.default_rng(ctx.seed)
        stopper = EarlyStopper(patience=ctx.patience)
        best_state = None
        bs = int(ctx.batch_size)
        epochs = int(ctx.epochs)
        n = len(train.texts)
        for epoch in range(epochs):
            self.head.train()
            # Sorted indices within each batch keep memmap reads mostly sequential.
            order = rng.permutation(n)
            for start in range(0, n, bs):
                check_cancel_event(ctx.cancel_event)
                idx = np.sort(order[start : start + bs])
                x = torch.as_tensor(np.asarray(x_train[idx], dtype=np.float32), device=self.device)
                t_idx = torch.from_numpy(idx)
                loss = weighted_bce(
                    self.head(x)[:, 0], labels[t_idx].to(self.device), weights[t_idx].to(self.device)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            metric = ctx.selection_metric(self._head_proba(x_val, ctx.cancel_event))
            improved = stopper.update(metric, epoch + 1, float(epoch + 1))
            logger.info("embedding_probe: epoch %d val metric %.6f%s", epoch + 1, metric, " (best)" if improved else "")
            ctx.report(
                _("Head epoch {e}/{E} — val metric {m:.4f}").format(e=epoch + 1, E=epochs, m=metric),
                0.7 + 0.3 * (epoch + 1) / epochs,
            )
            if improved:
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.head.state_dict().items()}
            if stopper.should_stop:
                break
        if best_state is not None:
            self.head.load_state_dict(best_state)
        self.head.eval()
        return {"embedding_dim": self.dim, "model_revision": self.revision, **stopper.summary()}

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[ProgressFn] = None,
    ) -> np.ndarray:
        emb = self._embeddings(texts, None, cancel_event, progress, _("texts"))
        return self._head_proba(emb, cancel_event)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "max_length": self.max_length,
            "embedding_dim": self.dim,
            "options": self.options,
        }

    def save_weights(self, model_dir: Path) -> None:
        from safetensors.torch import save_file

        state = {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}
        save_file(state, str(model_dir / _HEAD_FILE))

    @classmethod
    def load_weights(cls, model_dir: Path, meta: Dict[str, Any], device: str) -> EmbeddingProbeBackend:
        from safetensors.torch import load_file

        b = cls()
        b.options = {**cls.option_defaults, **(meta.get("options") or {})}
        b.model_id = meta["model_id"]
        b.max_length = int(meta.get("max_length", 64))
        b.device = device
        b.dim = int(meta["embedding_dim"])
        b._load_encoder()
        saved_rev = meta.get("model_revision")
        if saved_rev and b.revision and saved_rev != b.revision:
            logger.warning(
                "embedding_probe: %s loaded at revision %s, but the head was trained on %s",
                b.model_id,
                b.revision,
                saved_rev,
            )
        b.head = b._build_head().to(device)
        b.head.load_state_dict(load_file(str(model_dir / _HEAD_FILE)))
        b.head.eval()
        return b


BACKEND = EmbeddingProbeBackend
