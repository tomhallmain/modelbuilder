"""
Text-classification backends, one module per :class:`~mb.models.types.TextBackendType`.

Modules are imported on first use so choosing ``char_ngram_linear`` never imports torch or
transformers.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, Type

from mb.models.text_backends.base import (
    BACKEND_META_FILE,
    TextBackend,
    TextFitContext,
    TextFitData,
)
from mb.models.types import TextBackendType
from mb.utils.translations import _

_BACKEND_MODULES: Dict[TextBackendType, str] = {
    TextBackendType.CHAR_NGRAM_LINEAR: "mb.models.text_backends.char_ngram_linear",
    TextBackendType.EMBEDDING_PROBE: "mb.models.text_backends.embedding_probe",
    TextBackendType.ENCODER_FINETUNE: "mb.models.text_backends.encoder_finetune",
}


def get_backend_class(backend: TextBackendType) -> Type[TextBackend]:
    module = importlib.import_module(_BACKEND_MODULES[backend])
    return module.BACKEND


def load_backend(model_dir: Path, device: str) -> TextBackend:
    """Load a fitted backend from a run's ``model/`` directory."""
    model_dir = Path(model_dir)
    meta_path = model_dir / BACKEND_META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(_("No {file} in {path}").format(file=BACKEND_META_FILE, path=model_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backend = TextBackendType.try_from(meta.get("backend"))
    if backend is None:
        raise ValueError(_("Unknown backend in {path}: {b!r}").format(path=meta_path, b=meta.get("backend")))
    return get_backend_class(backend).load_weights(model_dir, meta, device)


__all__ = [
    "TextBackend",
    "TextFitContext",
    "TextFitData",
    "get_backend_class",
    "load_backend",
]
