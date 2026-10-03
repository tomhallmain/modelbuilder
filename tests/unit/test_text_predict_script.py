"""Generated ``predict.py``: every backend yields a complete, compilable script."""

from __future__ import annotations

import re

import pytest

from mb.evaluate.classification.text_predict_script import predict_script_source
from mb.models.text_backends import get_backend_class
from mb.models.types import TextBackendType

# Template placeholders are upper-case dunder names; Python's own (__file__, …) are lower-case.
_PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")


@pytest.mark.parametrize("backend", list(TextBackendType), ids=lambda b: b.value)
def test_predict_script_compiles_with_no_placeholders(backend: TextBackendType) -> None:
    src = predict_script_source(get_backend_class(backend))
    compile(src, "predict.py", "exec")
    assert _PLACEHOLDER.findall(src) == []
    assert f"BACKEND = {backend.value!r}" in src
    assert "def load_model(" in src


@pytest.mark.parametrize("backend", list(TextBackendType), ids=lambda b: b.value)
def test_predict_script_imports_no_mb(backend: TextBackendType) -> None:
    src = predict_script_source(get_backend_class(backend))
    assert not re.search(r"^\s*(from|import)\s+mb(\.|\s|$)", src, re.MULTILINE)
