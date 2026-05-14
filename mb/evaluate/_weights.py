"""PyTorch checkpoint / state-dict loading helpers for evaluation."""

from __future__ import annotations

from typing import Any, MutableMapping

from mb.utils.translations import _

RawStateDict = MutableMapping[str, Any]


def _looks_like_pytorch_state_dict(d: dict) -> bool:
    """Heuristic: flat mapping of torch.Tensor values (module parameters)."""
    if not d:
        return False
    v = next(iter(d.values()))
    t = type(v)
    return getattr(t, "__module__", "") == "torch" and getattr(t, "__name__", "") == "Tensor"


def extract_pytorch_state_dict(raw: object) -> RawStateDict:
    """
    Normalize ``torch.load`` output to a flat state dict suitable for ``load_state_dict``.

    Supports plain ``state_dict`` tensors and training checkpoints that wrap weights under
    ``model_state_dict``.
    """
    if not isinstance(raw, dict):
        raise ValueError(_("PyTorch file did not contain a dictionary checkpoint or state_dict."))

    inner = raw.get("model_state_dict")
    if isinstance(inner, dict) and _looks_like_pytorch_state_dict(inner):
        return inner

    if _looks_like_pytorch_state_dict(raw):
        return raw  # type: ignore[return-value]

    raise ValueError(
        _("Unrecognized PyTorch checkpoint layout (expected state_dict keys or model_state_dict).")
    )
