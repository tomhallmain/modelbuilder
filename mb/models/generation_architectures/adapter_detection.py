"""
Detects whether a path is a LoRA *adapter* artifact (``peft``'s ``save_pretrained()``
output — see :mod:`mb.training.lora_diffusion_trainer`), as opposed to a full model
checkpoint.

Used to give ``mb convert``/``mb export bundle`` a clean, LoRA-specific rejection instead
of a generic "unsupported source framework" message or (if the caller forces
``--framework pytorch``) a raw ``torch.load`` traceback. Neither of those commands has a
real conversion/export step to offer a LoRA adapter: it's already the final, portable
artifact produced by training (``adapter_model.safetensors`` + ``adapter_config.json``),
directly loadable by any ``diffusers``-based pipeline via ``pipe.load_lora_weights(...)``,
with no merge/reformat step needed — so the right behavior for both commands is a clear,
immediate rejection rather than an attempted (and doomed) conversion.
"""

from __future__ import annotations

from pathlib import Path

ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def looks_like_lora_adapter(path: Path) -> bool:
    """
    Best-effort check, not a guarantee: a directory containing ``adapter_config.json``
    (always written alongside adapter weights by ``peft.PeftModel.save_pretrained``), or a
    bare ``.safetensors`` file whose tensor names look like LoRA deltas (``lora_A``/``lora_B``,
    the standard ``peft`` key naming).
    """
    path = Path(path)
    if path.is_dir():
        return (path / ADAPTER_CONFIG_FILENAME).is_file()
    if path.is_file() and path.suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="np") as handle:
                return any("lora_a" in k.lower() or "lora_b" in k.lower() for k in handle.keys())
        except Exception:
            return False
    return False
