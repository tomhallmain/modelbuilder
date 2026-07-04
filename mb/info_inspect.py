"""
Inspect trained model files and ImageFolder-style dataset directories.

Used by ``mb info model`` / ``mb info dataset`` and the desktop Info page so
behavior stays aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from mb.conversion.converters import detect_model_framework
from mb.data.file_types import configured_media_suffixes
from mb.models.frameworks.registry import list_architectures
from mb.utils.translations import _


def _count_class_images(class_dir: Path) -> int:
    n = 0
    for ext in configured_media_suffixes():
        n += len(list(class_dir.glob(f"*{ext}")))
    return n


def _tensor_proto_nbytes(tensor) -> int:
    """Byte size of an ONNX TensorProto payload (embedded data only)."""
    if tensor.raw_data:
        return len(tensor.raw_data)
    try:
        from onnx import numpy_helper

        return int(numpy_helper.to_array(tensor).nbytes)
    except Exception:
        return 0


def _onnx_detail_lines(path: Path) -> List[str]:
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        return [_("ONNX details unavailable (install onnx: pip install onnx).")]

    try:
        # Do not pull external weight files into memory; report them separately.
        model = onnx.load(str(path), load_external_data=False)
    except Exception as e:
        return [_("Failed to load ONNX: {err}").format(err=e)]

    initializers = list(model.graph.initializer)
    embedded_bytes = 0
    external_count = 0
    external_locations: dict[str, int] = {}

    for init in initializers:
        is_external = (
            getattr(init, "data_location", TensorProto.DEFAULT) == TensorProto.EXTERNAL
            or bool(init.external_data)
        )
        if is_external:
            external_count += 1
            location = None
            length = 0
            for entry in init.external_data:
                if entry.key == "location":
                    location = entry.value
                elif entry.key == "length":
                    try:
                        length = int(entry.value)
                    except ValueError:
                        length = 0
            if location:
                external_locations[location] = external_locations.get(location, 0) + length
        else:
            embedded_bytes += _tensor_proto_nbytes(init)

    lines = [
        _("ONNX initializers (weight tensors): {n}").format(n=len(initializers)),
        _("Embedded weight bytes: {n:,}").format(n=embedded_bytes),
    ]

    if external_count:
        lines.append(_("External-data initializers: {n}").format(n=external_count))
        for location, declared in sorted(external_locations.items()):
            ext_path = (path.parent / location).resolve()
            if ext_path.is_file():
                lines.append(
                    _("  External file: {path} ({n:,} bytes)").format(
                        path=ext_path,
                        n=ext_path.stat().st_size,
                    )
                )
            else:
                missing = _("missing")
                declared_s = (
                    _("{n:,} bytes declared").format(n=declared) if declared else _("size unknown")
                )
                lines.append(
                    _("  External file {status}: {path} ({detail})").format(
                        status=missing,
                        path=path.parent / location,
                        detail=declared_s,
                    )
                )

    inputs = [i.name for i in model.graph.input if i.name not in {t.name for t in initializers}]
    outputs = [o.name for o in model.graph.output]
    if inputs:
        lines.append(_("Inputs: {names}").format(names=", ".join(inputs)))
    if outputs:
        lines.append(_("Outputs: {names}").format(names=", ".join(outputs)))

    if not initializers or (embedded_bytes == 0 and external_count == 0):
        lines.append(
            _(
                "Warning: no weight tensors found in this ONNX file "
                "(graph-only or incomplete export)."
            )
        )
    elif embedded_bytes == 0 and external_count > 0:
        missing_any = any(
            not (path.parent / loc).is_file() for loc in external_locations
        )
        if missing_any:
            lines.append(
                _(
                    "Warning: weights are stored in external data files that are missing "
                    "next to the ONNX file."
                )
            )

    return lines


def _pytorch_detail_lines(path: Path) -> List[str]:
    try:
        import torch
    except ImportError:
        return [_("PyTorch details unavailable (install torch).")]

    try:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location="cpu")
    except Exception as e:
        return [_("Failed to load PyTorch file: {err}").format(err=e)]

    from mb.evaluate._weights import extract_pytorch_state_dict

    try:
        state_dict = extract_pytorch_state_dict(raw)
    except ValueError as e:
        return [str(e)]

    n_tensors = len(state_dict)
    n_params = 0
    weight_bytes = 0
    for tensor in state_dict.values():
        n_params += int(tensor.numel())
        weight_bytes += int(tensor.numel() * tensor.element_size())

    lines = [
        _("State-dict tensors: {n}").format(n=n_tensors),
        _("Parameters: {n:,}").format(n=n_params),
        _("Weight bytes: {n:,}").format(n=weight_bytes),
    ]
    if isinstance(raw, dict) and "model_state_dict" in raw:
        lines.append(_("Checkpoint layout: model_state_dict"))
        extra = [k for k in raw if k != "model_state_dict"]
        if extra:
            lines.append(_("Other checkpoint keys: {keys}").format(keys=", ".join(sorted(extra))))
    return lines


def _safetensors_detail_lines(path: Path) -> List[str]:
    try:
        from safetensors import safe_open
    except ImportError:
        return [_("SafeTensors details unavailable (install safetensors).")]

    try:
        n_tensors = 0
        n_params = 0
        weight_bytes = 0
        with safe_open(str(path), framework="np") as handle:
            for key in handle.keys():
                arr = handle.get_tensor(key)
                n_tensors += 1
                n_params += int(arr.size)
                weight_bytes += int(arr.nbytes)
    except Exception as e:
        return [_("Failed to load SafeTensors file: {err}").format(err=e)]

    return [
        _("Tensors: {n}").format(n=n_tensors),
        _("Parameters: {n:,}").format(n=n_params),
        _("Weight bytes: {n:,}").format(n=weight_bytes),
    ]


def _format_detail_lines(path: Path, framework: str | None) -> List[str]:
    if framework == "onnx":
        return _onnx_detail_lines(path)
    if framework == "pytorch":
        return _pytorch_detail_lines(path)
    if framework == "safetensors":
        return _safetensors_detail_lines(path)
    return []


def model_info_text(model_path: Path) -> str:
    """
    Build a human-readable report for *model_path*.

    Ensures framework registration side effects have run so architecture lists are complete.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a regular file.
    """
    resolved = model_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    if not resolved.is_file():
        raise ValueError(_("Path is not a file: {path}").format(path=resolved))

    framework = detect_model_framework(resolved)
    lines = [
        _("Path: {path}").format(path=resolved),
        _("Size: {n:,} bytes").format(n=resolved.stat().st_size),
        _("Detected framework/type: {fw}").format(fw=framework or _("unknown")),
        "",
    ]

    detail = _format_detail_lines(resolved, framework)
    if detail:
        lines.extend(detail)
        lines.append("")

    try:
        from mb.models.frameworks import pytorch  # noqa: F401
    except Exception:
        pass
    try:
        from mb.models.frameworks import keras  # noqa: F401
    except Exception:
        pass
    archs = list_architectures()
    lines.append(_("Registered architectures:"))
    for fw, items in archs.items():
        lines.append(
            "- {fw}: {names}".format(
                fw=fw,
                names=", ".join(items) if items else _("(none)"),
            )
        )
    return "\n".join(lines)


def dataset_info_text(data_dir: Path) -> str:
    """
    Summarize ``train`` / ``test`` splits under *data_dir* (per-class subfolders, image counts).

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError: If *data_dir* is not a directory.
    """
    resolved = data_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    if not resolved.is_dir():
        raise ValueError(_("Path is not a directory: {path}").format(path=resolved))

    lines = [_("Data dir: {path}").format(path=resolved), ""]
    for split in ("train", "test"):
        split_dir = resolved / split
        lines.append(f"[{split}]")
        if not split_dir.exists():
            lines.append(_("  missing"))
            lines.append("")
            continue
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        total = 0
        for cls in class_dirs:
            count = _count_class_images(cls)
            total += count
            lines.append(f"  {cls.name}: {count}")
        lines.append(_("  total: {n}").format(n=total))
        lines.append("")
    return "\n".join(lines)
