"""
Post-export ONNX quantization.

Quantized variants trade accuracy for download size and latency, which is the trade a
browser or edge deployment usually wants to make. They are produced from an existing ONNX
file rather than during export, so the full-precision graph stays the reference artifact
and every variant is reproducible from it.

Each variant is written as a sibling file (``model.onnx`` -> ``model.fp16.onnx``) so a
caller can ship several and let the consumer pick.

Accuracy is not checked here. A quantized classifier can shift decision boundaries enough
to matter for a rare class, so a variant should be scored with ``mb evaluate metrics``
before it is deployed rather than assumed equivalent to its source.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from mb.models.types import OnnxQuantizationMode
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)


def quantized_output_path(source: Path, mode: OnnxQuantizationMode) -> Path:
    """Sibling path for *mode* (``model.onnx`` -> ``model.<mode>.onnx``)."""
    source = Path(source)
    return source.with_suffix(f".{mode.value}{source.suffix}")


def _quantize_fp16(source: Path, output: Path) -> bool:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        logger.error(
            _(
                "fp16 quantization needs onnx and onnxconverter-common. "
                "Install with: pip install onnxconverter-common"
            )
        )
        logger.debug("fp16 import failure: %s", e)
        return False

    model = onnx.load(str(source))
    # keep_io_types leaves the graph's inputs and outputs float32, so callers feed and read
    # the same tensors as the full-precision model and only the internals are halved.
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(converted, str(output))
    return True


def _quantize_dynamic(source: Path, output: Path, mode: OnnxQuantizationMode) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        logger.error(
            _(
                "int8/uint8 quantization needs onnxruntime. "
                "Install with: pip install -e \".[onnx]\""
            )
        )
        logger.debug("onnxruntime.quantization import failure: %s", e)
        return False

    weight_type = QuantType.QInt8 if mode == OnnxQuantizationMode.INT8 else QuantType.QUInt8
    quantize_dynamic(
        model_input=str(source),
        model_output=str(output),
        weight_type=weight_type,
    )
    return True


def quantize_onnx_model(
    source: Path,
    mode: Union[OnnxQuantizationMode, str],
    output: Optional[Path] = None,
) -> Optional[Path]:
    """
    Write a quantized copy of *source*.

    Args:
        source: Full-precision ``.onnx`` file.
        mode: Quantization variant to produce.
        output: Destination; defaults to :func:`quantized_output_path`.

    Returns:
        The path written, or None if the variant could not be produced.
    """
    resolved_mode = OnnxQuantizationMode.try_from(mode)
    if resolved_mode is None:
        logger.error(_("Unknown quantization mode: {mode}").format(mode=mode))
        return None

    source = Path(source)
    if not source.is_file():
        logger.error(_("ONNX file not found: {path}").format(path=source))
        return None

    destination = Path(output) if output is not None else quantized_output_path(source, resolved_mode)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        if resolved_mode == OnnxQuantizationMode.FP16:
            ok = _quantize_fp16(source, destination)
        else:
            ok = _quantize_dynamic(source, destination, resolved_mode)
    except Exception as e:
        logger.error(
            _("Quantization to {mode} failed: {err}").format(mode=resolved_mode.value, err=e),
            exc_info=True,
        )
        return None

    if not ok or not destination.is_file():
        return None

    # Metadata is not always carried across by the quantization tools, so re-apply the
    # source graph's decode contract: a variant that loses its label order is unusable.
    _copy_metadata(source, destination)

    size_mb = destination.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s variant: %s (%.1f MB)", resolved_mode.value, destination, size_mb)
    return destination


def _copy_metadata(source: Path, destination: Path) -> None:
    """Carry ``metadata_props`` from *source* onto *destination*, if onnx is available."""
    try:
        import onnx
    except ImportError:
        return
    try:
        src_model = onnx.load(str(source), load_external_data=False)
        metadata = {prop.key: prop.value for prop in src_model.metadata_props}
    except Exception as e:
        logger.debug("Could not read source ONNX metadata: %s", e)
        return
    if not metadata:
        return

    from mb.conversion.converters import write_onnx_metadata

    write_onnx_metadata(destination, metadata)


def quantize_onnx_variants(
    source: Path,
    modes: Sequence[Union[OnnxQuantizationMode, str]],
) -> List[Path]:
    """
    Produce several variants of *source*, skipping any that fail.

    A failure is logged and skipped rather than aborting the rest: the variants are
    independent, and losing one is not a reason to lose the others.
    """
    written: List[Path] = []
    for mode in modes:
        path = quantize_onnx_model(source, mode)
        if path is not None:
            written.append(path)
    return written
