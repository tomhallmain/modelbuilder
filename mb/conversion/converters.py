"""
Model conversion utilities.

This module provides functionality to convert models between different formats:
- PyTorch (.pth) to ONNX
- PyTorch (.pth) to SafeTensors
- Keras (.h5) to ONNX
"""

import gc
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mb.cancellation import check_cancel_event
from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Protobuf messages top out near 2 GiB; leave headroom before keeping external data.
_ONNX_EMBED_MAX_BYTES = 2 * 1024**3 - 64 * 1024**2


def inline_onnx_external_data(output_path: Path) -> bool:
    """
    Fold sibling external-data weight files into *output_path* when small enough.

    Recent ``torch.onnx.export`` defaults to ``external_data=True``, which writes
    ``model.onnx`` plus ``model.onnx.data``. For models under the protobuf size
    limit we prefer a single self-contained file.

    Returns:
        True if the ONNX file is self-contained after this call (or already was).
    """
    try:
        import onnx
        from onnx.external_data_helper import load_external_data_for_model
    except ImportError:
        logger.warning("onnx package not available; cannot inline external ONNX data")
        return False

    output_path = Path(output_path)
    try:
        # load_external_data=False keeps each tensor's ``external_data`` (sidecar file
        # location) metadata intact. Loading with load_external_data=True instead reads
        # the bytes AND clears that metadata in the same call, so external_files below
        # would always come up empty and we'd never find the sidecar file to remove.
        model = onnx.load(str(output_path), load_external_data=False)
    except Exception as e:
        logger.warning("Could not reload ONNX to inline external data: %s", e)
        return False

    external_files: set[Path] = set()
    for init in model.graph.initializer:
        for entry in init.external_data:
            if entry.key == "location":
                external_files.add((output_path.parent / entry.value).resolve())

    if not external_files:
        return True

    try:
        load_external_data_for_model(model, str(output_path.parent))
    except Exception as e:
        logger.warning("Could not load external ONNX data: %s", e)
        return False

    weight_bytes = sum(len(t.raw_data) for t in model.graph.initializer if t.raw_data)
    if weight_bytes > _ONNX_EMBED_MAX_BYTES:
        logger.warning(
            "ONNX weights (%s bytes) exceed the single-file protobuf limit; "
            "keeping external data files",
            f"{weight_bytes:,}",
        )
        return False

    try:
        onnx.save_model(model, str(output_path), save_as_external_data=False)
    except Exception as e:
        logger.warning("Failed to save self-contained ONNX: %s", e)
        return False

    # Drop our reference (and anything onnx/protobuf may still hold internally) before
    # deleting the sidecar file: on Windows an open handle blocks unlink, and a handle
    # from reading the external data can otherwise briefly outlive this point.
    del model
    gc.collect()

    for ext_path in external_files:
        if not ext_path.is_file() or ext_path.resolve() == output_path.resolve():
            continue
        _unlink_with_retry(ext_path)

    return True


def _unlink_with_retry(path: Path, *, attempts: int = 5, initial_delay: float = 0.1) -> None:
    """Delete *path*, retrying briefly on Windows file-lock races (AV scan, lingering handle)."""
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            path.unlink()
            logger.info("Removed external ONNX data file after inlining: %s", path)
            return
        except OSError as e:
            if attempt == attempts:
                logger.warning("Could not remove external ONNX data file %s: %s", path, e)
                return
            time.sleep(delay)
            delay *= 2


# New torch.onnx dynamo exporter implements opset 18+; requesting older versions
# triggers a failed downgrade (noisy traceback) and leaves the model at 18 anyway.
_DEFAULT_ONNX_OPSET = 18


def export_torch_module_to_onnx(
    model: Any,
    output_path: Path,
    *,
    image_size: int = 224,
    opset_version: int = _DEFAULT_ONNX_OPSET,
    verbose: bool = False,
) -> None:
    """
    Export a PyTorch module to a self-contained ONNX file when possible.

    Newer PyTorch builds default to writing weights as external data; we request
    an embedded export and then inline any sidecar ``*.onnx.data`` file.
    """
    import torch
    import torch.onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        "opset_version": opset_version,
        "do_constant_folding": True,
        "verbose": verbose,
    }

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            external_data=False,
            **export_kwargs,
        )
    except TypeError:
        # Older torch.onnx.export without external_data=
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)

    inline_onnx_external_data(output_path)


def convert_pytorch_to_onnx(
    model_path: Path,
    output_path: Path,
    architecture: str,
    num_classes: int,
    image_size: int = 224,
    **kwargs
) -> bool:
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        output_path: Path to save ONNX model
        architecture: Model architecture name (e.g., 'resnet34')
        num_classes: Number of output classes
        image_size: Input image size (assumes square)
        **kwargs: Additional conversion arguments
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Load PyTorch model
        from mb.models.frameworks.pytorch.trainer import PyTorchTrainer
        
        trainer = PyTorchTrainer()
        model = trainer.load_model(model_path, architecture, num_classes)
        model.eval()

        export_torch_module_to_onnx(
            model,
            output_path,
            image_size=image_size,
            opset_version=kwargs.get("opset_version", _DEFAULT_ONNX_OPSET),
            verbose=kwargs.get("verbose", False),
        )
        
        logger.info(f"Successfully converted PyTorch model to ONNX: {output_path}")
        return True
        
    except ImportError as e:
        logger.error(f"PyTorch not available for ONNX conversion: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert PyTorch model to ONNX: {e}", exc_info=True)
        return False


def convert_pytorch_to_safetensors(
    model_path: Path,
    output_path: Path,
    **kwargs
) -> bool:
    """
    Convert a PyTorch model to SafeTensors format.
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        output_path: Path to save SafeTensors model
        **kwargs: Additional conversion arguments
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        import torch
        
        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.error(
                "safetensors library not available. Install with: pip install safetensors"
            )
            return False
        
        # Load PyTorch state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # If it's a checkpoint dict, extract the model state dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Convert to SafeTensors format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(output_path))
        
        logger.info(f"Successfully converted PyTorch model to SafeTensors: {output_path}")
        return True
        
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert PyTorch model to SafeTensors: {e}", exc_info=True)
        return False


def convert_keras_to_onnx(
    model_path: Path,
    output_path: Path,
    **kwargs
) -> bool:
    """
    Convert a Keras model to ONNX format.
    
    Args:
        model_path: Path to Keras model (.h5 file)
        output_path: Path to save ONNX model
        **kwargs: Additional conversion arguments
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        try:
            import tf2onnx
        except ImportError:
            logger.error(
                "tf2onnx library not available. Install with: pip install tf2onnx"
            )
            return False
        
        # Load Keras model
        model = keras.models.load_model(str(model_path))
        
        # Get input shape
        input_shape = model.input_shape[1:]  # Remove batch dimension
        if len(input_shape) == 3:
            # (H, W, C) -> (1, H, W, C) for batch
            input_spec = tf.TensorSpec(
                shape=(None,) + input_shape,
                dtype=tf.float32,
                name='input'
            )
        else:
            logger.error(f"Unsupported input shape: {input_shape}")
            return False
        
        # Convert to ONNX
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        spec = (input_spec,)
        output_path_str = str(output_path)
        
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=kwargs.get('opset_version', 11),
            output_path=output_path_str
        )
        
        logger.info(f"Successfully converted Keras model to ONNX: {output_path}")
        return True
        
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert Keras model to ONNX: {e}", exc_info=True)
        return False


def detect_model_framework(model_path: Path) -> Optional[str]:
    """
    Detect the framework of a model file based on extension and content.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Framework name ('pytorch' or 'keras') or None if unknown
    """
    suffix = model_path.suffix.lower()
    
    if suffix == '.pth' or suffix == '.pt':
        return 'pytorch'
    elif suffix == '.h5' or suffix == '.keras':
        return 'keras'
    elif suffix == '.onnx':
        return 'onnx'
    elif suffix == '.safetensors':
        return 'safetensors'
    
    # Try to detect by loading
    try:
        import torch
        try:
            torch.load(model_path, map_location='cpu')
            return 'pytorch'
        except:
            pass
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        try:
            keras.models.load_model(str(model_path))
            return 'keras'
        except:
            pass
    except ImportError:
        pass
    
    return None


def convert_model(
    input_path: Path,
    output_path: Path,
    source_framework: Optional[str] = None,
    target_format: str = "onnx",
    architecture: Optional[str] = None,
    num_classes: Optional[int] = None,
    image_size: int = 224,
    cancel_event: Optional[threading.Event] = None,
    **kwargs
) -> bool:
    """
    Convert a model between formats.
    
    Args:
        input_path: Path to input model file
        output_path: Path to save converted model
        source_framework: Source framework ('pytorch' or 'keras'), auto-detected if None
        target_format: Target format ('onnx' or 'safetensors')
        architecture: Model architecture (required for PyTorch -> ONNX)
        num_classes: Number of classes (required for PyTorch -> ONNX)
        image_size: Input image size (default: 224)
        **kwargs: Additional conversion arguments
        
    Returns:
        True if conversion successful, False otherwise
    """
    if not input_path.exists():
        logger.error(f"Input model file not found: {input_path}")
        return False
    
    check_cancel_event(cancel_event)
    
    # Detect source framework if not provided
    if source_framework is None:
        source_framework = detect_model_framework(input_path)
        if source_framework is None:
            logger.error(f"Could not detect framework for {input_path}")
            return False
        logger.info(f"Detected source framework: {source_framework}")
    
    # Validate target format
    if target_format not in ['onnx', 'safetensors']:
        logger.error(f"Unsupported target format: {target_format}")
        logger.info("Supported formats: onnx, safetensors")
        return False
    
    # PyTorch conversions
    if source_framework == 'pytorch':
        if target_format == 'onnx':
            if architecture is None or num_classes is None:
                logger.error(
                    "Architecture and num_classes are required for PyTorch -> ONNX conversion"
                )
                return False
            return convert_pytorch_to_onnx(
                input_path, output_path, architecture, num_classes, image_size, **kwargs
            )
        elif target_format == 'safetensors':
            return convert_pytorch_to_safetensors(input_path, output_path, **kwargs)
    
    # Keras conversions
    elif source_framework == 'keras':
        if target_format == 'onnx':
            return convert_keras_to_onnx(input_path, output_path, **kwargs)
        elif target_format == 'safetensors':
            logger.error("Keras -> SafeTensors conversion not supported")
            logger.info("SafeTensors is primarily for PyTorch models")
            return False
    
    else:
        logger.error(f"Unsupported source framework: {source_framework}")
        return False
    
    return False
