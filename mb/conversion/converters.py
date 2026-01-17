"""
Model conversion utilities.

This module provides functionality to convert models between different formats:
- PyTorch (.pth) to ONNX
- PyTorch (.pth) to SafeTensors
- Keras (.h5) to ONNX
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


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
        import torch
        import torch.onnx
        
        # Load PyTorch model
        from mb.models.frameworks.pytorch.trainer import PyTorchTrainer
        
        trainer = PyTorchTrainer()
        model = trainer.load_model(model_path, architecture, num_classes)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, image_size, image_size)
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        # Export to ONNX
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=kwargs.get('opset_version', 11),
            do_constant_folding=True,
            verbose=kwargs.get('verbose', False)
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
