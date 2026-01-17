# Model Builder Application - Implementation Plan

## Executive Summary

This document outlines the plan to transform the existing discrete script-based pipeline into a unified, extensible model building application. The new application will support multiple deep learning frameworks (PyTorch, Keras/TensorFlow) and provide a single entry point for all operations.

## Current State Analysis

### Existing Components

1. **Data Pipeline Scripts:**
   - `gather_coherent_images.py` - Collects and organizes source images
   - `deduplicate_images.py` - Removes duplicate images
   - `upscale_small_images.py` - Upscales small images
   - `convert_to_jpeg.py` - Converts images to JPEG format
   - `create_datasets.py` - Creates train/test splits

2. **Training Scripts:**
   - `train_model.py` - Trains models using FastAI v2 (needs replacement)
   - `train_hyperparams.py` - Hyperparameter configuration

3. **Model Conversion Scripts:**
   - `convert_to_h5.py` - Converts models to HDF5 format
   - `convert_to_pytorch.py` - PyTorch conversion utilities
   - `convert_to_safetensors.py` - SafeTensors conversion

4. **Utility Modules:**
   - `logging_config.py` - Centralized logging
   - `snapshot_utils.py` - Image tracking through pipeline
   - `storage_utils.py` - Storage validation utilities
   - `timing_utils.py` - Performance tracking

### Current Limitations

1. **No unified entry point** - Each script must be run separately
2. **FastAI dependency** - Training relies on FastAI v2 (user wants to avoid)
3. **Framework lock-in** - Hard to switch between PyTorch/Keras
4. **No abstraction** - Model architectures are hardcoded
5. **Scattered configuration** - Hyperparameters and settings spread across files
6. **No model type abstraction** - Currently only supports image classification implicitly

## Requirements

### Functional Requirements

1. **Single Entry Point**
   - Unified CLI application (`modelbuilder` or `mb`)
   - Subcommands for different operations (data, train, convert, etc.)

2. **Model Type Support**
   - Image classification (initial implementation)
   - Extensible architecture for future model types (object detection, segmentation, etc.)

3. **Framework Support**
   - PyTorch (native)
   - Keras/TensorFlow (native)
   - FastAI support removed initially (note: add back as optional framework in future)

4. **Model Architecture Abstraction**
   - Support multiple architectures per framework
   - Easy to add new architectures
   - Framework-agnostic configuration

5. **Command Line Interface**
   - Intuitive subcommands
   - Consistent argument patterns
   - Helpful error messages

### Non-Functional Requirements

1. **Extensibility**
   - Easy to add new model types
   - Easy to add new architectures
   - Easy to add new frameworks

2. **Maintainability**
   - Clean separation of concerns
   - Well-documented code
   - Type hints where appropriate

## Proposed Architecture

### High-Level Structure

```
modelbuilder/
├── mb/                          # Main application package
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point and argument parsing
│   ├── config.py                # Configuration management
│   │
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── gather.py           # Image gathering (from gather_coherent_images.py)
│   │   ├── convert.py          # Format conversion (from convert_to_jpeg.py)
│   │   ├── deduplicate.py      # Deduplication (from deduplicate_images.py)
│   │   ├── upscale.py          # Image upscaling (from upscale_small_images.py)
│   │   └── dataset.py          # Dataset creation (from create_datasets.py)
│   │
│   ├── models/                  # Model-related modules
│   │   ├── __init__.py
│   │   ├── base.py             # Base classes for models
│   │   ├── types.py            # Model type definitions (classification, etc.)
│   │   │
│   │   ├── frameworks/         # Framework-specific implementations
│   │   │   ├── __init__.py
│   │   │   ├── pytorch/        # PyTorch implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py  # PyTorch training logic
│   │   │   │   ├── architectures.py  # ResNet, etc.
│   │   │   │   └── data_loader.py    # PyTorch DataLoader
│   │   │   │
│   │   │   ├── keras/          # Keras/TensorFlow implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py  # Keras training logic
│   │   │   │   ├── architectures.py  # ResNet, etc.
│   │   │   │   └── data_loader.py    # Keras data generators
│   │   │   │
│   │   │   └── fastai/         # FastAI implementations (future)
│   │   │       └── [placeholder for future FastAI support]
│   │   │
│   │   └── classification/     # Image classification specific
│   │       ├── __init__.py
│   │       ├── pytorch_classifier.py
│   │       └── keras_classifier.py
│   │
│   ├── training/               # Training orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py         # Generic training interface
│   │   └── hyperparams.py     # Hyperparameter management
│   │
│   ├── conversion/             # Model conversion utilities
│   │   ├── __init__.py
│   │   ├── converters.py     # Format conversion (from convert_to_h5.py, etc.)
│   │   └── formats.py        # Supported formats
│   │
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── logging.py         # From logging_config.py
│       ├── snapshot.py        # From snapshot_utils.py
│       ├── storage.py         # From storage_utils.py
│       └── timing.py          # From timing_utils.py
│
├── scripts/                    # Legacy scripts (kept for compatibility)
│   └── [existing batch files]
│
├── configs/                    # Configuration files
│   └── default.yaml           # Default configuration
│
├── tests/                      # Unit tests
│   └── ...
│
├── requirements.txt            # Updated dependencies
├── setup.py                   # Package installation
├── README.md                  # Updated documentation
└── IMPLEMENTATION_PLAN.md     # This file
```

### Core Design Patterns

#### 1. Strategy Pattern for Frameworks

```python
# Abstract base class for framework trainers
class FrameworkTrainer(ABC):
    @abstractmethod
    def create_model(self, architecture: str, num_classes: int, **kwargs):
        pass
    
    @abstractmethod
    def train(self, model, train_loader, val_loader, hyperparams):
        pass

# PyTorch implementation
class PyTorchTrainer(FrameworkTrainer):
    ...

# Keras implementation
class KerasTrainer(FrameworkTrainer):
    ...
```

#### 2. Factory Pattern for Model Types

```python
class ModelTypeFactory:
    @staticmethod
    def create(model_type: str, framework: str):
        if model_type == "image_classification":
            if framework == "pytorch":
                return PyTorchImageClassifier()
            elif framework == "keras":
                return KerasImageClassifier()
        ...
```

#### 3. Registry Pattern for Architectures

```python
# Architecture registry
ARCHITECTURE_REGISTRY = {
    "pytorch": {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        ...
    },
    "keras": {
        "resnet50": tf.keras.applications.ResNet50,
        "efficientnet": tf.keras.applications.EfficientNetB0,
        ...
    }
}
```

## Implementation Plan

### Phase 1: Foundation ✅

#### 1.1 Project Structure Setup ✅
- [x] Create new `mb/` package directory
- [x] Set up package structure with `__init__.py` files
- [x] Create `setup.py` for package installation
- [x] Update `requirements.txt` (remove FastAI, add PyTorch/Keras)

#### 1.2 Core Abstractions ✅
- [x] Implement `ModelType` base class and enum
- [x] Implement `FrameworkTrainer` abstract base class
- [x] Create architecture registry system
- [x] Implement configuration management (`config.py`)

#### 1.3 CLI Foundation ✅
- [x] Set up CLI entry point using `argparse` or `click`
- [x] Define subcommand structure:
  - `mb data gather` - Gather images
  - `mb data convert` - Convert formats
  - `mb data deduplicate` - Remove duplicates
  - `mb data create-dataset` - Create train/test splits
  - `mb train` - Train model
  - `mb convert` - Convert model formats
  - `mb info` - Show information about models/data

### Phase 2: Data Pipeline Migration ✅

#### 2.1 Migrate Data Processing Scripts ✅
- [x] Refactor `gather_coherent_images.py` → `mb/data/gather.py`
- [x] Refactor `convert_to_jpeg.py` → `mb/data/convert.py`
- [x] Refactor `deduplicate_images.py` → `mb/data/deduplicate.py`
- [x] Refactor `upscale_small_images.py` → `mb/data/upscale.py`
- [x] Refactor `create_datasets.py` → `mb/data/dataset.py`

#### 2.2 Utility Migration ✅
- [x] Migrate `logging_config.py` → `mb/utils/logging.py`
- [x] Migrate `snapshot_utils.py` → `mb/utils/snapshot.py`
- [x] Migrate `storage_utils.py` → `mb/utils/storage.py`
- [x] Migrate `timing_utils.py` → `mb/utils/timing.py`

#### 2.3 CLI Integration ✅
- [x] Implement `mb data` subcommands
- [x] Add argument parsing for each data operation
- [x] Test data pipeline functionality

### Phase 3: Framework Implementations ✅

#### 3.1 PyTorch Implementation ✅
- [x] Create `mb/models/frameworks/pytorch/trainer.py`
- [x] Implement PyTorch data loading (`data_loader.py`)
- [x] Implement architecture support (`architectures.py`)
- [x] Implement training loop with:
  - Frozen/unfrozen phases
  - Learning rate scheduling
  - Checkpointing
  - Evaluation metrics

#### 3.2 Keras Implementation ✅
- [x] Create `mb/models/frameworks/keras/trainer.py`
- [x] Implement Keras data generators (`data_loader.py`)
- [x] Implement architecture support (`architectures.py`)
- [x] Implement training with:
  - Transfer learning support
  - Callbacks (checkpointing, early stopping)
  - Evaluation metrics

#### 3.3 Training Orchestration ✅
- [x] Create `mb/training/trainer.py` - Generic training interface
- [x] Implement framework selection logic
- [x] Create hyperparameter management (`hyperparams.py`)
- [x] Add support for training resumption
- [x] Integrate training command into CLI

### Phase 4: Training CLI and Integration ✅

#### 4.1 Training Command ✅
- [x] Implement `mb train` command
- [x] Add arguments for:
  - Model type (classification)
  - Framework (pytorch/keras)
  - Architecture (resnet34, etc.)
  - Hyperparameters
  - Data paths
- [x] Integrate with existing data pipeline
- [x] Integrate unified snapshot tracking

#### 4.2 Model Conversion ✅
- [x] Create `mb/conversion/converters.py` with conversion utilities
- [x] Support ONNX conversion (PyTorch -> ONNX, Keras -> ONNX)
- [x] Support SafeTensors conversion (PyTorch -> SafeTensors)
- [x] Implement `mb convert` command

### Phase 5: Testing and Documentation

#### 5.1 Testing
- [ ] Unit tests for core abstractions
- [ ] Integration tests for data pipeline
- [ ] Framework-specific training tests
- [ ] CLI command tests

#### 5.2 Documentation
- [ ] Update README.md with new usage
- [ ] Create user guide
- [ ] Document architecture decisions
- [ ] Add code comments and docstrings

### Phase 6: Migration and Cleanup

#### 6.1 Final Integration
- [ ] Test full pipeline end-to-end
- [ ] Verify all CLI commands work correctly
- [ ] Ensure data formats are compatible

#### 6.2 Cleanup
- [ ] Remove FastAI dependencies
- [ ] Update requirements.txt
- [ ] Archive or remove obsolete code
- [ ] Final testing

## Detailed Component Specifications

### CLI Command Structure

```bash
# Data operations
mb data gather --source-dir PATH --subdirs DIR1 DIR2 --target-count N
mb data convert --raw-data-dir PATH --format jpeg
mb data deduplicate --raw-data-dir PATH
mb data upscale --raw-data-dir PATH --min-size SIZE
mb data create-dataset --raw-data-dir PATH --data-dir PATH --test-per-class N

# Training
mb train \
    --model-type image_classification \
    --framework pytorch \
    --architecture resnet34 \
    --data-dir PATH \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output-dir PATH

# Model conversion
mb convert --input MODEL.pth --output MODEL.h5 --framework pytorch --target keras

# Information
mb info model --path MODEL.pth
mb info dataset --data-dir PATH
```

### Configuration System

#### YAML Configuration File (`configs/default.yaml`)

```yaml
# Default configuration
model:
  default_type: image_classification
  default_framework: pytorch
  default_architecture: resnet34

data:
  raw_data_dir: raw_data
  data_dir: data
  test_images_per_class: 1000
  image_size: 224
  batch_size: null  # auto-detect

training:
  frozen_epochs: 5
  unfrozen_epochs: 20
  frozen_lr: 0.001
  unfrozen_lr_max: 0.0003
  unfrozen_lr_min: 0.00001
  num_workers: 12

paths:
  models_dir: data/models
  logs_dir: logs
  timing_dir: timing_data
```

### Framework Trainer Interface

```python
class FrameworkTrainer(ABC):
    """Abstract base class for framework-specific trainers."""
    
    @abstractmethod
    def create_model(
        self, 
        architecture: str, 
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> Any:
        """Create a model instance."""
        pass
    
    @abstractmethod
    def create_data_loaders(
        self,
        train_dir: Path,
        val_dir: Path,
        batch_size: int,
        image_size: int,
        num_workers: int = 0,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Create training and validation data loaders."""
        pass
    
    @abstractmethod
    def train(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        hyperparams: Dict[str, Any],
        output_dir: Path,
        **kwargs
    ) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        val_loader: Any,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate the model."""
        pass
    
    @abstractmethod
    def save_model(
        self,
        model: Any,
        path: Path,
        format: str = "native",
        **kwargs
    ) -> Path:
        """Save the model."""
        pass
    
    @abstractmethod
    def load_model(
        self,
        path: Path,
        **kwargs
    ) -> Any:
        """Load a saved model."""
        pass
```

### Model Type Interface

```python
class ModelType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    # Future: OBJECT_DETECTION, SEGMENTATION, etc.

class ModelTypeHandler(ABC):
    """Handler for specific model types."""
    
    @abstractmethod
    def get_num_classes(self, data_dir: Path) -> int:
        """Determine number of classes from data directory."""
        pass
    
    @abstractmethod
    def validate_data(self, data_dir: Path) -> bool:
        """Validate data structure for this model type."""
        pass
    
    @abstractmethod
    def get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters for this model type."""
        pass
```

## Migration Strategy

### Gradual Migration Strategy

1. **Keep existing scripts** - Existing scripts are copies of originals in a different directory, so no risk to breaking workflows
2. **Create new CLI** - Build alongside existing scripts in new `mb/` package
3. **Migrate functionality** - Port features from existing scripts to new structure
4. **Test thoroughly** - Ensure new CLI works correctly
5. **Note:** FastAI support will be added back as an optional framework in the future

## Dependencies

### Remove
- `fastai>=2.0.0` (replaced with native PyTorch/Keras)

### Add
- `torch>=2.0.0` (PyTorch)
- `torchvision>=0.15.0` (PyTorch vision models)
- `tensorflow>=2.10.0` or `keras>=2.10.0` (Keras/TensorFlow)
- `click>=8.0.0` (optional, for better CLI)
- `pyyaml>=6.0` (for configuration files)

### Keep
- `Pillow>=9.0.0` (image processing)
- `numpy>=1.25.2,<2.0` (numerical operations)
- All utility dependencies

## Testing Strategy

### Unit Tests
- Test each module independently
- Mock framework dependencies where needed
- Test configuration loading
- Test CLI argument parsing

### Integration Tests
- Test full data pipeline
- Test training with small datasets
- Test model conversion
- Test CLI commands end-to-end

### Compatibility Tests
- Test data format compatibility
- Test model loading from various formats

## Risk Mitigation

### Risk 1: Framework-Specific Bugs
**Mitigation:** Comprehensive testing, clear error messages, fallback options where possible

### Risk 2: Missing Features
**Mitigation:** Feature parity checklist (see below), user feedback (collected over time), iterative improvements

### Risk 3: Application Completeness
**Mitigation:** Focus on getting the application working first, optimization can follow. Ensure core functionality exists before fine-tuning.

## Feature Parity Checklist

This checklist tracks features from the original FastAI-based implementation that need to be replicated in the new framework-agnostic system.

### Data Pipeline Features
- [x] Image gathering with deduplication
- [x] Image format conversion (to JPEG)
- [x] Image deduplication across directories
- [x] Image upscaling for small images
- [x] Dataset creation with train/test splits
- [x] Corrupted image detection and removal
- [x] Invalid-sized image filtering
- [x] Hash-based filename generation
- [x] Unified snapshot tracking

### Training Features
- [ ] Two-phase training (frozen/unfrozen backbone)
- [ ] Learning rate scheduling
- [ ] Checkpoint saving and resumption
- [ ] Model evaluation with metrics
- [ ] Test Time Augmentation (TTA) evaluation
- [ ] Confusion matrix generation
- [ ] Multiple architecture support (ResNet variants)
- [ ] Transfer learning support
- [ ] Image augmentation
- [ ] Progress logging and timing

### Model Management Features
- [ ] Model saving in multiple formats (.pth, .h5, etc.)
- [ ] Model conversion between formats
- [ ] Model loading and inference
- [ ] Model metadata tracking

### Framework-Specific Parity

#### PyTorch
- [ ] ResNet architectures (18, 34, 50)
- [ ] DataLoader with proper transforms
- [ ] Training loop with optimizer and scheduler
- [ ] Model checkpointing

#### Keras/TensorFlow
- [ ] ResNet architectures
- [ ] Data generators with augmentation
- [ ] Training with callbacks
- [ ] Model checkpointing

### Utility Features
- [x] Comprehensive logging
- [x] Timing and performance tracking
- [x] Storage validation
- [x] Snapshot management

**Note:** This checklist will be updated as implementation progresses and user feedback is gathered.

## Success Criteria

1. ✅ Single entry point (`mb` command) works for all operations
2. ✅ PyTorch training works without FastAI
3. ✅ Keras training works
4. ✅ Multiple architectures supported per framework
5. ✅ Image classification model type implemented
6. ✅ Data pipeline fully functional
7. ✅ Model conversion works
8. ✅ Documentation is complete
9. ✅ Tests pass
10. ✅ Core features from original implementation are available

## Future Enhancements (Post-MVP)

**Note:** These enhancements are planned for future implementation. For the initial MVP, only the superstructure/container for additional model types and frameworks should be implemented - not the actual functionality.

1. **Additional Model Types** (superstructure only)
   - Object detection (container/interface)
   - Image segmentation (container/interface)
   - Text classification (container/interface)

2. **Additional Frameworks** (superstructure only)
   - FastAI (add back as optional framework)
   - JAX/Flax (container/interface)
   - ONNX Runtime (container/interface)

3. **Advanced Features** (not in MVP)
   - Hyperparameter tuning
   - Distributed training
   - Model serving
   - Experiment tracking (MLflow, Weights & Biases)

4. **UI/Web Interface** (not in MVP)
   - Web-based dashboard
   - Visual model comparison
   - Interactive training monitoring

## Next Steps

1. ✅ Plan reviewed and approved
2. Set up development environment
3. Create initial project structure
4. Begin Phase 1 implementation
5. Set up CI/CD for testing
6. Regular progress reviews

---

**Document Version:** 1.1  
**Last Updated:** 2026-01-16  
**Status:** Approved - Ready for Implementation
