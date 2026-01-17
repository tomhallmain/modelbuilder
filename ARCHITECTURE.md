# Model Builder Architecture

This document describes the architecture and design decisions for the Model Builder application.

## Overview

Model Builder is a unified CLI application for building machine learning models. It provides a framework-agnostic interface for training models using PyTorch or Keras/TensorFlow. Currently supports image classification, with an extensible architecture for additional model types.

## Core Design Principles

1. **Framework Agnosticism**: Training logic is abstracted from specific frameworks
2. **Extensibility**: Easy to add new model types, architectures, and frameworks
3. **Separation of Concerns**: Clear boundaries between data processing, model training, and conversion
4. **Configuration Flexibility**: Multiple configuration sources with clear precedence

## Architecture Layers

### 1. CLI Layer (`mb/cli.py`)

The command-line interface provides a unified entry point for all operations. Uses `argparse` for argument parsing with a subcommand structure:

- `mb data *` - Data processing operations
- `mb train` - Model training
- `mb convert` - Model format conversion
- `mb info *` - Information queries

### 2. Configuration Layer (`mb/config.py`)

Configuration management with precedence order:
1. CLI arguments (highest priority)
2. YAML config file (`configs/default.yaml`)
3. Default values (lowest priority)

Configuration is accessed via dot notation (e.g., `config.get('model.default_framework')`).

### 3. Data Processing Layer (`mb/data/`)

Modular data processing pipeline. Current implementation focuses on image data:
- **gather.py**: Collects data files from source directories
- **convert.py**: Converts data to target format
- **deduplicate.py**: Removes duplicate data files
- **upscale.py**: Upscales small data files (image-specific)
- **dataset.py**: Creates train/test splits

Each module is self-contained and can be run independently or as part of the pipeline. Future model types may require different data processing modules.

### 4. Model Layer (`mb/models/`)

#### 4.1 Model Types (`mb/models/types.py`)

Defines supported model types via `ModelType` enum and `ModelTypeHandler` abstract base class. Each model type handler knows:
- How to determine number of classes from data structure
- How to validate data structure
- Default hyperparameters for that model type

Currently implemented:
- `ImageClassificationHandler`: Handles image classification tasks (example implementation)

#### 4.2 Framework Abstraction (`mb/models/base.py`)

`FrameworkTrainer` abstract base class defines the interface all framework implementations must follow:
- `create_model()`: Create model instance
- `create_data_loaders()`: Create data loaders/generators
- `train()`: Train the model
- `evaluate()`: Evaluate the model
- `save_model()` / `load_model()`: Model persistence

#### 4.3 Framework Implementations (`mb/models/frameworks/`)

**PyTorch** (`mb/models/frameworks/pytorch/`):
- `trainer.py`: Implements `FrameworkTrainer` for PyTorch
- `data_loader.py`: PyTorch `DataLoader` with transforms
- `architectures.py`: Architecture factory functions

**Keras** (`mb/models/frameworks/keras/`):
- `trainer.py`: Implements `FrameworkTrainer` for Keras
- `data_loader.py`: Keras `ImageDataGenerator` with augmentation
- `architectures.py`: Architecture factory functions

#### 4.4 Architecture Registry (`mb/models/frameworks/registry.py`)

Global registry for model architectures. Allows dynamic registration and lookup of architectures by framework and name:

```python
from mb.models.frameworks.registry import register_architecture, get_architecture

# Register an architecture
register_architecture('pytorch', 'resnet34', factory_function)

# Get an architecture
factory = get_architecture('pytorch', 'resnet34')
model = factory(num_classes=3, pretrained=True)
```

### 5. Training Layer (`mb/training/`)

#### 5.1 Training Orchestrator (`mb/training/trainer.py`)

`ModelTrainer` class provides a framework-agnostic interface for training:
- Framework selection
- Model creation
- Data loading
- Training execution
- Model saving

#### 5.2 Hyperparameter Management (`mb/training/hyperparams.py`)

`HyperparameterManager` merges hyperparameters from multiple sources:
1. CLI arguments
2. Config file
3. Model type defaults

#### 5.3 Snapshot Integration (`mb/training/snapshot_integration.py`)

Updates unified snapshots with training data, tracking which data samples were used for training.

### 6. Conversion Layer (`mb/conversion/`)

Model format conversion utilities:
- **PyTorch → ONNX**: For cross-platform deployment
- **PyTorch → SafeTensors**: For safer model storage
- **Keras → ONNX**: For cross-platform deployment

Auto-detects source framework from file extension/content.

### 7. Utilities Layer (`mb/utils/`)

Shared utilities:
- **logging.py**: Centralized logging configuration
- **snapshot.py**: Unified snapshot system for tracking data through pipeline
- **storage.py**: Storage validation (internal vs external drives)
- **timing.py**: Performance tracking utilities

## Design Patterns

### Strategy Pattern

Framework implementations use the Strategy pattern via `FrameworkTrainer` abstract base class. This allows swapping frameworks without changing training orchestration code.

### Registry Pattern

Architecture registry allows dynamic registration and lookup of model architectures without hardcoding them in the training code.

### Factory Pattern

Architecture factory functions create model instances. Registered in the architecture registry and called dynamically based on user selection.

## Data Flow

### Training Pipeline

1. **CLI** parses arguments and loads configuration
2. **ModelTrainer** validates data structure and determines number of classes
3. **FrameworkTrainer** creates model and data loaders
4. **Training** executes in two phases:
   - Frozen phase: Train classifier head only
   - Unfrozen phase: Fine-tune all layers
5. **Evaluation** computes metrics
6. **Snapshot** updated with training data
7. **Model** saved to disk

### Data Processing Pipeline

1. **Gather**: Collect data files from source directories
2. **Convert**: Convert to target format
3. **Deduplicate**: Remove duplicates
4. **Upscale**: Upscale small data files (optional, model-type specific)
5. **Create Dataset**: Create train/test splits

Each step updates the unified snapshot for traceability. The specific operations depend on the model type being used.

## Extension Points

### Adding a New Framework

1. Create new directory under `mb/models/frameworks/`
2. Implement `FrameworkTrainer` interface
3. Create data loader module
4. Create architectures module with registration
5. Update `ModelTrainer` to support new framework

### Adding a New Model Type

1. Add enum value to `ModelType`
2. Create handler class inheriting from `ModelTypeHandler`
3. Implement required methods
4. Register handler in `mb/models/types.py`

### Adding a New Architecture

1. Create factory function in appropriate framework's `architectures.py`
2. Register with `register_architecture()`
3. Architecture becomes available automatically

## Configuration

Configuration is hierarchical and supports:
- Default values (built-in)
- YAML file (`configs/default.yaml`)
- CLI arguments (override)

Access via `config.get('path.to.value')` with dot notation.

## Snapshot System

The unified snapshot system tracks data samples through the entire pipeline, enabling full traceability from training data back to source. The specific data tracked depends on the model type:

- **Original**: Source data files
- **Converted**: After format conversion
- **Dataset**: After dataset creation
- **Training**: Data samples used in training

## Error Handling

- Framework-specific errors are caught and logged
- Missing dependencies are detected with helpful error messages
- Data validation occurs before training starts
- Checkpoint resumption handles partial training

## Future Enhancements

The architecture supports future additions:
- Additional model types (object detection, segmentation)
- Additional frameworks (FastAI, JAX)
- Additional architectures (via registry)
- Additional conversion formats

All without breaking existing functionality.
