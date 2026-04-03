# Model Builder Architecture

This document describes the architecture and design decisions for the Model Builder application.

## Overview

Model Builder is a unified **CLI-first** application (`mb`) for building machine learning models, with an optional **desktop GUI** (`ui/`, PySide6) that calls the same `mb` modules as the command line. It provides a framework-agnostic interface for training using PyTorch or Keras/TensorFlow. Image classification is implemented today; model types and frameworks are extensible via registries and handlers.

**Companion docs:** phased delivery and checklists — [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md); GUI goals — [GUI_PLAN.md](GUI_PLAN.md); safety and CLI-vs-GUI behavior — [GUI_BACKEND_PIPELINE_REVIEW.md](GUI_BACKEND_PIPELINE_REVIEW.md).

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

### 8. Desktop GUI (`ui/`)

The GUI is a **separate top-level package** (not inside `mb/`) so the CLI stays import-clean and packageable on headless systems if ever needed.

| Area | Role |
|------|------|
| **`ui/app.py`** | `QApplication` bootstrap (`mb-gui` / `python -m ui`) |
| **`ui/main_window.py`** | Main shell: sidebar nav, central **`QStackedWidget`** with object name `main_nav_stack`, workspace + optional YAML paths, About |
| **`ui/workspace.py`** | Workspace model (root dir, config path) persisted via `QSettings` |
| **`ui/pages/`** | **Data** (gather, convert, dedupe, upscale, create-dataset), **Train**, **Convert**, **Info**, **Home** — forms map to the same classes/functions `mb/cli.py` uses |
| **Long work** | **`ui/task_runner.py`** runs callables on `QThreadPool`; **`TaskSignals`** + `Qt.QueuedConnection` keep widget updates on the GUI thread |
| **`ui/task_context.py`** | **`LongTaskContext`**: cooperative **`cancel_event`**, **`progress`** callbacks |
| **`ui/lib/task_progress.py`** | Attaches a modal **`QProgressDialog`** to long tasks when the backend reports progress |
| **`ui/lib/qt_alert.py`** | **`qt_alert`**, **`qt_operation_error`** (modal failures, copy details, open log folder) |
| **`ui/main_thread_bridge.py`** | **`MainThreadBridge`**: `notification_manager` / **`AppActions`** paths that originate off the GUI thread must not touch widgets directly — bridge uses `QMetaObject.invokeMethod` |
| **`ui/spawn_mb_train.py`** | Optional **detached** `mb train --train-args-json …` subprocess so training survives closing the app |

**Configuration (two layers):** application shell settings (`utils.config` / packaged `mb/config/application.example.yaml`, or workspace `configs/application.yaml`, or legacy keys in `default.yaml`) vs pipeline/ML config (`mb.pipeline_config` / workspace `configs/pipeline.yaml`). The main window reloads both when the workspace or config file changes.

**Stable widget object names** (for headless UI tests): e.g. `main_nav_stack`, `train_page`, `train_architecture_edit`, `train_data_dir_edit`, `train_validate_btn`, `train_start_btn`, `train_output_log` (see `ui/pages/train_page.py`).

### 9. Automated tests (`tests/`)

| Directory | Purpose |
|-----------|---------|
| **`tests/fixtures/`** | Shared synthetic dataset builders |
| **`tests/integration/`** | Temp filesystem: data pipeline modules, no full train unless noted |
| **`tests/framework/`** | Optional short training smoke (torch / TF markers) |
| **`tests/unit/`** | Fast tests: CLI, cancellation, config, cache isolation, … |
| **`tests/ui/`** | Headless PySide6 (`pytest-qt`, `QT_QPA_PLATFORM=offscreen` in `tests/ui/conftest.py`) |
| **`tests/e2e/`** | Full pipeline (slow; may require `onnx` for export) |

Root **`tests/conftest.py`** sets **`MODELBUILDER_TEST_CACHE=1`** so tests do not touch the real encrypted app-info cache, and sorts collection order (synthetic fixture smoke → integration → framework → other → UI → E2E). Details and open follow-ups: **Phase 5** in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

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
- Optional GUI UX changes (non-modal progress, error banners, stricter CLI–GUI parity) — tracked in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

All without breaking existing functionality.
