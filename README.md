# Model Builder

A unified CLI application for building machine learning models. Supports PyTorch and Keras/TensorFlow frameworks. Currently implements image classification, with an extensible architecture for additional model types.

> **⚠️ Disclaimer**: This project is currently in active development and has not been thoroughly tested.

## Features

- **Framework Agnostic**: Train models with PyTorch or Keras
- **Multiple Architectures**: ResNet, EfficientNet, and more
- **Data Pipeline**: Modular data processing pipeline (gather, convert, deduplicate, dataset creation)
- **Transfer Learning**: Two-phase training (frozen/unfrozen) with learning rate scheduling
- **Model Conversion**: Convert models to ONNX or SafeTensors format
- **Snapshot Tracking**: Track data samples through the entire pipeline

## Installation

```bash
# Install package
pip install -e .

# Install with specific framework support
pip install -e .[pytorch]    # PyTorch only
pip install -e .[keras]       # Keras/TensorFlow only
pip install -e .[all]         # All frameworks
```

### Desktop GUI (PySide6)

**Phase 7.1** shell: sidebar navigation, workspace folder + optional YAML config (persisted), About dialog with `mb` version. PySide6 is part of the default dependency set (`requirements.txt` / `pip install -e .`).

```bash
pip install -e .
mb-gui
# or: python -m ui
```

See [docs/GUI_PLAN.md](docs/GUI_PLAN.md) for the full UI roadmap. Data/training/convert screens are placeholders until later Phase 7 tasks.

## Quick Start

### 1. Prepare Data

The following example shows data preparation for image classification:

```bash
# Gather data files
mb data gather --source-dir /path/to/source --subdirs dir1 dir2 --target-count 16000

# Convert to target format
mb data convert --raw-data-dir raw_data

# Remove duplicates
mb data deduplicate --raw-data-dir raw_data

# Create train/test splits
mb data create-dataset --raw-data-dir raw_data --data-dir data --test-per-class 1000
```

### 2. Train Model

```bash
# Train with PyTorch
mb train --framework pytorch --architecture resnet34 --data-dir data

# Train with Keras
mb train --framework keras --architecture resnet50 --data-dir data

# With custom hyperparameters
mb train --framework pytorch --architecture resnet34 \
    --frozen-epochs 5 --unfrozen-epochs 20 \
    --batch-size 32 --image-size 224
```

### 3. Convert Model

```bash
# Convert PyTorch to ONNX
mb convert --input model.pth --output model.onnx --target onnx \
    --architecture resnet34 --num-classes 3

# Convert PyTorch to SafeTensors
mb convert --input model.pth --output model.safetensors --target safetensors
```

## CLI Commands

### Data Operations

```bash
mb data gather --source-dir PATH --subdirs DIR1 DIR2 [--target-count N]
mb data convert --raw-data-dir PATH [--format jpeg]
mb data deduplicate --raw-data-dir PATH
mb data upscale --raw-data-dir PATH [--review-dir PATH]
mb data create-dataset --raw-data-dir PATH --data-dir PATH [--test-per-class N]
```

### Training

```bash
mb train [--framework pytorch|keras] [--architecture NAME] \
    [--data-dir PATH] [--output-dir PATH] \
    [--frozen-epochs N] [--unfrozen-epochs N] \
    [--batch-size N] [--image-size N] \
    [--resume-from PATH] [--run-id ID]
```

### Conversion

```bash
mb convert --input PATH --output PATH --target onnx|safetensors \
    [--framework pytorch|keras] \
    [--architecture NAME] [--num-classes N] [--image-size N]
```

### Information

```bash
mb info model --path PATH
mb info dataset --data-dir PATH
```

## Configuration

Configuration can be provided via:
1. Default values (built-in)
2. YAML config file (`configs/default.yaml`)
3. Command-line arguments (highest priority)

See `ARCHITECTURE.md` for detailed architecture documentation.

## Supported Architectures

**PyTorch**: resnet18, resnet34, resnet50, resnet101, resnet152, efficientnet_b0, efficientnet_b1

**Keras**: resnet50, resnet101, resnet152, efficientnet_b0, efficientnet_b1

## Project Structure

```
mb/
├── cli.py              # CLI entry point
├── config.py           # Configuration management
├── data/               # Data processing modules
├── models/             # Model abstractions and implementations
│   ├── base.py         # Framework trainer interface
│   ├── types.py        # Model type handlers
│   └── frameworks/     # PyTorch and Keras implementations
├── training/           # Training orchestration
├── conversion/         # Model conversion utilities
└── utils/              # Shared utilities
```

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Architecture and design decisions  
- **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** — Implementation plan and progress  
- **[docs/GUI_PLAN.md](docs/GUI_PLAN.md)** — Planned PySide6 GUI (Phase 7)

## License

[Your License Here]
