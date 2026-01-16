# Model Builder (mb) Package

This package provides a unified CLI application for building machine learning models.

## Phase 1 Implementation Status

✅ **Completed:**
- Project structure created
- Core abstractions implemented:
  - `ModelType` enum and handlers
  - `FrameworkTrainer` abstract base class
  - Architecture registry system
  - Configuration management
- CLI foundation with all subcommands defined

## Package Structure

```
mb/
├── __init__.py          # Package initialization
├── cli.py              # CLI entry point
├── config.py           # Configuration management
├── data/               # Data processing modules (Phase 2)
├── models/             # Model-related modules
│   ├── base.py         # Framework trainer base class
│   ├── types.py        # Model type definitions
│   ├── frameworks/     # Framework implementations
│   │   ├── registry.py # Architecture registry
│   │   ├── pytorch/    # PyTorch implementation (Phase 3)
│   │   └── keras/      # Keras implementation (Phase 3)
│   └── classification/ # Model type implementations (Phase 3)
├── training/           # Training orchestration (Phase 3)
├── conversion/         # Model conversion (Phase 4)
└── utils/              # Shared utilities
    └── logging.py       # Logging configuration
```

## Usage

### Installation

```bash
# Install package in development mode
pip install -e .

# Or install with specific framework support
pip install -e .[pytorch]
pip install -e .[keras]
pip install -e .[all]
```

### CLI Commands

```bash
# Show help
mb --help

# Data operations (Phase 2 - stubs only)
mb data gather --source-dir PATH --subdirs DIR1 DIR2
mb data convert --raw-data-dir PATH
mb data deduplicate --raw-data-dir PATH
mb data upscale --raw-data-dir PATH
mb data create-dataset --raw-data-dir PATH --data-dir PATH

# Training (Phase 3-4 - stubs only)
mb train --model-type image_classification --framework pytorch

# Model conversion (Phase 4 - stubs only)
mb convert --input MODEL.pth --output MODEL.h5

# Information (stubs only)
mb info model --path MODEL.pth
mb info dataset --data-dir PATH
```

## Configuration

Configuration can be provided via:
1. Default values (built-in)
2. YAML config file (`configs/default.yaml`)
3. Command-line arguments (override config)

## Next Steps

- **Phase 2:** Implement data pipeline modules
- **Phase 3:** Implement PyTorch and Keras trainers
- **Phase 4:** Complete training CLI and model conversion
