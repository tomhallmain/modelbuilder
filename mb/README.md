# Model Builder (mb) Package

This package provides a unified CLI application for building machine learning models.

## Documentation

- **README.md**: Main documentation with usage examples
- **ARCHITECTURE.md**: Detailed architecture and design decisions
- **IMPLEMENTATION_PLAN.md**: Implementation plan and progress

## Package Structure

```
mb/
├── __init__.py          # Package initialization
├── cli.py              # CLI entry point
├── config.py           # Configuration management
├── data/               # Data processing modules
├── models/             # Model-related modules
│   ├── base.py         # Framework trainer base class
│   ├── types.py        # Model type definitions
│   ├── frameworks/     # Framework implementations
│   │   ├── registry.py # Architecture registry
│   │   ├── pytorch/    # PyTorch implementation
│   │   └── keras/      # Keras implementation
│   └── classification/ # Model type implementations
├── training/           # Training orchestration
├── conversion/         # Model conversion
└── utils/              # Shared utilities
    └── logging.py       # Logging configuration
```

See ARCHITECTURE.md for detailed architecture documentation.
