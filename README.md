# Model Builder

CLI-first toolkit for training image-classification models with **PyTorch** or **Keras/TensorFlow**. Optional **PySide6** desktop UI (`mb-gui`) calls the same `mb` APIs as the command line.

> **Disclaimer:** Active development; not fully battle-tested.

## Features

- **Framework Agnostic**: Train models with PyTorch or Keras
- **Multiple Architectures**: ResNet, EfficientNet, and more
- **Data Pipeline**: Modular data processing pipeline (gather, convert, deduplicate, dataset creation)
- **Transfer Learning**: Two-phase training (frozen/unfrozen) with learning rate scheduling
- **Model Conversion**: Convert models to ONNX or SafeTensors format
- **Snapshot Tracking**: Track data samples through the entire pipeline

## Install

```bash
pip install -e .
# Optional extras: .[pytorch]  .[keras]  .[all]
```

Launch GUI after install: `mb-gui` or `python -m ui`. Rationale and Phase 7 detail: [docs/GUI_PLAN.md](docs/GUI_PLAN.md), [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Tests

From repo root with `requirements.txt` installed: `python -m pytest tests/`

Order and layout are controlled in `tests/conftest.py` (`integration/` → `framework/` → `unit/` & co. → `ui/` → `e2e/`). More context: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) (Phase 5). **`tests/e2e/`** runs train + ONNX via `mb.cli.main`. **`ui_e2e`** includes the same-sized PyTorch+ONNX path **through the Train and Convert pages** (`test_ui_e2e_headless.py`) plus a fast shell-only test (nav + About). E2E may need PyTorch + **onnx** (`pip install -e ".[onnx]"` or `.[all]`); missing deps **skip** tests. Quick run: `pytest tests/ -m "not slow"`. Skip reasons: `pytest tests/ -rs`.

## Quick start (image classification)

```bash
mb data gather --source-dir /path/to/source --subdirs dir1 dir2 --target-count 16000
mb data convert --raw-data-dir raw_data
mb data deduplicate --raw-data-dir raw_data
mb data create-dataset --raw-data-dir raw_data --data-dir data --test-per-class 1000

mb train --framework pytorch --architecture resnet34 --data-dir data
# mb train --framework keras --architecture resnet50 --data-dir data

mb convert --input model.pth --output model.onnx --target onnx \
  --architecture resnet34 --num-classes 3
```

Use `mb --help` and `mb <subcommand> --help` for full flags. Config precedence: defaults → YAML (e.g. `configs/default.yaml`) → CLI.

## Architectures (examples)

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

## Layout

Python package **`mb/`** holds CLI, data modules, trainers, conversion. **`ui/`** is the desktop shell. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, GUI, tests, design |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phases, GUI/testing follow-ups |
| [docs/GUI_PLAN.md](docs/GUI_PLAN.md) | GUI goals |
| [docs/GUI_BACKEND_PIPELINE_REVIEW.md](docs/GUI_BACKEND_PIPELINE_REVIEW.md) | Safety, threading, CLI vs GUI notes |

## License

[Your License Here]
