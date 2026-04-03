# Model Builder

CLI-first toolkit for training image-classification models with **PyTorch** or **Keras/TensorFlow**. Optional **PySide6** desktop UI (`mb-gui`) calls the same `mb` APIs as the command line.

> **Disclaimer:** Active development; not fully battle-tested.

## Features

- Data pipeline: gather, convert, deduplicate, upscale, train/test splits  
- Two-phase transfer learning and conversion (ONNX, SafeTensors)  
- Snapshot-style provenance across pipeline steps  

## Install

```bash
pip install -e .
# Optional extras: .[pytorch]  .[keras]  .[all]
```

Launch GUI after install: `mb-gui` or `python -m ui`. Rationale and Phase 7 detail: [docs/GUI_PLAN.md](docs/GUI_PLAN.md), [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Tests

From repo root with `requirements.txt` installed: `python -m pytest tests/`

Order and layout are controlled in `tests/conftest.py` (`integration/` → `framework/` → `unit/` & co. → `ui/` → `e2e/`). More context: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) (Phase 5). E2E may need PyTorch + **onnx** (`pip install -e ".[onnx]"` or `.[all]`); missing deps **skip** tests. Quick run: `pytest tests/ -m "not slow"`. Skip reasons: `pytest tests/ -rs`.

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

**PyTorch:** resnet18–152, efficientnet_b0/b1 · **Keras:** resnet50–152, efficientnet_b0/b1  

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
