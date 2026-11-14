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

## Quick start (text classification)

Binary classifiers over short strings, trained from a labelled `dataset.tsv` (`text`,
`label`, `split`; optional `tier`, `source`, `category_hint`, `group` and a reference-score
column) plus an optional `manifest.json`, `label_conflicts.tsv` and unlabeled file. Tier
roles (`gold_tiers`, `unreviewed_tier`), the reference-score column and the unlabeled file
name are set per dataset in the config. Everything is
configured by the pipeline YAML's `text_classification` section (see
`mb/config/text_classification.example.yaml`); unknown keys there are errors.

On Windows, PyPI only has CPU builds of torch. If installing an extra makes pip upgrade torch
(e.g. to satisfy a newer `transformers`), a CUDA build is silently replaced by a CPU one. Check
afterwards with `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`
and, if needed, restore CUDA without touching other packages:
`pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128`
(choose the CUDA tag for your driver at https://pytorch.org/get-started/locally/).

```bash
pip install -e ".[text]"            # char_ngram_linear (CPU)
pip install -e ".[pytorch,text]"    # also embedding_probe / encoder_finetune

mb --config my_text_pipeline.yaml text verify     # hashes, counts, uniqueness, group/split isolation
mb --config my_text_pipeline.yaml train           # model.default_type: text_classification
mb train --model-type text_classification --text-backend encoder_finetune \
    --text-model-id answerdotai/ModernBERT-base --epochs 3
mb text evaluate --model data/models/text_runs/<run>
mb text score --model data/models/text_runs/<run> --input lines.txt --output scores.tsv
mb text compare "data/models/text_runs/*"
```

Each run writes `<runs_dir>/<timestamp>_<backend>_<model>/` with `config.yaml`,
`environment.json`, `model/`, `calibrator.json`, `thresholds.json`, `metrics.json`,
`predictions_test.tsv`, `review_queue.tsv`, `unlabeled_scores.tsv`, `extra_reports.json`,
`MODEL_CARD.md` and `predict.py`. `predict.py` scores strings without `mb` installed (only numpy
and the backend's libraries): `python predict.py "some string"`, or
`from predict import load; load().score([...])`. In the GUI: Train page (model type `text_classification`), Text page
(verify/evaluate/score/compare), Pipeline page (Text classification tab).

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

Relative paths inside a pipeline YAML (`data.raw_data_dir`, `data.data_dir`, `paths.models_dir`, `text_classification.data_dir` / `runs_dir`) resolve against the folder holding that YAML, or against the GUI workspace root when the YAML lies inside it. With no YAML of your own (built-in defaults, or the `pipeline.yaml` the GUI saves in the app data directory) they resolve against the GUI workspace root if one is set, else `Documents/ModelBuilder` (set `MODELBUILDER_WORK_DIR` to choose another folder). Paths given on the command line (`--data-dir`, `--config`, …) resolve from the current directory as usual.

## Encrypted cache keys

The app info cache (`app_info_cache.enc`, with its `.bak*` backups) is encrypted and sits in `cache/` under the app data directory: `%APPDATA%\ModelBuilder` on Windows, `~/.local/share/ModelBuilder` elsewhere, next to `logs/` and `application.yaml`. Cache files left in the repo root by older versions are moved there on first start, unless a file of the same name is already there, in which case nothing is moved and a warning names the clash. Its keys sit in `MyPersonalApplicationsService/modelbuilder__key_store.json` under `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS and `$XDG_DATA_HOME` (default `~/.local/share`) elsewhere; only the passphrase protecting them stays in the OS keychain. Keys kept in the keychain by older versions are moved into that file on first start. After every write the key store is backed up to an external drive (set `MODELBUILDER_KEY_BACKUP_DIR` to choose where). Run `python scripts/key_material.py` to check that backup's state, or `python scripts/key_material.py backup` to take one by hand.

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
