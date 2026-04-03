# Testing plan (Model Builder)

This document turns **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** section **5.1** and **6.1** (“full pipeline end-to-end”) into an actionable roadmap. It focuses on **automated tests** (not manual GUI QA).

## Goals

| Priority | Goal |
|----------|------|
| **P0** | **End-to-end (E2E) image classification**: tiny fixed dataset → data prep → train (PyTorch) → `mb convert` (e.g. ONNX), asserting exit codes and key artifacts. |
| **P1** | **Unit tests** for core abstractions (`TrainingRunArgs`, cancellation helpers, hyperparameter merge, snapshot helpers where mockable). |
| **P2** | **Integration tests** for data pipeline steps (`ImageConverter`, `DatasetCreator`, …) using a temp directory, no full train. |
| **P3** | **CLI tests**: `mb.cli.main([...])` with argv subsets; optional subprocess smoke for the `mb` entry point. |
| **P4** | **Framework-specific training tests** (short runs, CPU): PyTorch first; Keras/TF behind optional markers if TensorFlow is installed. |

## Tooling conventions

- **Runner**: `pytest` (already used under `tests/`).
- **Layout** (recommended as tests are added):

  ```
  tests/
    conftest.py              # shared fixtures, temp dirs, optional torch skip
    unit/                    # fast, no heavy ML deps where possible
    integration/             # data pipeline pieces, temp filesystem
    e2e/                     # full pipeline; marked slow + requires_torch
  tests/fixtures/
    e2e_image_classification/
      raw/                   # or generated in fixture via Pillow
  ```

- **Markers** (define in `pytest.ini` or `pyproject.toml` when you add it):

  - `@pytest.mark.slow` — more than a few seconds; skip in quick local runs (`pytest -m "not slow"`).
  - `@pytest.mark.requires_torch` — skip if `torch` import fails.
  - `@pytest.mark.requires_tf` — skip if TensorFlow not installed (Keras paths).
  - `@pytest.mark.e2e` — full pipeline tests.

- **Collection order:** `tests/conftest.py` defines `pytest_collection_modifyitems` so tests in `test_synthetic_dataset_factory.py` run **first** in the session (pytest’s default order is not guaranteed).

- **CI**: run `unit` + `integration` on every push; run `e2e` on nightly or `main` only, or with `torch` CPU wheel cached. Use **CPU-only** for reproducibility unless you add a self-hosted GPU runner later.

---

## P0 — E2E image classification pipeline

### Intent

One test (or a small chain of tests sharing one workspace) that mirrors the **expected product flow** for image classification:

1. **Inputs**: A **very small** labeled image set committed under `tests/fixtures/` (or generated once per session from deterministic seeds) so tests are reproducible and offline-friendly.

2. **Data layout** (align with `DatasetCreator` / `ModelType` handlers):

   - **Raw** area: class-named subfolders with a few images each (minimum **2 classes**, **1–2 images per class** for train path; the dataset step can split train/test).
   - After **`mb data create-dataset`** (or direct API `DatasetCreator.run()`): under `--data-dir`, **`train/<class>/`** and **`test/<class>/`** with the layout `ModelTrainer` expects (`train_dir`, `test` as val).

3. **Steps** (CLI subprocess **or** Python API with same semantics):

   | Step | Command / API | Success criterion |
   |------|----------------|-------------------|
   | A | Optional: `mb data convert` on raw images | Completes; normalized formats under raw tree |
   | B | `mb data create-dataset` with small `--test-per-class` | Exit 0; `train/` + `test/` class folders exist |
   | C | `mb train` — **PyTorch**, tiny architecture (e.g. `resnet18` if supported, else smallest ResNet in registry), **`--frozen-epochs 0 --unfrozen-epochs 1`** (or equivalent CLI flags), small `--batch-size`, CPU | Exit 0; checkpoint or final model file under `--output-dir` |
   | D | `mb convert --input … --output … --target onnx --framework pytorch --architecture … --num-classes …` | Exit 0; ONNX file exists and is non-empty |

   **Note:** Skip **gather / deduplicate / upscale** in the minimal E2E unless you need coverage for those; they lengthen runtime and widen failure surface. Add separate integration tests for them.

4. **Assertions** (beyond exit codes):

   - Paths exist: `data_dir/train`, `data_dir/test`, at least one exported weight file, converted ONNX path.
   - Optional: ONNX load with `onnx` package if listed as test dependency (or file size / magic bytes only to avoid extra deps).

5. **Performance budget** (CI-friendly):

   - Target **&lt; 5–15 minutes** on CPU for the full E2E, by forcing **1 epoch**, **tiny resolution** if configurable, and **2×2** or **4×4** dummy images only if the pipeline allows (some models require minimum sizes — validate against `image_size` in pipeline config; use **64×64** or **128×128** only if loaders and model support it; otherwise **224×224** with very few steps).

6. **Isolation**:

   - Use `tmp_path` (pytest) for all outputs; never write under the user’s real `data/` or `models/`.
   - Point `reload_pipeline_config` at a **test YAML** under `tests/fixtures/` if defaults are unsuitable.

### Implementation sketch

- **`tests/e2e/test_image_classification_pipeline.py`**

  - Fixture: copy or symlink `tests/fixtures/e2e_image_classification/raw` → `tmp_path / raw`.
  - Invoke CLI via `subprocess.run([sys.executable, "-m", "mb.cli", ...], check=True, cwd=tmp_path, env=...)` **or** call `mb.cli.main([...])` in-process (faster; one process for all steps if logging is configured).
  - Parse `--data-dir` / `--output-dir` consistently with your pipeline YAML.

- **Failure triage**: If E2E fails, split into **sub-tests** (dataset only, train only, convert only) sharing fixtures so CI pinpoints the broken stage.

---

## P1 — Unit tests

Expand beyond current tests (`test_mb_cancellation`, `test_task_context`, `test_training_run_args`):

- **`TrainingRunArgs`**: JSON round-trip, path edge cases (`resume_from` `None`).
- **`mb.cancellation`**: `check_cancel_event` behavior (already partly covered).
- **`get_training_hyperparams`**: merge order — pipeline defaults vs CLI dict (pure function, no I/O).
- **`mb.pipeline_config`**: load minimal YAML string into temp file; assert keys (optional).

Keep these **import-light** so they run without PyTorch.

---

## P2 — Integration tests (data pipeline)

Under `tests/integration/`, each test uses `tmp_path`:

- **`ImageConverter`**: a few synthetic PNGs → expected outputs.
- **`DatasetCreator`**: minimal raw tree → `train`/`test` split counts.
- **Optional**: `ImageGatherer` with tiny copy-only scenario (if fast enough).

Use **real filesystem** operations; avoid network.

---

## P3 — CLI tests

- **`mb.cli.create_parser()`** or **`main([...])`**: unknown command → non-zero; `--version`; `train --help` contains expected flags.
- **`handle_train` with `--train-args-json`**: temp JSON file + minimal dirs (may need `requires_torch`).

Subprocess: one test `subprocess.run(["mb", "--help"], ...)` only if `mb` is on PATH in CI (often use `python -m mb.cli` instead).

---

## P4 — Framework training tests

- **PyTorch**: smoke test `ModelTrainer.train` with `TrainingRunArgs` pointing at **fixture `data_dir`** with 2 classes, **1 epoch**, CPU — marked `slow` + `requires_torch`.
- **Keras**: same pattern behind `requires_tf`; skip in default CI if TF is heavy.

---

## Optional dependencies for CI

Document in `README` or `requirements-dev.txt` (when added):

- `pytest`, `pytest-cov` (optional)
- `torch`, `torchvision` for E2E / train smoke (CPU wheels)
- `onnx` for optional ONNX structural check
- TensorFlow only for Keras-marked tests

---

## Rollout order (recommended)

1. **pytest config** (`pytest.ini`) + markers + `tests/conftest.py` with `skipif` for optional deps.
2. **Fixture tree** + **DatasetCreator** integration test (validates layout before E2E).
3. **E2E image classification** (PyTorch train + ONNX convert).
4. **CLI argparse / main** smoke tests.
5. **Expand unit/integration** coverage for regressions.

---

## Relation to the GUI

Automated E2E here targets **`mb` CLI and Python APIs**. GUI tests (PySide6) are out of scope for this document; if added later, prefer **a few smoke tests** (import `MainWindow`, no crash) or dedicated tools (e.g. `pytest-qt`) in a separate plan.

---

**Document version:** 1.0  
**Last updated:** 2026-04-02
