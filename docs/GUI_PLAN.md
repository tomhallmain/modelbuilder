# Model Builder — GUI Plan

This document describes how the **PySide6** GUI sits on top of the **Model Builder** framework (`mb/` package and CLI). It is a companion to [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (Phase 7 status) and [ARCHITECTURE.md](ARCHITECTURE.md) (GUI layer and tests). It does not replace CLI-first workflows.

## Goals

1. **Reuse, don’t fork** — Business logic stays in `mb` (training, data pipeline, conversion, snapshots). The GUI is a thin layer that invokes stable Python APIs or the same code paths the CLI uses.
2. **Parity with CLI** — Every GUI action should map to an existing or planned `mb` operation so behavior and docs stay aligned.
3. **Local-first** — Default assumption: single-user, local machine, paths chosen by the user; no mandatory cloud account.
4. **Unified dependencies** — **PySide6** (Qt for Python) is listed in the root `requirements.txt` and `install_requires` in `setup.py` with the rest of the app.

## Non-goals (initial phases)

- Replacing the CLI or duplicating configuration formats with a second source of truth.
- Embedding a web engine or shipping a browser-based UI — the desktop shell is **PySide6** only.

## Source layout

The GUI lives in a **separate top-level package** next to `mb/` (not inside `mb/`):

- **`ui/`** — PySide6 app (`app.py`, `main_window.py`, `workspace.py`, `app_theme.py` for global dark styling)
- Entry points: `mb-gui` (see `setup.py` `gui_scripts`) and `python -m ui`

## Architecture

### Layering

```
┌─────────────────────────────────────┐
│  PySide6 UI (main window, dialogs)  │  forms, wizards, log panes
├─────────────────────────────────────┤
│  Adapter / API layer                │  validate paths, map UI → args / config
├─────────────────────────────────────┤
│  mb: ModelTrainer, data modules,    │  unchanged core
│      conversion, snapshot utils     │
└─────────────────────────────────────┘
```

- **Preferred integration:** run the GUI in the same Python process as `mb`: import `mb` modules and call `ModelTrainer`, dataset helpers, and conversion functions with parameters built from widgets (`QLineEdit`, `QFileDialog`, etc.). This matches PySide6’s native embedding model.
- **Acceptable alternative:** `QProcess` to run `python -m mb.cli ...` for long-running jobs if you need hard isolation; parse stdout/stderr into a `QPlainTextEdit` log view.

### Technology stack (fixed)

| Component | Choice | Notes |
|-----------|--------|--------|
| Desktop UI | **PySide6** | Qt 6 bindings; LGPL, official Qt for Python |
| Packaging | TBD | PyInstaller / briefcase / etc. in a later phase |

Older alternatives (local web UI, Electron/Tauri) are **out of scope** for this project’s GUI.

### Configuration

- Reuse YAML + CLI override semantics (`mb/config.py`, existing patterns): the GUI reads/writes the same config files where possible, or generates equivalent CLI argument lists.
- Avoid a parallel “GUI-only” config schema unless necessary; if introduced, document mapping to YAML/CLI.

## Phased deliverables

Aligned with **Phase 7** in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Checkboxes there track status; this file holds rationale and acceptance notes.

### Phase 7.1 — Shell and project model

- PySide6 application entry point (`QApplication`, main window) with navigation placeholders (e.g. `QStackedWidget` or sidebar).
- “Project” or “workspace” concept: root directory, optional config path, remembered last paths (`QSettings`, local only).
- About dialog / version label from `mb.__version__`.

### Phase 7.2 — Data workflow UI

- Forms or wizards for: gather, convert, deduplicate, upscale, create-dataset — mirroring `mb data` subcommands.
- `QFileDialog` / path fields with validation; surface errors from underlying `mb` code clearly (`QMessageBox` or inline labels).
- Read-only view of relevant snapshot summary if present (optional: button to open JSON in system editor).

### Phase 7.3 — Training UI

- Framework and architecture selection (from registry / same lists as CLI).
- Hyperparameter fields with sensible defaults from model type handlers / config.
- Output directory selection; optional validation-only step before launch.
- Long-running job: append logging to a log widget (`QPlainTextEdit` + `logging.Handler`, or tail subprocess output); document cancel (`QProcess::terminate` or worker thread cancellation).

### Phase 7.4 — Conversion and info UI

- Convert flow: input file, target format (`onnx`, `safetensors`), framework detection hints — aligned with `mb convert`.
- Info panels: dataset layout checks, registered architectures (reuse `mb info` behavior or direct imports).

### Phase 7.5 — Packaging and docs

- Document install: `pip install -r requirements.txt` and/or `pip install -e .`.
- Update root `README.md` with a short “GUI (PySide6)” subsection pointing here.
- Smoke test on Windows (primary user environment).

## Security and UX notes

- Treat all paths as user-supplied; never pass unvalidated strings to `QProcess` or `os.system`.
- Keep file pickers defaulting to user-accessible directories; avoid requiring elevated privileges.

## Relationship to CLI

- The CLI remains the contract for automation and CI; the GUI is an optional façade.
- New capabilities should be added to `mb` first, then exposed in the PySide6 UI.

---

**Document version:** 1.1  
**Last updated:** 2026-03-27
