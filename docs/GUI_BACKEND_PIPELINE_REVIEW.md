# GUI + Backend Pipeline Review

**Purpose:** Assess architectural pitfalls, data-loss risks, and behavioral parity when using the PySide6 UI (`ui/`) versus the `mb` CLI—especially before large image-classification runs.

**Related docs:** [GUI_PLAN.md](GUI_PLAN.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Architecture: GUI on top of `mb`

### How it works today

- The GUI imports **`mb` Python modules** directly (`ImageGatherer`, `ImageConverter`, `ModelTrainer`, `convert_model`, etc.)—the same classes/functions the CLI handlers in `mb/cli.py` use.
- Long operations run on a **background thread pool** (`ui/task_runner.py`) so the window stays responsive.

### Strengths

- **Single source of truth:** No duplicate business logic between “script” and UI for wired commands.
- **Parity:** Intended behavior matches `mb data …`, `mb train`, `mb convert`, etc., when the same paths and options are used.

### Pitfalls (general)

| Topic | Risk | Notes |
|--------|------|--------|
| **Process coupling** | Closing the app or a crash can stop a long job mid-run | Same as killing a CLI process: partial outputs may exist. **Normal File → Exit:** `MainWindow.closeEvent` warns if **`QThreadPool.globalInstance().activeThreadCount() > 0`**, then stops the cache timer, stores app info cache, saves workspace—still **does not wait** for pool work to finish if the user confirms exit. **Force quit** skips even that flush. |
| **Progress in the UI (not log dumps)** | — | **`mb` uses `logging`** for full traces. Long GUI operations use **`TaskSignals.progress`** + **`ui/lib/task_progress.attach_progress_dialog`** (modal progress; phase text + bar when the backend reports a 0–1 fraction). Per-page **`QTextEdit`** still logs high-level lines. **Errors:** **`qt_operation_error`** can open the log folder (§7.6). |
| **Cancellation** | Long jobs need cooperative stop | **`LongTaskContext.cancel_event`** + **`attach_progress_dialog(..., cancellable=True)`**. **`mb.cancellation.check_cancel_event`** in training (epoch boundaries) and in **data** **`run()`** methods (gather/convert/dedupe/upscale/dataset) plus **`convert_model`** (before work starts). Cancel → **`OperationCancelled`** → **`TaskSignals.cancelled`**. |
| **Threading + Qt (main-thread bridge)** | Workers must not touch QWidget APIs directly | **`ui/task_runner.py`** uses **`TaskSignals`** + **`Qt.QueuedConnection`**. **`NotificationController`** uses signals for toasts / `title_notify`. **`ui/main_thread_bridge.py`** (`MainThreadBridge`) marshals **`AppActions.title`** and **`AppActions._alert`** onto the GUI thread so **`notification_manager`** timers and other non-GUI callers stay safe. |
| **Config** | ~~Single merged YAML for everything~~ | **Two layers:** application (`gui`/`app`) vs pipeline (`model`/`data`/`training`/`paths`). See §7.5. Train uses **`get_pipeline_config()`**; shell uses **`get_application_config()`**. |
| **Error handling in the UI** | Failures must be visible without relying on log files alone | **Addressed in §7.6:** **`qt_operation_error`** (with copy + open log folder), **`qt_alert`**, **`task_runner`** logging; modal task failures vs toasts for supplementary feedback. |

*Expanded discussion for each row:* **§7**.

---

## 2. End-to-end pipeline (recommended order)

Typical image-classification flow implied by `mb` and `mb/data/dataset.py`:

1. **Gather** (`mb data gather` / Data → Gather) — copy samples into a coherent/raw layout under `target_dir` (e.g. `raw_data/coherent/...`).
2. **Convert** (`mb data convert` / Data → Convert) — normalize to JPEG under each class (often under `JPEG_IMAGES/`), and build/update **unified snapshot** metadata.
3. **Deduplicate** (optional, `mb data deduplicate`) — **destructive** within `raw_data_dir` (see §3).
4. **Upscale** (optional, `mb data upscale`) — reads from **review** folders (e.g. small images); writes upscaled outputs under the review tree.
5. **Create dataset** (`mb data create-dataset` / Data → Create Dataset) — **requires a unified snapshot** from convert; writes **`data_dir/train`** and **`data_dir/test`**; can delete, move, or balance files (see §3).
6. **Train** (`mb train` / Train) — reads `data_dir/train` and `data_dir/test`; writes model files under **output dir** (e.g. `.pth` / `.h5`).
7. **Convert model** (`mb convert` / Convert page) — reads an existing checkpoint; writes a **new** file (ONNX / SafeTensors); does not delete the input.

**Critical:** Step **create-dataset** will **fail** if no unified snapshot exists (`DatasetCreator.run()` logs an error and returns `False`). The message tells you to run conversion (snapshot creation) first. This is backend behavior, not GUI-specific.

---

## 3. Destructive or irreversible operations (by step)

This section answers: *“Could I lose my dataset?”*

### Gather (`ImageGatherer`)

- **Writes:** Copies (and possibly converts) into `target_dir` run subfolders; maintains cache under raw data.
- **Deletes:** Can **remove duplicate files** inside directories during gather’s internal dedupe passes (`gather.py` uses `unlink` on duplicates in some paths).
- **Data loss risk:** Usually **does not delete your original source tree**; risk is mainly **duplicate removal inside target/gather working areas** and **overwriting** if you point `target_dir` at something important.

### Convert (`ImageConverter`)

- **Writes:** JPEG outputs and snapshot JSON; may create `JPEG_IMAGES` subtrees per class.
- **Deletes:** Not typically “wipe dataset”; may replace or skip per implementation—**review `mb/data/convert.py`** for your version if you rely on originals in-place.

### Deduplicate (`ImageDeduplicator`)

- **Destructive:** **Removes duplicate files** (`unlink`) and may **remove very small images** or **move** files to review—**inside `raw_data_dir`**.
- **Data loss risk:** **High** if `raw_data_dir` is your only copy of images. **Back up** before running dedupe on a precious folder.

### Upscale (`ImageUpscaler`)

- **Writes:** New upscaled images under the review / upscaled output paths.
- **Deletes:** Primarily additive; still **validate paths** so you do not point `review_dir` at a folder you only have one copy of if the implementation moves files (check `mb/data/upscale.py` for your run).

### Create dataset (`DatasetCreator`)

- **Requires:** **Unified snapshot** present (from convert pipeline). Without it, run **aborts**—no train/test folder build.
- **Copies** from raw (`JPEG_IMAGES` per class) into `data_dir/train/...`.
- **Destructive inside `data_dir`:**
  - **Deletes** “corrupted” images (failed PIL verify) from **train** copies.
  - **Moves** invalid-size images to `data_dir/invalid_size_review/`.
  - **Deletes** excess training images when **balancing** or **max_train_per_class** is used.
  - **Moves** a random subset from train → test (files **leave** train; they are not copied).
- **Raw data:** Source under `raw_data_dir` is generally **copied from**, not deleted—but **always** use a dedicated `data_dir` for outputs and **back up** before re-running on the same `data_dir` if you care about previous train/test splits.

**Hardcoded classes:** `DatasetCreator` uses fixed class names (`coherent`, `semi-incoherent`, `incoherent` in `mb/data/dataset.py`). If your directory uses different folder names, counts may be zero or behavior may not match expectations—**verify layout matches the code** before large runs.

### Train (`ModelTrainer`)

- **Writes/overwrites:** Saves `*_model.pth` or `*_model.h5` under **output_dir**; may write checkpoints during training.
- **Deletes:** Does not delete your image dataset; **can overwrite** previous model files with the same names.

### Model convert (`convert_model`)

- **Writes** new ONNX/SafeTensors files.
- **Does not delete** the input model file.

---

## 4. GUI vs CLI behavioral gaps

| Area | CLI (`mb`) | GUI (`ui/pages`) |
|------|------------|------------------|
| **Same-drive confirmation (create-dataset)** | May call `confirm_user_action()` with **`input()`** for edge cases (non–system drive) | Uses **`QMessageBox.question`** only when `check_same_drive` is true—**not** identical to every CLI branch of `confirm_user_action` |
| **External storage** | `--allow-external-storage` | Checkbox on Create Dataset tab; same `check_target_external_storage` |
| **Training / pipeline defaults** | `--config PATH` (pipeline only) | **`mb.pipeline_config`**: CLI train uses **`--config`** for pipeline YAML; GUI loads **`configs/pipeline.yaml`** or legacy combined **`configs/default.yaml`** under the workspace when present |
| **Logging** | Console + log files | QTextEdit snippets + **same** underlying loggers (check project log directory for full traces) |

These gaps are **operational**, not “wrong backend”—but for parity testing, prefer **`mb …` from the command line** with the same paths once, then mirror in the GUI.

---

## 5. Recommendations before testing on a large image directory

1. **Copy or backup** the image root (or use a **staging copy** of `raw_data`) before **deduplicate** or **create-dataset** with balancing enabled.
2. **Run convert** and confirm a **unified snapshot** exists before **create-dataset** (CLI or GUI will both fail otherwise by design).
3. Use a **fresh or dedicated `data_dir`** (e.g. `data_run_2026`) so you do not merge with an old train/test tree unintentionally.
4. **Align folder names** with `CLASS_NAMES` in `dataset.py` or adjust the code for your taxonomy before scaling up.
5. For **training**, point **data_dir** at the folder that contains **`train/` and `test/`** subfolders with per-class subdirs.
6. Monitor **disk space**: gather + copy + train checkpoints can use **multiples** of raw data size.
7. Prefer **CLI + log files** for the first full pipeline dry run; use the GUI once paths are proven.

---

## 6. Summary verdict

- **Architecture:** GUI-on-`mb` is sound: same modules, fewer duplicated bugs than a parallel script layer.
- **Safety:** The **backend** (not the GUI alone) contains **intentionally destructive** steps (dedupe, dataset cleaning, balancing, train→test moves). Treat **deduplicate** and **create-dataset** as **production-affecting** unless paths are disposable copies.
- **“Will it work for image classification?”** Yes, **if** directory layout, snapshot, and class names match what `DatasetCreator` and `ModelType` handlers expect; otherwise fix data layout or extend the code for custom class sets.

---

## 7. Pitfall details (expanded)

The following subsections mirror the **Pitfalls (general)** table in §1 and spell out intent, constraints, and how the current `ui/` code addresses them where applicable.

### 7.1 Process coupling and job lifecycle

The GUI and the `mb` work run in the **same OS process**. **`MainWindow.closeEvent`** asks for confirmation when **`QThreadPool.globalInstance().activeThreadCount() > 0`**, then stops the periodic cache timer, **stores app info cache**, and saves workspace settings. It does **not** block until pool tasks finish—if the user confirms exit, the process can still shut down while work is running. **Force quit** skips even the confirmation and cache flush.

**Implications:** Partial writes remain possible (e.g. half-copied dataset, incomplete checkpoint).

**Subprocess training (implemented):** The Train page can start **`mb train --train-args-json …`** in a **detached OS process** (checkbox: run in a separate process). Logs go to **`<output_dir>/mb_train_subprocess.log`**. The GUI still shares the process for the default in-thread path; only the detached option isolates training from the app process.

**Relation to cancellation:** Cooperative cancellation (§7.3) uses the **same mental model**—early stop may leave partial artifacts.

### 7.2 Progress in the UI (not log dumps)

The backend will continue to emit rich diagnostics through Python **`logging`** (including file handlers configured in `utils.logging_setup`). That stream is appropriate for **debugging and support**, not for mirroring every log line in the main window.

**Implemented:** **`LongTaskContext.progress`** / **`TaskSignals.progress`**; **`attach_progress_dialog`** (`ui/lib/task_progress.py`) shows a modal **`QProgressDialog`**. **`ModelTrainer.train`** uses **`progress_cb`**; PyTorch/Keras report epoch fractions. Per-page **`QTextEdit`** for a short run log. **Reveal log folder** on failures: **`qt_operation_error`** (§7.6).

**Optional later:** A main-window status strip or non-modal progress; an explicit **Open log file** control alongside **Reveal log folder**.

### 7.3 Cancellation

**Training:** Cooperative cancel at **batch** boundaries during PyTorch train/validation loops, and via **Keras** `LambdaCallback` **`on_batch_begin`** during `fit` (training batches). Epoch-start checks remain where applicable. **Data pipeline** (`ImageGatherer`, `ImageConverter`, `ImageDeduplicator`, `ImageUpscaler`, `DatasetCreator`): **`run(cancel_event=…)`** checks between phases and every N files in long loops. **Model convert** (`convert_model`): check **before** export (single-shot; cannot interrupt mid-export cleanly without deeper hooks).

**Not done / future:** Interrupting a single **ONNX/export** call mid-flight; optional dedicated **`QThread` per job**. Prefer polite stop over force-killing threads.

**Partial-state:** Same as §7.1—cancel may leave partial artifacts; Train / Data / Convert pages append a short “stopped” line.

### 7.4 Threading and Qt (main-thread bridge)

Qt **GUI objects must be used only on the thread that created them** (typically the main / GUI thread). Worker threads must **not** call `QLabel.setText`, append to `QTextEdit`, open dialogs, etc. directly.

**In this repo:** `ui/task_runner.py` constructs **`TaskSignals` on the main thread** (when `start_task` runs) and connects **`success` / `error` / `done` with `Qt.QueuedConnection`**, so handlers that touch widgets run on the GUI thread. **`NotificationController.toast`** and **`title_notify`** emit Qt signals so slots run on the GUI thread.

**Ad-hoc / non-pool callers:** **`utils.notification_manager`** drives **`AppActions.title`** (and sometimes **`toast`**) from **`threading.Timer`** callbacks—not the Qt main thread. **`MainWindow`** therefore creates **`MainThreadBridge`** (`ui/main_thread_bridge.py`), wraps **`setWindowTitle`** and **`NotificationController.alert`** for **`AppActions`**, and registers **`notification_manager.set_app_actions(self.app_actions, …)`** so those paths always execute Qt work on the GUI thread. **`invoke` / `wrap`** use **`QMetaObject.invokeMethod`** with **`BlockingQueuedConnection`** (direct call when already on the GUI thread to avoid deadlock).

**Why it matters once progress and cancel exist:** Queued signal delivery keeps pool task UI updates deterministic; timers and any future worker code that bypasses `TaskSignals` still need **`MainThreadBridge`** or an equivalent.

#### `MainWindow._build_app_actions` (Model Builder)

```python
def _build_app_actions(self) -> AppActions:
    nc = self._notifications
    ts = self._thread_bridge.wrap

    return AppActions(
        {
            "get_window": lambda: self,
            "toast": nc.toast,
            "_alert": ts(nc.alert),
            "title_notify": nc.title_notify,
            "refresh": lambda: None,
            "title": ts(self.setWindowTitle),
        },
        master=self,
    )
```

The full **`MainThreadBridge`** implementation (including **`invoke`**, **`wrap`**, and the **`@Slot`** **`_execute`** slot) lives in **`ui/main_thread_bridge.py`**.

### 7.5 Config: application vs pipeline (two YAML concerns)

Settings are split intentionally:

| Layer | Python API | Typical YAML |
|--------|------------|----------------|
| **Application shell** (gui, app — toasts, window size, debug) | `utils.config.get_application_config` / `reload_application_config` | Repo `configs/application.yaml`, or legacy `configs/default.yaml` (only `gui` / `app` keys are read) |
| **Pipeline / ML jobs** (model, data, training, paths) | `mb.pipeline_config.get_pipeline_config` / `reload_pipeline_config` | `mb/config/default_pipeline.yaml` (package), or workspace `configs/pipeline.yaml`, or the same file as app if it is a legacy combined `default.yaml` |

**GUI (`MainWindow`):** calls **`reload_application_config(_effective_application_config_path())`** and **`reload_pipeline_config(_effective_pipeline_config_path())`** on startup, after **Set workspace folder**, and after **Set config file**. Application path: explicit file from the menu, else `workspace/configs/application.yaml`, else `workspace/configs/default.yaml`. Pipeline path: `workspace/configs/pipeline.yaml` if present, else the menu path (filters pipeline keys), else `workspace/configs/default.yaml`, else packaged defaults.

**CLI (`mb train`):** uses **`reload_pipeline_config(args.config)`** and **`get_pipeline_config()`** only — no application shell YAML unless you load it elsewhere. **`mb train --train-args-json PATH`** loads **`TrainingRunArgs`** from JSON (see **`mb.training.run_args`**) and ignores other train flags; use **`mb --config … train --train-args-json …`** for pipeline YAML.

**Aliases:** `get_config` / `reload_config` in `utils.config` still map to the **application** singleton for backward compatibility.

**Caveat:** Import-time `get_application_config()` before `MainWindow` can pin early defaults; the GUI loads both layers before pages are built.

### 7.6 Error handling in the UI

**Implemented:** **`qt_operation_error`** and **`qt_alert`** in `ui/lib/qt_alert.py`—critical / warning / info / yes-no flows with translated buttons where applicable. Train, Convert, and Data task failures use **`qt_operation_error`**; Info, Data (pre-flight checks), and MainWindow use **`qt_alert`**. **`ui/task_runner.py`** logs the **full traceback** with `logger.exception` on worker failure (the dialog still shows `str(exc)` to avoid dumping stacks in the UI).

**Operation errors (modal):** **`qt_operation_error`** is the single place for “backend task failed” UX: critical icon, short summary, optional **Show Details** (`setDetailedText`), plus **Copy details** (summary + detail to the clipboard) and **Open log folder** (opens the same directory **`get_log_directory()`** in `utils/logging_setup.py` uses for `modelbuilder_*.log`). Pass **`with_log_actions=False`** if a caller must show a minimal dialog without those actions.

**“Central error controller”:** There is no separate controller class by design—**`qt_operation_error`** plus **`task_runner`** logging are the coordinated surface. A future **`MainWindow`** error signal would only be needed if multiple subsystems must subscribe to the same failure event without a dialog.

**Non-blocking feedback:** Use **`NotificationController`** / **`notification_manager`** toasts for supplementary status (cache, workspace, etc.). Full task failures stay **modal** so they are not missed; optional non-blocking **error** banners remain a future enhancement if product wants failures visible without blocking focus.

(Config path wiring for the GUI is covered in §7.5.)




### Follow-up items (tracked in the implementation plan)

Optional **product / parity** work called out above (§4, §7.2, §7.6) is **not** a bug list. Roadmap rows and status live in **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — section **“GUI / CLI parity and UX follow-ups”** (together with Phase 5 testing notes).

**Implemented from earlier review notes:** detached `mb train --train-args-json`, cooperative cancel on Data / Convert / gather / dataset / train (batch-level checks where applicable), integration and UI tests under `tests/` (see **[ARCHITECTURE.md](ARCHITECTURE.md)** §9).

---

**Document version:** 1.11  
**Last updated:** 2026-04-02
