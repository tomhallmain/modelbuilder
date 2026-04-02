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
| **Process coupling** | Closing the app or a crash can stop a long job mid-run | Same practical outcome as terminating a CLI process: the worker does not complete; partial outputs may exist on disk. **Cancellation** (below) should follow the same lifecycle model—cooperative stop on the worker, not a second OS process—so “user cancelled” and “app exited” behave similarly from the job’s perspective. |
| **Progress in the UI (not log dumps)** | — | ~~Expecting the main window to mirror full `logging` output would overwhelm users.~~ The product direction is **not** to stream raw backend logs into the primary UI. **`mb` modules keep using `logging`** (e.g. file handlers in `mb.utils.logging`) for detailed traces. In the GUI, **progress** should be handled deliberately: e.g. a **unified progress callback** (phase, percent, or indeterminate state) feeding a **single progress widget or dialog** for long operations. Optional “View log file…” can link to disk logs for power users. |
| **Cancellation** | Long jobs cannot be stopped cleanly today | **Required:** a first-class **Cancel** that signals the running work to stop cooperatively (shared flag, `QThread.requestInterruption()`, or equivalent). Semantics align with **process coupling** above: cancelled jobs should leave the same class of partial state as an abrupt exit, but under user control. |
| **Threading + Qt (main-thread bridge)** | Workers must not touch QWidget APIs directly | **Required:** any callback from a worker that updates UI (progress, errors, done) should go through a **main-thread bridge**—the same pattern as `_MainThreadBridge` in other PySide6 codebases: wrap GUI-facing callables so they always execute on the GUI thread via Qt signals/slots (or `QMetaObject.invokeMethod` with `Qt.QueuedConnection`). Raw `QThreadPool` + fire-and-forget handlers are insufficient once progress and cancel are wired everywhere. |
| **Config** | Training/conversion do not yet use the workspace YAML path | Today the GUI may call `get_config(None)` while the File menu stores a path separately. **Fix is small:** thread the workspace/config path from `ui/workspace` (or the menu) into `get_config(path)` for train/convert and any other command that respects CLI `--config`. Same defaults as CLI once that path is passed. |
| **Error handling in the UI** | Failures must be visible without relying on log files alone | **Beyond** ad-hoc `QMessageBox` on exceptions: structured **error presentation** in the UI—e.g. non-blocking banner or dialog with a short user-facing message, optional “Details” (traceback / logger last line), and a consistent path for “copy error” or “open log folder”. Backend exceptions stay the source of truth; the UI layer maps them to recoverable, testable surfaces (signals to a small error controller, not scattered `print`/`messageBox` in each page). |

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
| **Training config file** | `--config PATH` | Workspace / File → config **not** automatically applied to `ModelTrainer` unless you add wiring |
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

The following subsections mirror the **Pitfalls (general)** table in §1 and spell out intent, constraints, and implementation direction. They are **design notes** for the `ui/` package, not a commitment that every item is already implemented.

### 7.1 Process coupling and job lifecycle

The GUI and the `mb` work run in the **same OS process**. If the user closes the window, the OS kills the process, or an unhandled fault occurs, any in-flight operation (gather, dataset creation, training, conversion) stops **without a graceful backend “shutdown” hook** unless one is added explicitly.

**Implications:** Partial writes are possible (e.g. half-copied dataset, incomplete checkpoint). That is the same class of risk as stopping `mb` in a terminal with Ctrl+C or killing the shell. Documentation and UX should set expectations: **long jobs are not isolated processes** unless you later introduce a separate worker executable or subprocess.

**Relation to cancellation:** Cooperative cancellation (§7.3) should use the **same mental model**—the job ends early and may leave partial artifacts—so users understand “Cancel” vs “crash” vs “success” similarly at the filesystem level.

### 7.2 Progress in the UI (not log dumps)

The backend will continue to emit rich diagnostics through Python **`logging`** (including file handlers configured in `mb.utils.logging`). That stream is appropriate for **debugging and support**, not for pasting line-by-line into the main window.

**Intended GUI behavior:** Surface **progress**, not a verbatim log tail:

- A **unified progress contract** (e.g. callback or signal emitting phase name, optional numeric percent, and indeterminate vs determinate mode) invoked from the worker side at safe points.
- A **single** primary surface for long operations: e.g. a **modal or modeless progress dialog**, or a dedicated strip in the main window, so the user always knows *something is running* and *how far along* it is in product terms (“Gathering…”, “Epoch 3/10”), not raw logger lines.
- Optional **“Open log file…”** or **“Reveal log folder”** for users who need the full trace—keeping the default experience calm.

This avoids overwhelming non-expert users while preserving observability on disk.

### 7.3 Cancellation

Long-running work must be **stoppable without killing the whole application** (when safe). Today, background tasks may run until completion; **cancellation** is a required enhancement.

**Design goals:**

- **Cooperative cancellation:** The worker checks a shared **cancel flag** or `QThread.isInterruptionRequested()` (if the work runs on a `QThread`) at loop boundaries—between files, between epochs, between pipeline stages—not mid-syscall where unsafe.
- **Same partial-state story as §7.1:** After cancel, the user may see incomplete outputs; the UI should say so (“Stopped by user—check output folder before re-running”).
- **Implementation options:** A dedicated `QThread` per job (easier interruption) vs `QThreadPool` + periodic flag checks—either is acceptable if cancellation is tested on real workloads.

Force-killing threads from outside is discouraged; prefer **polite** stop requests.

### 7.4 Threading and Qt (main-thread bridge)

Qt **GUI objects must be used only on the thread that created them** (typically the main / GUI thread). Worker threads must **not** call `QLabel.setText`, append to `QTextEdit`, open dialogs, etc. directly.

**Required pattern:** A **main-thread bridge**—a small helper that forwards calls to the GUI thread via Qt’s event loop, e.g.:

- **Signals** emitted from the worker and **connected** to slots on a QObject living in the main thread (Qt will use **queued** delivery across threads when the connection type allows), or
- **`QMetaObject.invokeMethod(..., Qt.QueuedConnection)`** on a GUI-side QObject.

This matches the idea of classes like `_MainThreadBridge` in other PySide6 projects: wrap any “touch the UI” callable so workers only invoke **thread-safe** entry points. Progress updates, error toasts, and “job finished” should all go through this layer.

**Why it matters once progress and cancel exist:** Without a bridge, cross-thread UI bugs are intermittent and hard to reproduce; with one, behavior stays deterministic.

#### `_MainThreadBridge` class:

```python
class _MainThreadBridge(QWidget):
    """Marshals arbitrary callables from worker threads to the main/GUI thread.

    Uses ``QMetaObject.invokeMethod`` with ``BlockingQueuedConnection`` so that
    the calling (worker) thread blocks until the callable finishes on the main
    thread.  When already on the main thread the callable runs directly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()  # invisible helper widget
        self._lock = threading.Lock()
        self._func = None
        self._args = ()
        self._kwargs = {}
        self._result = None
        self._error = None

    @Slot()
    def _execute(self):
        try:
            self._result = self._func(*self._args, **self._kwargs)
        except Exception as e:
            self._error = e

    def invoke(self, func, *args, **kwargs):
        """Call *func* on the main thread, blocking until it returns."""
        app = QApplication.instance()
        if app is None or QThread.currentThread() == app.thread():
            return func(*args, **kwargs)
        with self._lock:
            self._func = func
            self._args = args
            self._kwargs = kwargs
            self._result = None
            self._error = None
            QMetaObject.invokeMethod(
                self, "_execute", Qt.ConnectionType.BlockingQueuedConnection,
            )
            if self._error:
                raise self._error
            return self._result

    def wrap(self, func):
        """Return a wrapper that always invokes *func* on the main thread."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.invoke(func, *args, **kwargs)
        return wrapper
```

#### Example Usage

```python
    def _build_app_actions(self) -> AppActions:
        """Wire the AppActions dict, mapping action names to controller methods.

        Actions that touch the Qt GUI are wrapped via :class:`_MainThreadBridge`
        so that the compare engine (which runs on a worker thread) can call them
        safely.  Pure-data / thread-safe getters are left unwrapped.
        """
        ts = self._thread_bridge.wrap  # shorthand

        actions = {
            # Window title -- thread-safe via Signal so that
            # notification_manager's Timer thread never touches Qt directly
            "title": self._sig_set_title.emit,
            # Window management (static)
            "new_window": ts(WindowManager.add_secondary_window),
            "get_window": WindowManager.get_window,
            "get_open_windows": WindowManager.get_open_windows,
            "refresh_all_compares": ts(WindowManager.refresh_all_compares),
            # etc.
        }
        self.app_actions = actions
```

### 7.5 Config path wiring (workspace YAML)

The File / workspace flow can persist a **YAML config path** (`ui/workspace`, `QSettings`). The **CLI** uses `get_config(args.config)` so `--config` applies globally to training and other steps.

**Gap:** Pages that call `get_config(None)` ignore that stored path unless it is passed through.

**Intended fix (small):**

- When launching train, convert, or any action that respects `mb`’s config merge, pass **`Path`** from the current workspace into **`get_config(config_path)`** (or equivalent), falling back to `None` when unset.
- Keep a single source of truth: either the workspace owns the path, or the main window injects it into page controllers once per session.

After wiring, GUI behavior should match **`mb --config …`** for the same file.

### 7.6 Error handling in the UI

Relying only on **ad-hoc `QMessageBox` in each page** makes errors inconsistent and hard to test; relying only on **log files** fails users who never open them.

**Intended direction:**

- A small **error surface** (object or module) that pages and workers call with: short **user message**, optional **detail** (exception string, last log line), and **severity** (info / warning / error).
- **Presentation:** Non-blocking **banner** or **dialog** with optional **“Details”** expander, plus actions like **Copy error** and **Open log folder** where applicable.
- **Implementation:** Prefer **signals** from workers → main-thread slots that call this surface, not `QMessageBox` from inside `run()` on a pool thread.

Backend exceptions remain authoritative; the UI layer **maps** them to consistent, accessible feedback.

---

**Document version:** 1.2  
**Last updated:** 2026-04-02
