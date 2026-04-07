# Pipeline directory layout and snapshot files

This document walks through the **image classification** data pipeline using the **default pipeline configuration** in `mb/config/default_pipeline.yaml`. Paths below are shown **relative to your project / workspace root** (the directory you use as the current working directory for CLI commands, or where your app resolves `raw_data` and `data`).

**Default pipeline path keys (relevant here):**

| Key | Default value |
|-----|----------------|
| `data.raw_data_dir` | `raw_data` |
| `data.data_dir` | `data` |
| `paths.models_dir` | `data/models` |
| `paths.logs_dir` | `logs` |
| `data.gather.default_target_dir` | `coherent` (resolved under `raw_data`) |
| `data.gather.default_rejected_dir` | `rejected` (resolved under `raw_data`) |

**Unified snapshot JSON:** the same logical snapshot (same `run_id`) is updated across stages. It is **written to disk** in different roots at different steps—see each phase below.

---

## Phase 0 — Before the pipeline

You might start with only external sources, or an empty tree. For conversion you eventually need **class folders** under `raw_data`, each containing media in subfolders (for example `IMAGES/` or loose subdirs—see `mb data convert` scanning rules).

Illustrative starting point (three classes—same labels as in the synthetic test fixtures—originals under `IMAGES/`):

```text
raw_data/
  coherent/
    IMAGES/
      photo1.jpg
  semi-incoherent/
    IMAGES/
      photo2.png
  incoherent/
    IMAGES/
      photo3.jpg
data/                    # may not exist yet
logs/                    # optional; training/logging may create
```

No `snapshot_*.json` yet.

---

## Phase 1 — Gather (`mb data gather`) — optional

Gather copies from a **source** tree into a **target** under `raw_data`, with a **new run-scoped folder** each time.

**Defaults (from `gather_pipeline_defaults()`):**

- Target root: `raw_data/coherent/`
- Per run: `raw_data/coherent/<YYYYMMDD_HHMMSS>/` (timestamped)
- Rejected (optional manual staging): `raw_data/rejected/`
- Hash cache: `raw_data/.gather_cache.pkl`

**No unified snapshot** is produced by gather alone.

Example after one gather run (the first `coherent` segment is **`data.gather.default_target_dir`**, i.e. the configured gather bucket under `raw_data`, not a renaming of the class list):

```text
raw_data/
  coherent/                    # gather target root (default_target_dir)
    20260115_143022/           # one folder per gather run
      coherent/
        …
      semi-incoherent/
        …
      incoherent/
        …
  rejected/                    # if used
  .gather_cache.pkl
```

Your class layout may differ; convert expects discoverable class buckets under `raw_data` consistent with `data.class_names` / `data.class_qualifying_subdir`.

---

## Phase 2 — Convert (`mb data convert`)

**Inputs:** walks each **class directory** under `raw_data`, finds media under non-reserved subfolders (skips `CONVERTED`, `JPEG_IMAGES`, `visual_media_review` as input roots), and writes normalized JPEGs.

**Adds per class:**

- `raw_data/<class>/CONVERTED/` — normalized `.jpg` outputs (and copies/re-encodes of existing JPEGs).
- `raw_data/<class>/visual_media_review/` — only if there are **videos or animated GIFs** (extracted frame JPEGs duplicated here for review).

**Unified snapshot (first write):**

- File: **`raw_data/snapshot_<run_id>.json`**
- `run_id` is generated at the start of convert (see `generate_run_id()` in `mb/utils/snapshot.py`).

Example:

```text
raw_data/
  coherent/
    IMAGES/
      photo1.jpg
    CONVERTED/
      photo1.jpg
  semi-incoherent/
    IMAGES/
      photo2.png
    CONVERTED/
      photo2.jpg
  incoherent/
    IMAGES/
      photo3.jpg
    CONVERTED/
      photo3.jpg
  snapshot_20260115_143045_a1b2c3d4.json
```

Optional: `raw_data/.mb_space_estimate.json` may appear for convert space-estimate caching.

---

## Phase 3 — Create dataset (`mb data create-dataset`)

**Inputs:** JPEGs under each class’s resolved media directory (often `CONVERTED/` when using the usual layout).

**Adds under `data/` (default `data.data_dir`):**

- `data/train/<class>/*.jpg` — hashed filenames
- `data/test/<class>/*.jpg` — holdout split
- `data/invalid_size_review/` — images moved aside for byte-size or geometry rules (see `mb/data/dataset.py`)

**Unified snapshot (update + second save location):**

- Loads the snapshot for the same pipeline `run_id` (searches under `raw_data` / parents as implemented).
- Updates dataset-related fields in memory.
- Saves again as **`data/snapshot_<run_id>.json`** (same filename pattern, same `run_id`).

So you typically have the snapshot **both** under `raw_data/` (from convert) **and** under `data/` (authoritative copy after dataset creation), unless you delete one manually.

Example:

```text
data/
  train/
    coherent/
      <sha256>.jpg
    semi-incoherent/
      <sha256>.jpg
    incoherent/
      <sha256>.jpg
  test/
    coherent/
      <sha256>.jpg
    semi-incoherent/
      <sha256>.jpg
    incoherent/
      <sha256>.jpg
  invalid_size_review/       # only if any images were moved here
  snapshot_20260115_143045_a1b2c3d4.json
```

---

## Phase 4 — Train (`mb train` / GUI Train)

**Inputs:** `data/train/` and `data/test/` (ImageFolder-style).

**Default outputs:**

- **`data/models/`** — saved model file (e.g. `resnet34_model.pth` or `.h5` for Keras), relative to workspace; exact name depends on architecture and framework.

**Logs:** application/training logging may use `paths.logs_dir` (`logs/`) depending on setup; not a full duplicate of this doc.

**Unified snapshot (optional):**

- If **`update_snapshot`** is enabled and a snapshot is found, training updates per-image `training` metadata and sets **`training_timing`** (wall-clock train/eval seconds), then saves:

  **`data/snapshot_<run_id>.json`**

  (same path pattern as create-dataset; overwrites/updates that file).

Example additions:

```text
data/
  models/
    resnet34_model.pth       # example; name follows architecture
  snapshot_20260115_143045_a1b2c3d4.json   # updated with training_timing + training fields

logs/
  …                          # if your logging config writes here
```

---

## Snapshot file summary

| After step | Typical snapshot path | Notes |
|------------|------------------------|--------|
| Convert | `raw_data/snapshot_<run_id>.json` | First persistence of the unified snapshot. |
| Create-dataset | `data/snapshot_<run_id>.json` | Same `run_id`, extended with dataset (and later training) data. |
| Train (with `update_snapshot`) | `data/snapshot_<run_id>.json` | Adds `training_timing` and training-stage image links when applicable. |

Downstream code (space estimates, training) often searches **`data/`** and **`data/..`** for `snapshot_*.json` and picks the latest or a specific `run_id`—see `find_unified_snapshot` / `find_latest_unified_snapshot_path` in `mb/utils/snapshot.py`.

---

## Paths not covered here

- **Workspace-only** GUI settings (`application.yaml`) live outside this tree.
- **App data** pipeline copies (`%AppData%/ModelBuilder/pipeline.yaml`, etc.) follow the same **logical** keys (`raw_data`, `data`, …) but root on disk wherever you configure the workspace.

For command-specific flags (absolute `raw_data`, custom `--run-id`, etc.), refer to `mb data --help` and `mb train --help`.
