# `mb` package — legacy and follow-up inventory

This document lists **leftover patterns** from the script-era pipeline (standalone `.py` tools, three fixed class folders, duplicate constants) so they can be triaged against current priorities. It is **not** a commitment to change any item.

---

## 1. CLI stubs

| Location | Notes |
|----------|--------|
| *(resolved)* | Model/dataset inspection is implemented in `mb/info_inspect.py` (`model_info_text`, `dataset_info_text`). `mb info model|dataset` and `ui/pages/info_page.py` call the same helpers. |

---

## 2. Standalone `if __name__ == "__main__"` entry points

Data modules expose ``python -m mb.data.<module>`` as a thin wrapper around the unified CLI:

| Module | Delegation |
|--------|------------|
| `mb/data/gather.py` | `run_data_subcommand_cli("gather")` |
| `mb/data/convert.py` | `run_data_subcommand_cli("convert")` |
| `mb/data/dataset.py` | `run_data_subcommand_cli("create-dataset")` |
| `mb/data/deduplicate.py` | `run_data_subcommand_cli("deduplicate")` |
| `mb/data/upscale.py` | `run_data_subcommand_cli("upscale")` |
| `mb/cli.py` | Canonical `mb` CLI (`run_data_subcommand_cli` lives here) |

### 2a. Parity: `mb data <sub>` vs `-m` (resolved)

- **Flags and behavior** for gather/convert/dedup/upscale/create-dataset are defined in `mb/cli.py`; per-module argparse `main()` blocks were removed so there is a single source of truth.
- **Longer help strings** from the old module parsers (e.g. `--subdir-weights`, dedup/upscale/raw-data context) were folded into the centralized parsers where needed.
- **Convert `--format`:** still not wired into `ImageConverter` in the handler—see TODO in `handle_data_convert`.
- **create-dataset:** canonical flag is `--test-per-class`; there is **no** legacy `--test-images-per-class` alias. Library API and pipeline default key use **`test_per_class`** (replacing `test_images_per_class` / `TEST_IMAGES_PER_CLASS`).

---

## 3. Module docstrings still say “Script to…”

**Status:** `mb/data/dataset.py`, `deduplicate.py`, `upscale.py`, `convert.py`, and `gather.py` module docstrings now describe **library + CLI** roles and pipeline order. If you find a stray “Script to…” elsewhere, treat it as cleanup.

---

## 4. Duplicate `IMAGE_EXTENSIONS` (and drift from pipeline YAML)

The same extension set is **redefined** in multiple places:

- `mb/data/gather.py`, `convert.py`, `deduplicate.py`, `upscale.py`
- `mb/utils/snapshot.py` (also imported by `mb/training/snapshot_integration.py`)

Pipeline config already has `data.image_types` (and `video_types`), but gather/convert paths **do not** read it today; sets are similar but not guaranteed identical (e.g. vs `.heic` / `.avif` in YAML).

**Follow-up:** Centralize (e.g. `mb.data.file_types` or read from `get_pipeline_config()`), with a clear rule for **pre-convert** vs **post-convert** scanning.

---

## 5. Hardcoded three-way “coherent / incoherent / semi-incoherent” taxonomy

**Status:** Raw staging buckets for dedup / gather’s optional dedup pass / small-image review upscaling use **`data.class_names` or `discover_class_names`** via :func:`mb.data.class_layout.discover_raw_data_bucket_names` and :func:`~mb.data.class_layout.discover_review_bucket_names` (excluding ``rejected``, ``small_images_review``, ``upscaled_small_images``). Cross-directory duplicate reporting is a single pass across **all** discovered buckets (no special “primary vs others” step).

**Tests / fixtures** may still use example folder names (e.g. three-way split); that is intentional.

---

## 6. Gather naming and defaults

- Gather logging uses script name **`gather_images`**.
- Example default gather target in YAML may still show a concrete path (e.g. under ``raw_data/``); override per project in ``data.gather``.

---

## 7. Convert output folder `JPEG_IMAGES`

`mb/data/convert.py` always writes converted JPEGs under **`class_dir/JPEG_IMAGES/`** (`JPEG_IMAGES_DIR`). That matches the legacy layout and pairs with `class_layout.resolve_class_media_dir` (which prefers `JPEG_IMAGES` then `IMAGES` when no qualifier is set).

**Follow-up:** Optionally make the **output** subdir configurable (parallel to `class_qualifying_subdir` for inputs), if you need convert output to land in `IMAGES` instead of `JPEG_IMAGES`.

---

## 8. User-facing strings referencing old script filenames

Example: `mb/data/dataset.py` error text still says to run **`convert_to_jpeg.py`** first.

**Follow-up:** Point users at **`mb data convert`** (and/or `python -m mb.data.convert`) for consistency.

---

## 9. Comments referencing removed or external scripts

| Location | Text |
|----------|------|
| `mb/data/gather.py` | Dedup “handled by separate script (`deduplicate_images.py`)” |
| `mb/data/deduplicate.py` | Cache shared with `gather_coherent_images.py` |
| `mb/utils/snapshot.py` | Preload cache from `gather_coherent_images.py` |
| `mb/training/snapshot_integration.py` | “original train_model.py script” |

**Follow-up:** Rephrase to **module or CLI** names (`mb.data.gather`, `mb data deduplicate`, etc.).

---

## 10. Training / snapshot wording

- `snapshot_integration.py` docstring references **train_model.py** — historical only.

**Follow-up:** One sentence pointing to `mb.training` / `ModelTrainer` as the source of truth.

---

## 11. Already improved (context)

Recent work reduced some legacy surface area:

- **Class lists** — `data.class_names`, `data.class_qualifying_subdir`, `mb.data.class_layout`
- **Gather defaults** — `data.gather` in pipeline YAML; `gather_pipeline_defaults()`
- **Synthetic tests** — `SYNTHETIC_DEFAULT_CLASS_NAMES` instead of a hardcoded export from `dataset.py`

---

## Suggested prioritization (non-binding)

1. **Low risk:** Docstring / comment / error-message cleanup (sections 3, 8, 9, 10).  
2. **Medium:** CLI `mb info` parity with GUI (section 1).  
3. **Larger refactors:** Dedup/gather/upscale three-folder assumption (section 5); unified `IMAGE_EXTENSIONS` (section 4); optional convert output dir (section 7).

If you add new items, keep them as **bullet + file path + one-line action** so this file stays scannable.
