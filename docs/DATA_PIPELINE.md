# Data pipeline (image classification)

This document describes how the **`mb data`** steps fit together: what order to run, when steps are optional, how storage layout works, and how **very large** vs **very small** images are treated.

## Default step order

The usual pipeline is:

1. **`mb data gather`** — copy samples from external source trees into class folders under your raw-data root (e.g. `raw_data/<class>/…`).
2. **`mb data convert`** — normalize inputs to JPEG under each class’s `CONVERTED/` folder (and `visual_media_review/` for random frames from video / animated GIF when the model type is image classification). Writes a unified snapshot for later steps.
3. **`mb data deduplicate`** *(optional but common)* — duplicate removal; also handles **dimension-based** quarantine of tiny images (see below).
4. **`mb data upscale`** *(optional)* — upscale images staged from the small-image review area after you are satisfied with what to keep.
5. **`mb data create-dataset`** — build `train/` and `test/` under your chosen **output data directory** from the raw tree (using converted media where applicable).

Documented defaults also appear in `mb/pipeline_config` (module docstring) and pipeline YAML under `data.*`.

## Skipping gather

**Yes — you can skip gather** if your data is **already** laid out as **one folder per class** under your raw-data root (the same structure gather would have produced). Point **`--raw-data-dir`** (and the Data page “Raw data dir” fields) at that root, whether it lives on an internal or external drive.

Gather is only for **pulling** files from separate labeled source directories into that staging tree. If your buckets are already correct, start at **convert**.

## Order when gather is skipped

1. **`convert`**
2. **`deduplicate`** (optional)
3. **`upscale`** (optional; only if you use the small-image review flow)
4. **`create-dataset`**

## Storage: external drives and where to put data

- **Convert** reads originals and writes **`CONVERTED/`** (and related review subfolders) **next to them** under the raw-data root. Heavy duplication (originals plus JPEGs) therefore lands on **whatever volume holds that tree**, unless you use copies/symlinks elsewhere.
- **Create-dataset** writes the **training-ready** tree to **`--data-dir`** (`train/` / `test/`). That is often a good place to use your **main (large) internal disk** while keeping raw + convert on slower or fuller external storage—at the cost of slower I/O during those steps.

There is no single enforced “move” step in the tools; it is an operational choice:

- To **minimize internal disk use**, keep raw + convert on the external drive and set **create-dataset’s output** to an internal path when you want the final dataset off the removable disk.
- To **maximize speed** (especially over USB), copy or sync the raw tree to an internal drive **before** convert, or run convert internally then point create-dataset at that tree.

**Rule of thumb:** the **normalized JPEGs** are fixed in layout at **convert**; the **train/test split** is materialized at **create-dataset** into `data_dir`. Moving data “onto the main computer” for long-term storage is most natural **when you choose `data_dir` on the internal drive**, or **after** convert if you manually relocate only `CONVERTED/` (and any review dirs you care about) before create-dataset.

## Very large vs very small images

These are handled differently:

### Large images (high resolution / large pixel count)

- **Convert** raises Pillow’s decompression limit and **downscales** so the **longest edge is at most 4000px** (same idea for extracted video/GIF frames in `mb.data.media_utils`).
- **Create-dataset** applies additional **per-file size** limits when building the split (see `mb/data/dataset.py`).

So “very large” inputs are normalized for the training pipeline without using the same **review folder** path as tiny-by-dimension files.

### Very small images (small width/height)

- **`deduplicate`** (not convert) applies rules such as: remove images whose minimum dimension is **below ~80px**, and move **~80–250px** images into a **review directory** under the raw root (by default `small_images_review/`), mirroring class structure where applicable.
- **`upscale`** reads from that review tree when you want to generate upscaled outputs.

So small-dimension assets are **quarantined and optionally upscaled** via **deduplicate → review → upscale**, which is separate from the **large-image** path in convert.

## Related UI

The **Data** page in the desktop app mirrors these commands; hover tooltips on tabs and groups summarize behavior and order.
