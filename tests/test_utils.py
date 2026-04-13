"""
Shared helpers for integration and E2E tests (snapshots, synthetic trees, paths).
"""

from __future__ import annotations

from pathlib import Path

from mb.utils.snapshot import UnifiedSnapshot, save_unified_snapshot, generate_run_id

from tests.fixtures.synthetic_dataset import build_synthetic_raw_data_dir


def repo_root() -> Path:
    """Repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parent.parent


def default_pipeline_config_path() -> Path:
    """Bundled pipeline YAML used by ``mb --config`` in tests."""
    return repo_root() / "mb" / "config" / "default_pipeline.yaml"


def write_minimal_unified_snapshot(
    raw_data_dir: Path,
    run_id: str | None = None,
) -> str:
    """
    Write an empty (no image rows) unified snapshot JSON under *raw_data_dir*.

    :class:`mb.data.dataset.DatasetCreator` refuses to run without
    ``snapshot_*.json``; the creator then fills ``images`` while copying from
    ``CONVERTED/``. A minimal file satisfies the loader.
    """
    rid = run_id or generate_run_id()
    snap = UnifiedSnapshot(run_id=rid, raw_data_dir=str(raw_data_dir.resolve()))
    path = save_unified_snapshot(snap, raw_data_dir, logger=None)
    if path is None:
        raise RuntimeError(f"Failed to write unified snapshot under {raw_data_dir}")
    return rid


def prepare_synthetic_raw_with_snapshot(
    workspace: Path,
    *,
    total_images: int = 100,
    image_seed: int = 42,
    snapshot_run_id: str = "e2e_minimal_snapshot",
    per_class_counts: dict[str, int] | None = None,
) -> Path:
    """
    Build ``<class>/CONVERTED/*.jpg`` under ``workspace/raw_data`` and write
    ``snapshot_<snapshot_run_id>.json`` beside it.

    Returns:
        Path to ``raw_data`` (pass as ``raw_data_dir`` to :class:`mb.data.dataset.DatasetCreator`).
    """
    raw = workspace / "raw_data"
    build_synthetic_raw_data_dir(
        raw,
        total_images=total_images,
        seed=image_seed,
        per_class_counts=per_class_counts,
    )
    write_minimal_unified_snapshot(raw, run_id=snapshot_run_id)
    return raw
