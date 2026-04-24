"""
Persist recent GUI job runs to :mod:`utils.app_info_cache` for the Home page.

Each entry may include optional ``snapshot_path`` (newest unified ``snapshot_*.json``
after convert / create-dataset / training with snapshot updates) for the Home page
history display.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, List, Optional, Union

from mb.utils.constants import DataPipelineSubcommand, ModelBuilderTaskType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger("recent_run_history")

RunHistoryKindArg = Union[ModelBuilderTaskType, str]
DataSubcommandArg = Union[DataPipelineSubcommand, str, None]

RECENT_RUN_HISTORY_KEY = "recent_run_history"
MAX_RECENT_RUNS = 50


def _storage_kind(kind: RunHistoryKindArg) -> str:
    if isinstance(kind, ModelBuilderTaskType):
        return kind.value
    return (kind or "?").strip()[:32] or "?"


def _storage_data_subcommand(sub: DataSubcommandArg) -> Optional[str]:
    if sub is None:
        return None
    if isinstance(sub, DataPipelineSubcommand):
        return sub.value
    t = str(sub).strip()[:64]
    return t or None


def append_recent_run(
    kind: RunHistoryKindArg,
    summary: str,
    ok: bool,
    detail: str = "",
    *,
    data_subcommand: DataSubcommandArg = None,
    snapshot_path: Optional[str] = None,
) -> None:
    """
    Prepend a run record and trim to :data:`MAX_RECENT_RUNS`.

    Persists via :meth:`~utils.app_info_cache.AppInfoCache.store` on a short-lived
    background thread (same pattern as periodic GUI cache flush).
    """
    from utils.app_info_cache import app_info_cache

    kind_str = _storage_kind(kind)
    summary = (summary or "").strip()[:500]
    detail = (detail or "").strip()[:2000]
    sub_str = _storage_data_subcommand(data_subcommand)

    raw = app_info_cache.get(RECENT_RUN_HISTORY_KEY, default_val=None)
    items: List[Any] = list(raw) if isinstance(raw, list) else []

    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_type": kind_str,
        "kind": kind_str,
        "summary": summary,
        "ok": bool(ok),
        "detail": detail,
    }
    if sub_str is not None:
        entry["data_subcommand"] = sub_str
    sp = (snapshot_path or "").strip()[:1000]
    if sp:
        entry["snapshot_path"] = sp

    items.insert(0, entry)
    del items[MAX_RECENT_RUNS:]

    app_info_cache.set(RECENT_RUN_HISTORY_KEY, items)

    def _persist() -> None:
        try:
            app_info_cache.store()
        except Exception as e:
            logger.warning("recent_run_history persist failed: %s", e)

    threading.Thread(
        target=_persist, daemon=True, name="recent_run_history_store"
    ).start()


def get_recent_runs() -> List[dict[str, Any]]:
    """Return recent run dicts newest-first (copy safe for iteration)."""
    from utils.app_info_cache import app_info_cache

    raw = app_info_cache.get(RECENT_RUN_HISTORY_KEY, default_val=None)
    if not isinstance(raw, list):
        return []
    out: List[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _display_kind(raw: object) -> str:
    if raw is None:
        return "?"
    s = str(raw).strip().lower()
    try:
        return ModelBuilderTaskType(s).value
    except ValueError:
        return str(raw)[:32]


def _display_task_column(task_kind: str, data_sub: object) -> str:
    if data_sub:
        sub = str(data_sub).strip()
        if sub:
            return f"{task_kind}/{sub}"
    return task_kind


def format_recent_runs_for_display(entries: List[dict[str, Any]], *, limit: int = 30) -> str:
    """Plain-text block for :class:`~PySide6.QtWidgets.QPlainTextEdit` or ``QLabel``."""
    if not entries:
        return _(
            "No runs recorded yet.\n\n"
            "Completed and failed jobs from Data, Train, Convert, and Export are listed here."
        )
    lines: List[str] = []
    for e in entries[:limit]:
        ts_raw = str(e.get("ts") or "")
        ts = ts_raw[:19].replace("T", " ") if ts_raw else _("—")
        kind = _display_kind(e.get("task_type") or e.get("kind"))
        col = _display_task_column(kind, e.get("data_subcommand"))
        ok = bool(e.get("ok"))
        status = _("ok") if ok else _("fail")
        summary = str(e.get("summary") or "").strip()
        detail = str(e.get("detail") or "").strip()
        line = f"{ts}  [{col}] {status}  {summary}"
        if detail:
            if len(detail) <= 100:
                line += f" — {detail}"
            else:
                line += f" — {detail[:97]}…"
        snap = str(e.get("snapshot_path") or "").strip()
        if snap:
            line += f"\n    {_('Snapshot')}: {snap}"
        lines.append(line)
    return "\n".join(lines)
