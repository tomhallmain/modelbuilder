"""
Inspect trained model files and ImageFolder-style dataset directories.

Used by ``mb info model`` / ``mb info dataset`` and the desktop Info page so
behavior stays aligned.
"""

from __future__ import annotations

from pathlib import Path

from mb.conversion.converters import detect_model_framework
from mb.models.frameworks.registry import list_architectures
from mb.utils.translations import _

# Match legacy Info page / CLI: common image extensions under class folders
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _count_class_images(class_dir: Path) -> int:
    n = 0
    for ext in _IMAGE_EXTS:
        n += len(list(class_dir.glob(f"*{ext}")))
    return n


def model_info_text(model_path: Path) -> str:
    """
    Build a human-readable report for *model_path*.

    Ensures framework registration side effects have run so architecture lists are complete.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a regular file.
    """
    resolved = model_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    if not resolved.is_file():
        raise ValueError(_("Path is not a file: {path}").format(path=resolved))

    framework = detect_model_framework(resolved)
    lines = [
        _("Path: {path}").format(path=resolved),
        _("Size: {n:,} bytes").format(n=resolved.stat().st_size),
        _("Detected framework/type: {fw}").format(fw=framework or _("unknown")),
        "",
    ]
    try:
        from mb.models.frameworks import pytorch  # noqa: F401
    except Exception:
        pass
    try:
        from mb.models.frameworks import keras  # noqa: F401
    except Exception:
        pass
    archs = list_architectures()
    lines.append(_("Registered architectures:"))
    for fw, items in archs.items():
        lines.append(
            "- {fw}: {names}".format(
                fw=fw,
                names=", ".join(items) if items else _("(none)"),
            )
        )
    return "\n".join(lines)


def dataset_info_text(data_dir: Path) -> str:
    """
    Summarize ``train`` / ``test`` splits under *data_dir* (per-class subfolders, image counts).

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError: If *data_dir* is not a directory.
    """
    resolved = data_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    if not resolved.is_dir():
        raise ValueError(_("Path is not a directory: {path}").format(path=resolved))

    lines = [_("Data dir: {path}").format(path=resolved), ""]
    for split in ("train", "test"):
        split_dir = resolved / split
        lines.append(f"[{split}]")
        if not split_dir.exists():
            lines.append(_("  missing"))
            lines.append("")
            continue
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        total = 0
        for cls in class_dirs:
            count = _count_class_images(cls)
            total += count
            lines.append(f"  {cls.name}: {count}")
        lines.append(_("  total: {n}").format(n=total))
        lines.append("")
    return "\n".join(lines)
