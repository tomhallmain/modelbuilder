"""Where bundled resources and user work data live.

Bundled read-only files (``locale/``, ``mb/config/*.yaml``) resolve through
:func:`resource_path`: in a checkout that is the repo, in a Nuitka build the folder
the executable runs from, where the build has to place them at the same relative
paths. App state (logs, settings, cache) lives in the app data dir
(:func:`mb.utils.logging_setup.get_app_data_dir`). Datasets, models and runs are
user data: relative paths for them resolve against the pipeline config's base dir
(:func:`mb.pipeline_config.pipeline_base_dir`), which falls back to
:func:`default_work_dir`.

No ``mb`` imports: modules across the package resolve their paths through this one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Nuitka defines __compiled__ in every module it compiles.
IS_BUILD = "__compiled__" in globals()

#: The repo checkout, or the folder a compiled build runs from (mb/utils/ -> mb/ -> root).
RESOURCE_ROOT = Path(__file__).resolve().parents[2]

#: Replaces :func:`default_work_dir` as a whole; the test suite points it at a scratch dir.
WORK_DIR_ENV_VAR = "MODELBUILDER_WORK_DIR"
_WORK_DIR_NAME = "ModelBuilder"


def resource_path(*parts: str) -> Path:
    """Absolute path of a bundled, read-only file or folder."""
    return RESOURCE_ROOT.joinpath(*parts)


def _windows_documents_dir() -> Optional[Path]:
    """The Documents folder as Windows reports it, which follows a redirect (e.g. to OneDrive)."""
    try:
        import ctypes
        from ctypes import wintypes

        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        # CSIDL_PERSONAL (Documents), SHGFP_TYPE_CURRENT
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:
            return Path(buf.value)
    except Exception:
        pass
    return None


def documents_dir() -> Path:
    """The user's Documents folder."""
    if sys.platform == "win32":
        found = _windows_documents_dir()
        if found is not None:
            return found
    return Path.home() / "Documents"


def default_work_dir() -> Path:
    """``MODELBUILDER_WORK_DIR`` if set, else ``<Documents>/ModelBuilder``. Not created here.

    Relative dataset/model/run paths resolve here when no user pipeline YAML and no GUI
    workspace supply a base. The app data dir is not used for this: on Windows it is the
    roaming profile, and datasets and models can run to many gigabytes.
    """
    override = os.environ.get(WORK_DIR_ENV_VAR)
    if override:
        return Path(override)
    return documents_dir() / _WORK_DIR_NAME
