"""The ``--smoke-test`` mode of the build's entry scripts, run from the checkout.

The full runs start ``app_cli.py`` / ``app_gui.py`` in a fresh process: the smoke
test rewrites the environment and AppInfo's identity, installs a Qt message handler
(which would replace pytest-qt's) and loads torch, transformers and an event loop.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.app_info_cache as aic

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_TEST_FILE = REPO_ROOT / "mb" / "utils" / "smoke_test.py"


def _load_smoke_test_module():
    """mb/utils/smoke_test.py by path, as build_exe.py loads it."""
    spec = importlib.util.spec_from_file_location("smoke_test_under_test", SMOKE_TEST_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_points_into_scratch(tmp_path):
    module = _load_smoke_test_module()
    env = module.smoke_test_environment(str(tmp_path))
    scratch = os.path.normcase(str(tmp_path))
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                 "MODELBUILDER_WORK_DIR", "MODELBUILDER_KEY_STORE_DIR", "MODELBUILDER_KEY_BACKUP_DIR",
                 "MODELBUILDER_SETTINGS_INI"):
        assert os.path.normcase(env[name]).startswith(scratch), name
    assert env["MODELBUILDER_TEST_CACHE"] == "0"
    assert env[module.passphrase_env_var(module.SMOKE_SERVICE_NAME)]
    assert env[module.SCRATCH_DIR_ENV] == str(tmp_path)


def test_smoke_identity_is_not_the_apps():
    from mb.utils.constants import AppInfo

    module = _load_smoke_test_module()
    assert module.SMOKE_SERVICE_NAME != AppInfo.SERVICE_NAME
    assert module.SMOKE_APP_IDENTIFIER != AppInfo.APP_IDENTIFIER


def test_options():
    module = _load_smoke_test_module()
    assert module.smoke_test_options(["--smoke-test", "--expect-oqs"]) == (True, True)
    assert module.smoke_test_options(["--smoke-test"]) == (True, False)
    assert module.smoke_test_options(["train"]) == (False, False)


def test_legacy_cache_is_not_moved_in_a_smoke_run(tmp_path, monkeypatch):
    legacy = tmp_path / "repo"
    target = tmp_path / "app_data"
    legacy.mkdir()
    target.mkdir()
    (legacy / aic.CACHE_FILENAME).write_bytes(b"real cache")
    monkeypatch.setattr(aic, "_LEGACY_CACHE_DIR", str(legacy))
    monkeypatch.delenv("MODELBUILDER_TEST_APP_DATA", raising=False)
    monkeypatch.setenv("MODELBUILDER_SMOKE_DIR", str(tmp_path))
    assert aic._move_legacy_cache_files(str(target)) == str(target)
    assert (legacy / aic.CACHE_FILENAME).is_file()
    assert not (target / aic.CACHE_FILENAME).exists()


def test_settings_ini_override(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings

    import ui.workspace as workspace

    ini = tmp_path / "settings.ini"
    monkeypatch.setenv(workspace.SETTINGS_INI_ENV_VAR, str(ini))
    settings = workspace.default_settings()
    assert settings.format() == QSettings.Format.IniFormat
    assert os.path.normcase(settings.fileName()) == os.path.normcase(str(ini))


def _run_entry(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    for name in ("torch", "torchvision", "onnx", "onnxscript", "onnxruntime", "transformers",
                 "sklearn", "imageio_ffmpeg"):
        pytest.importorskip(name)
    env = dict(os.environ, MODELBUILDER_SMOKE_DIR=str(tmp_path / "scratch"))
    os.makedirs(env["MODELBUILDER_SMOKE_DIR"])
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--smoke-test"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900,
    )


@pytest.mark.slow
@pytest.mark.timeout(960)
def test_headless_smoke_test_passes_in_checkout(tmp_path):
    result = _run_entry("app_cli.py", tmp_path)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


@pytest.mark.slow
@pytest.mark.timeout(960)
def test_gui_smoke_test_passes_in_checkout(tmp_path):
    pytest.importorskip("PySide6")
    result = _run_entry("app_gui.py", tmp_path)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
