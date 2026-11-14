"""Where logs and the app info cache live, and the one-time move of repo-root cache files.

Tests that exercise the move point the legacy dir and target into tmp_path and
unset ``MODELBUILDER_TEST_APP_DATA``, so a failure cannot move the developer's
real cache.
"""

from __future__ import annotations

import logging
import os

import pytest

import mb.utils.logging_setup as logging_setup
import utils.app_info_cache as aic


def _norm(path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    legacy = tmp_path / "repo"
    target = tmp_path / "app_data" / "cache"
    legacy.mkdir()
    target.mkdir(parents=True)
    monkeypatch.setattr(aic, "_LEGACY_CACHE_DIR", str(legacy))
    # The move is skipped while this is set; conftest sets it for the session.
    monkeypatch.delenv("MODELBUILDER_TEST_APP_DATA", raising=False)
    return legacy, target


def _write(directory, name, content="x"):
    (directory / name).write_text(content, encoding="utf-8")


def test_log_files_are_written_inside_the_test_run():
    """conftest sets MODELBUILDER_TEST_APP_DATA before the first logger is made;
    set too late, the file handlers open files in the real logs folder."""
    root = _norm(os.environ["MODELBUILDER_TEST_APP_DATA"])
    handlers = [
        handler
        for name, log in logging.Logger.manager.loggerDict.items()
        if name.startswith("modelbuilder.") and isinstance(log, logging.Logger)
        for handler in log.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert handlers
    for handler in handlers:
        assert _norm(handler.baseFilename).startswith(root + os.sep)


def test_app_data_dir_honors_test_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBUILDER_TEST_APP_DATA", str(tmp_path / "override"))
    assert logging_setup.get_app_data_dir() == tmp_path / "override"
    assert logging_setup.get_log_directory() == tmp_path / "override" / "logs"


def test_app_data_dir_on_windows_uses_appdata(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELBUILDER_TEST_APP_DATA", raising=False)
    monkeypatch.setattr(logging_setup.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert logging_setup.get_app_data_dir() == tmp_path / "ModelBuilder"


def test_app_data_dir_elsewhere_uses_local_share(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELBUILDER_TEST_APP_DATA", raising=False)
    monkeypatch.setattr(logging_setup.sys, "platform", "linux")
    monkeypatch.setattr(
        logging_setup.os.path, "expanduser", lambda p: p.replace("~", str(tmp_path), 1)
    )
    expected = tmp_path / ".local" / "share" / "ModelBuilder"
    assert _norm(logging_setup.get_app_data_dir()) == _norm(expected)
    assert expected.is_dir()


def test_cache_dir_is_under_test_app_data():
    assert _norm(aic.app_info_cache_dir()) == _norm(
        os.path.join(os.environ["MODELBUILDER_TEST_APP_DATA"], "cache")
    )


def test_production_cache_paths_are_in_the_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBUILDER_TEST_APP_DATA", str(tmp_path))
    # No cache file there, so load() returns before reaching the encryptor.
    cache = aic.AppInfoCache()
    assert cache.cache_loc == os.path.join(str(tmp_path), "cache", aic.CACHE_FILENAME)
    assert cache.json_loc == os.path.join(str(tmp_path), "cache", aic.JSON_FILENAME)


def test_moves_cache_backups_and_json(dirs):
    legacy, target = dirs
    names = [aic.CACHE_FILENAME, aic.CACHE_FILENAME + ".bak", aic.CACHE_FILENAME + ".bak2",
             aic.JSON_FILENAME]
    for name in names:
        _write(legacy, name, name)

    assert aic._move_legacy_cache_files(str(target)) == str(target)

    for name in names:
        assert not (legacy / name).exists()
        assert (target / name).read_text(encoding="utf-8") == name


def test_name_clash_moves_nothing(dirs):
    legacy, target = dirs
    _write(legacy, aic.CACHE_FILENAME, "old")
    _write(legacy, aic.CACHE_FILENAME + ".bak", "old-bak")
    _write(target, aic.CACHE_FILENAME + ".bak", "new-bak")

    aic._move_legacy_cache_files(str(target))

    assert (legacy / aic.CACHE_FILENAME).read_text(encoding="utf-8") == "old"
    assert (legacy / (aic.CACHE_FILENAME + ".bak")).exists()
    assert not (target / aic.CACHE_FILENAME).exists()
    assert (target / (aic.CACHE_FILENAME + ".bak")).read_text(encoding="utf-8") == "new-bak"


def test_failed_main_move_keeps_everything_in_the_repo_root(dirs, monkeypatch):
    legacy, target = dirs
    _write(legacy, aic.CACHE_FILENAME)
    _write(legacy, aic.CACHE_FILENAME + ".bak")

    def _locked(src, dest):
        raise PermissionError(src)

    monkeypatch.setattr(aic.shutil, "move", _locked)

    assert aic._move_legacy_cache_files(str(target)) == str(legacy)
    assert (legacy / aic.CACHE_FILENAME).exists()
    assert (legacy / (aic.CACHE_FILENAME + ".bak")).exists()


def test_skipped_under_test_app_data(dirs, monkeypatch, tmp_path):
    legacy, target = dirs
    _write(legacy, aic.CACHE_FILENAME)
    monkeypatch.setenv("MODELBUILDER_TEST_APP_DATA", str(tmp_path / "scratch"))

    aic._move_legacy_cache_files(str(target))

    assert (legacy / aic.CACHE_FILENAME).exists()
    assert not (target / aic.CACHE_FILENAME).exists()


def test_unrelated_repo_files_stay(dirs):
    legacy, target = dirs
    _write(legacy, "app_info_cache.py")
    _write(legacy, aic.CACHE_FILENAME)

    aic._move_legacy_cache_files(str(target))

    assert (legacy / "app_info_cache.py").exists()
    assert not (target / "app_info_cache.py").exists()
