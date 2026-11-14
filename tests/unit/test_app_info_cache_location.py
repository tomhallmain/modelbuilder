"""The production cache lives under the app data dir; repo-root files move there once."""

from __future__ import annotations

import os

import pytest

import utils.app_info_cache as aic


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


def test_cache_dir_is_under_test_app_data():
    assert os.path.normcase(aic.app_info_cache_dir()) == os.path.normcase(
        os.path.join(os.environ["MODELBUILDER_TEST_APP_DATA"], "cache")
    )


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
