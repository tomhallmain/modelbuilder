"""Bundled resource paths, the default work dir, and how pipeline config paths resolve."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import mb.pipeline_config as pipeline_config
import mb.utils.app_paths as app_paths
from mb.cli import create_parser
from mb.pipeline_config import PipelineConfig, pipeline_base_dir, set_workspace_root
from mb.training.text_config import resolve_text_run_config
from mb.utils.logging_setup import get_app_data_dir


def _write_yaml(path: Path, text: str = "model:\n  default_framework: pytorch\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestResources:
    def test_checkout_is_not_a_build(self):
        assert app_paths.IS_BUILD is False

    def test_bundled_files_resolve_under_the_resource_root(self):
        from mb.utils.translations import I18N
        from utils.config import DEFAULT_APPLICATION_YAML

        for path in (pipeline_config.DEFAULT_PIPELINE_YAML, DEFAULT_APPLICATION_YAML):
            assert path.is_file()
            assert path.is_relative_to(app_paths.RESOURCE_ROOT)
        assert Path(I18N.localedir) == app_paths.resource_path("locale")
        assert Path(I18N.localedir).is_dir()


class TestWorkDir:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(app_paths.WORK_DIR_ENV_VAR, str(tmp_path / "work"))
        assert app_paths.default_work_dir() == tmp_path / "work"

    def test_defaults_to_documents(self, tmp_path, monkeypatch):
        monkeypatch.delenv(app_paths.WORK_DIR_ENV_VAR, raising=False)
        monkeypatch.setattr(app_paths, "documents_dir", lambda: tmp_path / "Docs")
        assert app_paths.default_work_dir() == tmp_path / "Docs" / "ModelBuilder"

    def test_documents_outside_windows_is_under_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_paths.sys, "platform", "linux")
        monkeypatch.setattr(app_paths.Path, "home", classmethod(lambda cls: tmp_path))
        assert app_paths.documents_dir() == tmp_path / "Documents"

    def test_not_created_on_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setenv(app_paths.WORK_DIR_ENV_VAR, str(tmp_path / "work"))
        pipeline_base_dir(None)
        assert not (tmp_path / "work").exists()


class TestPipelineBaseDir:
    def test_user_yaml_resolves_against_its_folder(self, tmp_path):
        yml = _write_yaml(tmp_path / "datasets" / "pipe.yaml")
        assert pipeline_base_dir(yml) == tmp_path / "datasets"

    def test_yaml_inside_the_workspace_resolves_against_the_workspace_root(self, tmp_path):
        yml = _write_yaml(tmp_path / "ws" / "configs" / "pipeline.yaml")
        set_workspace_root(tmp_path / "ws")
        assert pipeline_base_dir(yml) == tmp_path / "ws"

    def test_yaml_outside_the_workspace_keeps_its_folder(self, tmp_path):
        yml = _write_yaml(tmp_path / "elsewhere" / "pipe.yaml")
        set_workspace_root(tmp_path / "ws")
        assert pipeline_base_dir(yml) == tmp_path / "elsewhere"

    def test_packaged_defaults_use_the_work_dir(self):
        assert pipeline_base_dir(None) == app_paths.default_work_dir()
        assert pipeline_base_dir(pipeline_config.DEFAULT_PIPELINE_YAML) == app_paths.default_work_dir()

    def test_app_data_yaml_uses_the_work_dir(self):
        yml = _write_yaml(get_app_data_dir() / "probe" / "pipeline.yaml")
        try:
            assert pipeline_base_dir(yml) == app_paths.default_work_dir()
        finally:
            yml.unlink()

    def test_packaged_defaults_prefer_the_workspace_root(self, tmp_path):
        set_workspace_root(tmp_path / "ws")
        assert pipeline_base_dir(None) == tmp_path / "ws"


class TestResolvePath:
    def test_relative_absolute_and_home(self, tmp_path):
        cfg = PipelineConfig(_write_yaml(tmp_path / "pipe.yaml"))
        assert cfg.resolve_path("raw") == tmp_path / "raw"
        assert cfg.resolve_path(str(tmp_path / "abs")) == tmp_path / "abs"
        assert cfg.resolve_path("~/x") == Path(os.path.expanduser("~/x"))

    def test_path_reads_the_config_and_falls_back_on_blank(self, tmp_path):
        yml = _write_yaml(tmp_path / "pipe.yaml", "data:\n  data_dir: split\npaths:\n  models_dir: ''\n")
        cfg = PipelineConfig(yml)
        assert cfg.path("data.data_dir", "data") == tmp_path / "split"
        assert cfg.path("paths.models_dir", "data/models") == tmp_path / "data" / "models"
        # get() keeps the stored value, which is what is saved back to YAML.
        assert cfg.get("data.data_dir") == "split"


def test_text_section_paths_resolve_but_overrides_stay_as_given(tmp_path):
    yml = _write_yaml(
        tmp_path / "text" / "pipeline.yaml",
        "text_classification:\n  data_dir: .\n  runs_dir: runs\n",
    )
    cfg = resolve_text_run_config(PipelineConfig(yml))
    assert cfg.data_dir == tmp_path / "text"
    assert cfg.runs_dir == tmp_path / "text" / "runs"

    overridden = resolve_text_run_config(PipelineConfig(yml), {"runs_dir": "cli_runs"})
    assert overridden.runs_dir == Path("cli_runs")


def test_gather_parser_leaves_config_defaults_to_the_handler(tmp_path):
    """The parser is built before --config loads, so it must not bake in path defaults."""
    args = create_parser().parse_args(
        ["data", "gather", "--source-dir", str(tmp_path), "--subdirs", "a"]
    )
    assert args.raw_data_dir is None
    assert args.target_dir is None
    assert args.rejected_dir is None
    assert args.target_count is None


@pytest.mark.parametrize(
    "argv, attrs",
    [
        (["data", "convert"], ("raw_data_dir",)),
        (["data", "deduplicate"], ("raw_data_dir",)),
        (["data", "upscale"], ("raw_data_dir",)),
        (["data", "create-dataset"], ("raw_data_dir", "data_dir")),
        (["data", "estimate-space", "--operation", "convert"], ("raw_data_dir",)),
    ],
)
def test_data_parsers_leave_path_defaults_to_the_handlers(argv, attrs):
    args = create_parser().parse_args(argv)
    for attr in attrs:
        assert getattr(args, attr) is None


def test_unset_data_paths_come_from_the_named_config(tmp_path):
    from argparse import Namespace

    from mb.cli import _fill_data_path_defaults
    from mb.pipeline_config import reload_pipeline_config

    yml = _write_yaml(tmp_path / "pipe.yaml", "data:\n  raw_data_dir: raw\n  data_dir: split\n")
    reload_pipeline_config(yml, force=True)
    try:
        args = Namespace(raw_data_dir=None, data_dir=None)
        _fill_data_path_defaults(args, data_dir=True)
        assert args.raw_data_dir == tmp_path / "raw"
        assert args.data_dir == tmp_path / "split"

        given = Namespace(raw_data_dir=Path("cli_raw"), data_dir=None)
        _fill_data_path_defaults(given)
        assert given.raw_data_dir == Path("cli_raw")
        assert given.data_dir is None
    finally:
        reload_pipeline_config(None, force=True)


def test_convert_handler_uses_the_config_raw_dir(tmp_path, monkeypatch):
    import mb.cli as cli

    seen = {}

    class _Converter:
        def __init__(self, raw_data_dir, model_type):
            seen["raw_data_dir"] = raw_data_dir

        def run(self, **kwargs):
            return True

    monkeypatch.setattr(cli, "ImageConverter", _Converter)
    yml = _write_yaml(tmp_path / "pipe.yaml", "data:\n  raw_data_dir: raw\n")
    try:
        assert cli.main(["--config", str(yml), "data", "convert"]) == 0
    finally:
        cli.reload_pipeline_config(None, force=True)
    assert seen["raw_data_dir"] == tmp_path / "raw"


def test_cache_migration_is_skipped_in_a_build(tmp_path, monkeypatch):
    import utils.app_info_cache as aic

    legacy = tmp_path / "repo"
    target = tmp_path / "cache"
    legacy.mkdir()
    target.mkdir()
    (legacy / aic.CACHE_FILENAME).write_text("x", encoding="utf-8")
    monkeypatch.setattr(aic, "_LEGACY_CACHE_DIR", str(legacy))
    monkeypatch.delenv("MODELBUILDER_TEST_APP_DATA", raising=False)
    monkeypatch.setattr(aic, "IS_BUILD", True)

    assert aic._move_legacy_cache_files(str(target)) == str(target)
    assert (legacy / aic.CACHE_FILENAME).exists()


@pytest.fixture(autouse=True)
def _clear_workspace_root():
    set_workspace_root(None)
    yield
    set_workspace_root(None)
