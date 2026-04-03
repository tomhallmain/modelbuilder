"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

# Ephemeral app-data root: application.yaml, pipeline.yaml, and logs never use the developer's
# Roaming folder during pytest (see :func:`utils.config.get_user_application_config_path`).
_test_app_data = tempfile.mkdtemp(prefix="modelbuilder_test_appdata_")
os.environ["MODELBUILDER_TEST_APP_DATA"] = _test_app_data
atexit.register(lambda: shutil.rmtree(_test_app_data, ignore_errors=True))

# Isolate :mod:`utils.app_info_cache` from encrypted store / user cache (see ``IsolationAppInfoCache``).
os.environ.setdefault("MODELBUILDER_TEST_CACHE", "1")

from pathlib import Path
from typing import List

import pytest

from PIL import Image

from tests.fixtures.synthetic_dataset import build_synthetic_raw_data_dir


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """
    Stabilize order (pytest discovery order is not guaranteed):

    1. ``test_synthetic_dataset_factory`` — smoke-test the shared builder first
    2. ``tests/integration/`` — data pipeline steps
    3. ``tests/framework/`` — optional framework smoke (torch/tf)
    4. Other unit tests (e.g. CLI, cancellation, run args)
    5. ``tests/ui/`` — headless PySide6 (pytest-qt); ``test_ui_e2e_headless`` runs **last**
       within this folder so Train → Convert → Info widget flow runs after page unit tests
    6. ``tests/e2e/`` — full CLI pipeline last
    """

    def _phase(nodeid: str) -> int:
        n = nodeid.replace("\\", "/")
        if "test_synthetic_dataset_factory" in n:
            return 0
        if "/integration/" in n:
            return 1
        if "/framework/" in n:
            return 2
        if "/ui/" in n:
            return 5
        if "/e2e/" in n:
            return 6
        return 3

    def _ui_e2e_last(nodeid: str) -> int:
        """Within ``tests/ui/``, run ``test_ui_e2e_headless.py`` after all other UI modules."""
        n = nodeid.replace("\\", "/")
        if "/ui/" in n and "test_ui_e2e_headless.py" in n:
            return 1
        return 0

    items[:] = sorted(
        items,
        key=lambda it: (_phase(it.nodeid), _ui_e2e_last(it.nodeid), it.nodeid),
    )


@pytest.fixture
def synthetic_raw_data_dir(tmp_path: Path) -> Path:
    """
    A temporary raw data directory with ~100 synthetic JPEGs under
    ``<class>/CONVERTED/``, matching :class:`mb.data.dataset.DatasetCreator`.
    """
    root = tmp_path / "raw_data"
    build_synthetic_raw_data_dir(root, total_images=100, seed=42)
    return root


@pytest.fixture
def synthetic_raw_data_dir_custom_total(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """
    Same as ``synthetic_raw_data_dir`` but total image count comes from
    ``request.param`` (default 100). Use::

        @pytest.mark.parametrize("synthetic_raw_data_dir_custom_total", [24], indirect=True)
    """
    total = getattr(request, "param", 100)
    root = tmp_path / "raw_data"
    build_synthetic_raw_data_dir(root, total_images=int(total), seed=42)
    return root


@pytest.fixture
def two_class_classification_data_dir(tmp_path: Path) -> Path:
    """
    Minimal ImageFolder layout (two classes) for :class:`~mb.training.trainer.ModelTrainer` smoke tests.
    """
    data = tmp_path / "data"
    for split, per_class in (("train", 4), ("test", 2)):
        for ci, cls in enumerate(("class_a", "class_b")):
            d = data / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                path = d / f"img_{i:02d}.jpg"
                Image.new(
                    "RGB",
                    (64, 64),
                    (10 + i * 8, 30 + ci * 15, 90),
                ).save(path, quality=92)
    return data
