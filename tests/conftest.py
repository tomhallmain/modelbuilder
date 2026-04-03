"""Shared pytest configuration and fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from tests.fixtures.synthetic_dataset import build_synthetic_raw_data_dir

def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """
    Stabilize order (pytest discovery order is not guaranteed):

    1. ``test_synthetic_dataset_factory`` — smoke-test the shared builder first
    2. ``tests/integration/`` — dataset step before E2E
    3. Other unit tests (e.g. cancellation, run args)
    4. ``tests/e2e/`` — full pipeline last
    """

    def _phase(nodeid: str) -> int:
        n = nodeid.replace("\\", "/")
        if "test_synthetic_dataset_factory" in n:
            return 0
        if "/integration/" in n:
            return 1
        if "/e2e/" in n:
            return 3
        return 2

    items[:] = sorted(items, key=lambda it: (_phase(it.nodeid), it.nodeid))


@pytest.fixture
def synthetic_raw_data_dir(tmp_path: Path) -> Path:
    """
    A temporary raw data directory with ~100 synthetic JPEGs under
    ``<class>/JPEG_IMAGES/``, matching :class:`mb.data.dataset.DatasetCreator`.
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
