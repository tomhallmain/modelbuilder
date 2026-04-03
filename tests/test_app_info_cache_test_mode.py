"""Ensure pytest runs use :class:`utils.app_info_cache.IsolationAppInfoCache`."""

from __future__ import annotations

import os

from utils.app_info_cache import IsolationAppInfoCache, app_info_cache


def test_pytest_enables_modelbuilder_test_cache_env() -> None:
    assert os.environ.get("MODELBUILDER_TEST_CACHE") == "1"


def test_app_info_cache_is_test_isolation_instance() -> None:
    assert isinstance(app_info_cache, IsolationAppInfoCache)
