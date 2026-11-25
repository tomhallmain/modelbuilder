"""Pages set the image size to a resolution-locked architecture's native size."""

from __future__ import annotations

import pytest

from ui.pages.convert_page import ConvertPage
from ui.pages.evaluate_page import EvaluatePage
from ui.pages.export_page import ExportPage
from ui.pages.image_size_sync import native_image_size
from ui.pages.train_page import TrainPage


def test_native_image_size_only_for_locked_architectures() -> None:
    assert native_image_size("siglip2_so400m_patch16_512") == 512
    assert native_image_size("siglip2_base_patch16_384") == 384
    assert native_image_size("resnet34") is None
    assert native_image_size("focalnet_base") is None
    assert native_image_size("siglip2_so400m") is None  # partial typing
    assert native_image_size("black-forest-labs/FLUX.1-dev") is None  # a LoRA base model


@pytest.mark.ui
@pytest.mark.parametrize("page_cls", [ExportPage, EvaluatePage, TrainPage, ConvertPage])
def test_choosing_a_locked_architecture_sets_its_native_size(qtbot, page_cls) -> None:
    page = page_cls()
    qtbot.addWidget(page)
    page.architecture.setText("siglip2_so400m_patch16_384")
    assert page.image_size.value() == 384

    # An unlocked architecture leaves whatever size is there.
    page.image_size.setValue(300)
    page.architecture.setText("resnet34")
    assert page.image_size.value() == 300


@pytest.mark.ui
@pytest.mark.parametrize("page_cls", [ExportPage, EvaluatePage, TrainPage, ConvertPage])
def test_a_size_set_after_the_architecture_still_wins(qtbot, page_cls) -> None:
    """Restore order (architecture, then size) keeps a deliberately saved mismatched size."""
    page = page_cls()
    qtbot.addWidget(page)
    page.architecture.setText("siglip2_so400m_patch16_512")
    page.image_size.setValue(320)
    assert page.image_size.value() == 320
