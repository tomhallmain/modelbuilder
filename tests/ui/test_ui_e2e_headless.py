"""
Headless UI tests at two levels:

1. **Shell smoke** — main window, all sidebar pages, About dialog (fast).
2. **Pipeline E2E through widgets** — same data + train + ONNX steps as
   ``tests/e2e/test_image_classification_pipeline.py``, but fields and buttons
   on **Train** / **Convert** drive :class:`~mb.training.trainer.ModelTrainer`
   and :func:`~mb.conversion.converters.convert_model` (no ``mb.cli.main``).

Uses ``QT_QPA_PLATFORM=offscreen`` from ``tests/ui/conftest.py``.

Long waits use :meth:`QApplication.processEvents` plus short ``time.sleep``
(not ``qtbot.wait``): on **Windows**, hundreds of nested ``QEventLoop.exec``
calls from ``qtbot.wait`` while PyTorch runs in a pool thread has triggered
native **access violations**. Polling still yields often enough for **Ctrl+C**
between sleeps. For wall-clock caps use ``pytest --timeout=…``
(``pytest-timeout``). Training time is mostly in worker threads; interrupting
the test does not cancel torch work until threads finish.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from mb import __version__ as MB_VERSION
from mb.data.class_layout import SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.data.dataset import DatasetCreator
from ui.main_window import MainWindow
from ui.pages.convert_page import ConvertPage
from ui.pages.info_page import InfoPage
from ui.pages.train_page import TrainPage

from tests.test_utils import prepare_synthetic_raw_with_snapshot
from tests.ui.qt_helpers import main_nav_stacked_widget

_RS = 'install PyTorch extras: pip install -e ".[pytorch]" or see requirements-ml.txt'
_ONNX = "install ONNX for conversion: pip install onnx (see requirements-ml.txt)"

_NAV_TRAIN = 2
_NAV_CONVERT = 3
_NAV_INFO = 6


def _sync_nav_and_stack(main_window: MainWindow, row: int) -> None:
    """
    Keep list selection and :class:`QStackedWidget` in sync.

    ``QListWidget.setCurrentRow`` does not emit ``currentRowChanged`` when the row
    is already selected, so the stack can stay on the old page if we only rely on
    the signal. Matches :meth:`MainWindow._on_nav_changed`.
    """
    nav = main_window.nav_widget
    stack = main_nav_stacked_widget(main_window)
    nav.setCurrentRow(row)
    if 0 <= row < stack.count():
        stack.setCurrentIndex(row)


def _poll_until(
    qtbot,
    predicate,
    *,
    timeout_ms: int,
    poll_ms: int = 100,
    err_msg: str = "condition not met within timeout",
) -> None:
    """
    Poll without ``qtbot.wait`` / nested ``QEventLoop.exec`` (see module doc).

    *qtbot* is kept so call sites match pytest-qt style; it is unused here.
    """
    _ = qtbot
    deadline = time.monotonic() + timeout_ms / 1000.0
    app = QApplication.instance()
    sleep_s = max(poll_ms, 1) / 1000.0
    while time.monotonic() < deadline:
        if predicate():
            return
        if app is not None:
            app.processEvents()
        time.sleep(sleep_s)
    raise AssertionError(err_msg)


@pytest.mark.slow
@pytest.mark.ui
@pytest.mark.ui_e2e
@pytest.mark.timeout(180)
def test_headless_ui_full_navigation_and_about(qtbot, main_window: MainWindow) -> None:
    nav = main_window.nav_widget
    stack = main_nav_stacked_widget(main_window)

    for row in range(nav.count()):
        _sync_nav_and_stack(main_window, row)
        qtbot.wait(20)
        assert stack.currentIndex() == row
        assert nav.currentRow() == row

    closed: list[bool] = []

    def close_about() -> None:
        w = QApplication.activeModalWidget()
        if isinstance(w, QMessageBox):
            assert MB_VERSION in w.text()
            w.close()
            closed.append(True)

    QTimer.singleShot(0, close_about)
    main_window._show_about()
    _poll_until(qtbot, lambda: bool(closed), timeout_ms=3000, err_msg="About dialog did not close")

    _sync_nav_and_stack(main_window, 0)
    assert stack.currentIndex() == 0


@pytest.mark.slow
@pytest.mark.ui
@pytest.mark.ui_e2e
@pytest.mark.e2e
@pytest.mark.timeout(180)  # typical ~1 min locally; cap hung runs (polls below fail faster with a message)
def test_headless_ui_train_pytorch_and_convert_onnx(
    qtbot, main_window: MainWindow, tmp_path: Path
) -> None:
    """
    Dataset prep stays in Python (same as CLI E2E); **train** and **convert**
    run exclusively via the **Train** / **Convert** pages (same widget instances
    as in :attr:`MainWindow.page_widgets`).
    """
    pytest.importorskip("torch", reason=_RS)
    pytest.importorskip("torchvision", reason=_RS)
    pytest.importorskip("onnx", reason=_ONNX)

    random.seed(42)
    raw = prepare_synthetic_raw_with_snapshot(tmp_path, total_images=100)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tmp_path / "model_ui_e2e.onnx"

    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=10,
    )
    assert creator.run() is True
    assert (data_dir / "train").is_dir() and (data_dir / "test").is_dir()
    for name in SYNTHETIC_DEFAULT_CLASS_NAMES:
        assert (data_dir / "train" / name).is_dir()
        assert (data_dir / "test" / name).is_dir()

    expected_pth = models_dir / "resnet18_model.pth"
    train_page = main_window.page_widgets[_NAV_TRAIN]
    convert_page = main_window.page_widgets[_NAV_CONVERT]
    assert isinstance(train_page, TrainPage)
    assert isinstance(convert_page, ConvertPage)

    # --- Train page ---
    _sync_nav_and_stack(main_window, _NAV_TRAIN)

    assert not train_page.train_subprocess.isChecked()
    train_page.framework.setCurrentText("pytorch")
    train_page.architecture.setText("resnet18")
    train_page.data_dir.setText(str(data_dir))
    train_page.output_dir.setText(str(models_dir))
    train_page.skip_snapshot.setChecked(True)
    train_page.frozen_epochs.setValue(0)
    train_page.unfrozen_epochs.setValue(1)
    train_page.batch_size.setValue(4)
    train_page.num_workers.setValue(0)

    qtbot.mouseClick(train_page.btn_validate, Qt.MouseButton.LeftButton)
    assert train_page.btn_start.isEnabled()

    qtbot.mouseClick(train_page.btn_start, Qt.MouseButton.LeftButton)

    def _train_done() -> bool:
        text = train_page.output.toPlainText().lower()
        if "[done] training complete" in text:
            return True
        if "[error]" in text:
            tail = train_page.output.toPlainText()[-3000:]
            raise AssertionError(f"Train page logged an error before success:\n{tail}")
        return False

    _poll_until(
        qtbot,
        _train_done,
        timeout_ms=120_000,
        poll_ms=200,
        err_msg="Timeout waiting for training (see Train log in failure output)",
    )
    assert expected_pth.is_file() and expected_pth.stat().st_size > 0

    # --- Convert page ---
    _sync_nav_and_stack(main_window, _NAV_CONVERT)

    convert_page.framework.setCurrentIndex(1)  # pytorch
    convert_page.target.setCurrentIndex(0)  # onnx
    convert_page.input_model.setText(str(expected_pth))
    convert_page.output_model.setText(str(onnx_path))
    convert_page.architecture.setText("resnet18")
    convert_page.num_classes.setValue(3)
    convert_page.image_size.setValue(224)
    qtbot.mouseClick(convert_page.btn_validate, Qt.MouseButton.LeftButton)
    assert convert_page.btn_convert.isEnabled()
    qtbot.mouseClick(convert_page.btn_convert, Qt.MouseButton.LeftButton)

    def _convert_done() -> bool:
        text = convert_page.output.toPlainText().lower()
        if "[done] conversion succeeded" in text:
            return True
        if "[error]" in text or "[failed] conversion failed" in text:
            tail = convert_page.output.toPlainText()[-3000:]
            raise AssertionError(f"Convert page logged failure:\n{tail}")
        return False

    _poll_until(
        qtbot,
        _convert_done,
        timeout_ms=60_000,
        poll_ms=200,
        err_msg="Timeout waiting for ONNX conversion",
    )
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0

    # --- Info page (after train + convert produced artifacts) ---
    info_page = main_window.page_widgets[_NAV_INFO]
    assert isinstance(info_page, InfoPage)
    _sync_nav_and_stack(main_window, _NAV_INFO)
    info_page.model_path.setText(str(onnx_path))
    qtbot.mouseClick(info_page.btn_model_info, Qt.MouseButton.LeftButton)
    info_text = info_page.model_output.toPlainText()
    assert str(onnx_path) in info_text or onnx_path.name in info_text
    assert "onnx" in info_text.lower()
