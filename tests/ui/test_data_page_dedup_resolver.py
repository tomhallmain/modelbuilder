"""Headless UI tests: Data page list-only dedup + duplicate resolver dialog."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QGroupBox, QListWidget, QPushButton

from mb.utils.snapshot import UnifiedSnapshot
from ui.lib.duplicates_resolver_window import DuplicatesResolverDialog
from ui.main_window import MainWindow
from ui.pages.data_page import DataPage

from tests.ui.qt_helpers import main_nav_stacked_widget, stacked_inner_page


def _sync_nav_and_stack(main_window: MainWindow, row: int) -> None:
    nav = main_window.nav_widget
    stack = main_nav_stacked_widget(main_window)
    nav.setCurrentRow(row)
    if 0 <= row < stack.count():
        stack.setCurrentIndex(row)


def _poll_until(
    predicate,
    *,
    timeout_s: float = 15.0,
    step_s: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_s
    app = QApplication.instance()
    while time.monotonic() < deadline:
        if predicate():
            return
        if app is not None:
            app.processEvents()
        time.sleep(step_s)
    raise AssertionError("condition not met within timeout")


def _put_converted_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_raw_and_snapshot_for_dedup(tmp_path: Path, *, run_id: str) -> Path:
    """
    Raw layout: two classes, intra-class duplicate pairs, and one cross-class byte-identical pair.

    Files use arbitrary bytes; :class:`ImageDeduplicator` hashes by MD5. Small-image handling uses
    PIL and skips files that are not valid images (exception path), leaving them in place.

    Snapshot lists converted paths. Cross-class review metadata is pre-set via
    :meth:`UnifiedSnapshot.set_deduplication_results` so "open resolver from snapshot" works
    without re-running dedup (the same metadata is replaced when list-only dedup finishes).
    """
    raw = tmp_path / "raw_data"
    cats_c = raw / "cats" / "CONVERTED"
    dogs_c = raw / "dogs" / "CONVERTED"
    _put_converted_file(cats_c / "unique_c.jpg", b"unique_c")
    intra_cat = b"intra_cat_pair"
    _put_converted_file(cats_c / "intra_a.jpg", intra_cat)
    _put_converted_file(cats_c / "intra_b.jpg", intra_cat)
    _put_converted_file(dogs_c / "unique_d.jpg", b"unique_d")
    intra_dog = b"intra_dog_pair"
    _put_converted_file(dogs_c / "intra_x.jpg", intra_dog)
    _put_converted_file(dogs_c / "intra_y.jpg", intra_dog)
    cross = b"cross_class_same_bytes"
    _put_converted_file(cats_c / "cross.jpg", cross)
    _put_converted_file(dogs_c / "cross_other_name.jpg", cross)

    snap = UnifiedSnapshot(run_id=run_id, raw_data_dir=str(raw.resolve()))
    rows = [
        ("k1", "cats/CONVERTED/unique_c.jpg", "unique_c.jpg", "cats"),
        ("k2", "cats/CONVERTED/intra_a.jpg", "intra_a.jpg", "cats"),
        ("k3", "cats/CONVERTED/intra_b.jpg", "intra_b.jpg", "cats"),
        ("k4", "dogs/CONVERTED/unique_d.jpg", "unique_d.jpg", "dogs"),
        ("k5", "dogs/CONVERTED/intra_x.jpg", "intra_x.jpg", "dogs"),
        ("k6", "dogs/CONVERTED/intra_y.jpg", "intra_y.jpg", "dogs"),
        ("k7", "cats/CONVERTED/cross.jpg", "cross.jpg", "cats"),
        ("k8", "dogs/CONVERTED/cross_other_name.jpg", "cross_other_name.jpg", "dogs"),
    ]
    for key, rel, bn, cls in rows:
        snap.images[key] = {
            "original": {
                "hash": key,
                "basename": bn,
                "path": rel.replace("/CONVERTED/", "/IMAGES/"),
                "format": ".jpg",
            },
            "converted": {"path": rel, "basename": bn, "class": cls},
            "dataset": None,
            "training": None,
        }
    cross_paths = [
        str((cats_c / "cross.jpg").resolve()),
        str((dogs_c / "cross_other_name.jpg").resolve()),
    ]
    snap.set_deduplication_results([{"hash": "fixture_cross_md5", "files": cross_paths}])
    out = raw / f"snapshot_{run_id}.json"
    assert snap.save(out)
    return raw.resolve()


def _resolver_dialog_from_data_page(data_page: DataPage) -> DuplicatesResolverDialog | None:
    """Prefer the page-held reference: parented QDialog may not appear in topLevelWidgets."""
    dlg = getattr(data_page, "_duplicates_resolver_dialog", None)
    return dlg if isinstance(dlg, DuplicatesResolverDialog) else None


@pytest.mark.ui
def test_duplicates_resolver_copy_and_remove_peer(tmp_path: Path, qtbot, english_gui_locale) -> None:
    p_keep = tmp_path / "keep.jpg"
    p_peer = tmp_path / "peer.jpg"
    p_keep.write_bytes(b"x")
    p_peer.write_bytes(b"y")
    items = [
        {
            "absolute_converted_path": str(p_keep),
            "converted_path": "cats/CONVERTED/keep.jpg",
            "duplicate_group_ids": ["dup_group_1"],
        },
        {
            "absolute_converted_path": str(p_peer),
            "converted_path": "dogs/CONVERTED/peer.jpg",
            "duplicate_group_ids": ["dup_group_1"],
        },
    ]
    dlg = DuplicatesResolverDialog(items)
    qtbot.addWidget(dlg)
    dlg.show()
    boxes = dlg.findChildren(QGroupBox)
    assert boxes
    lists = boxes[0].findChildren(QListWidget)
    assert len(lists) == 1
    lw = lists[0]
    assert lw.count() == 2
    lw.setCurrentRow(0)
    remove_btns = [b for b in boxes[0].findChildren(QPushButton) if "Remove" in b.text() or "remove" in b.text().lower()]
    assert remove_btns
    remove_btns[0].click()
    QApplication.processEvents()
    assert p_keep.exists()
    assert not p_peer.exists()
    dlg.close()


@pytest.mark.ui
def test_data_page_list_only_dedup_opens_resolver_cross_class_groups(
    qtbot,
    main_window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    english_gui_locale,
) -> None:
    run_id = "ui_dedup_fixture"
    raw = _build_raw_and_snapshot_for_dedup(tmp_path, run_id=run_id)
    cats_cross = raw / "cats" / "CONVERTED" / "cross.jpg"
    dogs_cross = raw / "dogs" / "CONVERTED" / "cross_other_name.jpg"

    def stub_qt_alert(parent, title, message, kind: str = "info"):
        # Auto-accept any yes/no prompt (locale-safe; avoids missing substring in translations).
        if kind == "askyesno":
            return True
        return None

    monkeypatch.setattr("ui.pages.data_page.qt_alert", stub_qt_alert)

    import ui.lib.task_progress as task_progress_mod

    real_attach = task_progress_mod.attach_progress_dialog

    def attach_non_modal(parent, title, handle, *, cancellable=True):
        dlg = real_attach(parent, title, handle, cancellable=cancellable)
        dlg.setModal(False)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        return dlg

    monkeypatch.setattr("ui.pages.data_page.attach_progress_dialog", attach_non_modal)

    _sync_nav_and_stack(main_window, 1)
    main_window.retranslate_shell_ui()
    data_page = stacked_inner_page(main_window, 1)
    assert isinstance(data_page, DataPage)
    data_page.dedup_raw_data_dir.setText(str(raw))
    data_page.dedup_list_only.setChecked(True)
    data_page.tabs.setCurrentIndex(2)
    assert "duplicate" in data_page.btn_dedup_open_resolver_from_snapshot.text().lower()
    assert "snapshot" in data_page.btn_dedup_open_resolver_from_snapshot.text().lower()
    assert data_page._dedup_resolver_fallback_note.objectName() == "dedup_resolver_fallback_note"
    assert len(data_page._dedup_resolver_fallback_note.text().strip()) > 10
    assert data_page.btn_dedup_resolver_run_id_latest.text() == "Latest"
    data_page._validate_inputs()
    assert data_page.btn_run.isEnabled(), "dedup run button should be enabled for temp raw_data"

    qtbot.mouseClick(data_page.btn_run, Qt.MouseButton.LeftButton)
    _poll_until(lambda: _resolver_dialog_from_data_page(data_page) is not None, timeout_s=60.0)
    dlg = _resolver_dialog_from_data_page(data_page)
    assert dlg is not None

    groups = dlg.findChildren(QGroupBox)
    assert len(groups) >= 1
    list_widgets: list[QListWidget] = []
    for box in groups:
        for lw in box.findChildren(QListWidget):
            if lw.count() > 0:
                list_widgets.append(lw)
    cross_lists = [lw for lw in list_widgets if lw.count() == 2]
    assert cross_lists, "expected a duplicate group with two cross-class files"
    texts = {cross_lists[0].item(i).text() for i in range(cross_lists[0].count())}
    assert any("cross.jpg" in t for t in texts)
    assert any("cross_other_name.jpg" in t for t in texts)

    cross_lists[0].setCurrentRow(0)
    copy_btns = [
        b
        for b in cross_lists[0].parentWidget().findChildren(QPushButton)
        if "copy" in b.text().lower() and "path" in b.text().lower()
    ]
    assert copy_btns
    copy_btns[0].click()
    QApplication.processEvents()
    clip = QGuiApplication.clipboard().text()
    assert clip
    assert Path(clip).name in ("cross.jpg", "cross_other_name.jpg")

    assert cats_cross.exists() and dogs_cross.exists()
    remove_btns = [
        b
        for b in cross_lists[0].parentWidget().findChildren(QPushButton)
        if "remove" in b.text().lower()
    ]
    assert remove_btns
    remove_btns[0].click()
    QApplication.processEvents()
    assert sum(1 for p in (cats_cross, dogs_cross) if p.exists()) == 1

    dlg.close()
    QApplication.processEvents()


@pytest.mark.ui
def test_deduplicate_tab_has_snapshot_resolver_recovery_widgets(
    qtbot,
    main_window: MainWindow,
    english_gui_locale,
) -> None:
    _sync_nav_and_stack(main_window, 1)
    main_window.retranslate_shell_ui()
    data_page = stacked_inner_page(main_window, 1)
    assert isinstance(data_page, DataPage)
    data_page.tabs.setCurrentIndex(2)
    btn = data_page.btn_dedup_open_resolver_from_snapshot
    assert btn is not None
    t = btn.text().lower()
    assert "duplicate" in t and "snapshot" in t
    assert data_page.dedup_resolver_run_id is not None
    assert data_page.btn_dedup_resolver_run_id_latest.text() == "Latest"
    note = data_page._dedup_resolver_fallback_note
    assert note.objectName() == "dedup_resolver_fallback_note"
    assert len(note.text().strip()) > 10


@pytest.mark.ui
def test_dedup_resolver_latest_button_fills_snapshot_run_id(
    qtbot,
    main_window: MainWindow,
    tmp_path: Path,
    english_gui_locale,
) -> None:
    run_id = "ui_dedup_latest_rid"
    raw = _build_raw_and_snapshot_for_dedup(tmp_path, run_id=run_id)
    _sync_nav_and_stack(main_window, 1)
    main_window.retranslate_shell_ui()
    data_page = stacked_inner_page(main_window, 1)
    assert isinstance(data_page, DataPage)
    data_page.tabs.setCurrentIndex(2)
    data_page.dedup_raw_data_dir.setText(str(raw))
    data_page.dedup_resolver_run_id.clear()
    qtbot.mouseClick(data_page.btn_dedup_resolver_run_id_latest, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert data_page.dedup_resolver_run_id.text().strip() == run_id


@pytest.mark.ui
def test_data_page_open_duplicate_resolver_from_snapshot_button(
    qtbot,
    main_window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    english_gui_locale,
) -> None:
    run_id = "ui_dedup_open_from_snap"
    raw = _build_raw_and_snapshot_for_dedup(tmp_path, run_id=run_id)
    cats_cross = raw / "cats" / "CONVERTED" / "cross.jpg"
    dogs_cross = raw / "dogs" / "CONVERTED" / "cross_other_name.jpg"

    def stub_qt_alert(parent, title, message, kind: str = "info"):
        # Avoid QMessageBox.exec() in headless runs if a code path shows info/warning.
        if kind == "askyesno":
            return True
        return None

    monkeypatch.setattr("ui.pages.data_page.qt_alert", stub_qt_alert)

    _sync_nav_and_stack(main_window, 1)
    main_window.retranslate_shell_ui()
    data_page = stacked_inner_page(main_window, 1)
    assert isinstance(data_page, DataPage)
    data_page.tabs.setCurrentIndex(2)
    data_page.dedup_raw_data_dir.setText(str(raw))
    data_page.dedup_resolver_run_id.setText(run_id)

    qtbot.mouseClick(data_page.btn_dedup_open_resolver_from_snapshot, Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    dlg = _resolver_dialog_from_data_page(data_page)
    assert dlg is not None
    groups = dlg.findChildren(QGroupBox)
    assert len(groups) >= 1
    list_widgets: list[QListWidget] = []
    for box in groups:
        for lw in box.findChildren(QListWidget):
            if lw.count() > 0:
                list_widgets.append(lw)
    cross_lists = [lw for lw in list_widgets if lw.count() == 2]
    assert cross_lists, "expected a duplicate group with two cross-class files"
    texts = {cross_lists[0].item(i).text() for i in range(cross_lists[0].count())}
    assert any("cross.jpg" in t for t in texts)
    assert any("cross_other_name.jpg" in t for t in texts)
    assert cats_cross.exists() and dogs_cross.exists()
    dlg.close()
    QApplication.processEvents()
