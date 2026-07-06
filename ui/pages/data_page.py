"""Data operations page mirroring `mb data` subcommands."""

from __future__ import annotations

import os
import random
import shlex
import threading
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.data.class_layout import clear_class_discovery_cache
from mb.data.convert import ImageConverter
from mb.data.dataset import DatasetCreator, unified_snapshot_search_paths_for_dataset
from mb.data.deduplicate import ImageDeduplicator
from mb.data.gather import ImageGatherer
from mb.data.upscale import ImageUpscaler
from mb.models.types import ModelBuildStepCommand, ModelType
from mb.pipeline_config import (
    data_class_layout_defaults,
    gather_pipeline_defaults,
    get_pipeline_config,
    reload_pipeline_config,
)
from mb.space_estimate import run_convert_estimate, run_create_dataset_estimate
from mb.utils.storage import check_same_drive, check_target_external_storage
from ui.lib.qt_alert import qt_alert, qt_operation_error
from mb.utils.constants import DataPipelineSubcommand, ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.snapshot import (
    find_latest_unified_snapshot_path,
    get_duplicate_review_context_from_raw_data_dir,
    find_loadable_unified_snapshot_path_for_run_id,
    run_id_from_latest_unified_snapshot,
)
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _
from ui.controllers.model_type_field_visibility import apply_model_type_field_visibility
from ui.lib.duplicates_resolver_window import DuplicatesResolverDialog
from ui.lib.qt_log_bridge import QtLogBridge, tee_logger_to_qt
from ui.lib.fast_directory_picker_qt import get_existing_directory, get_open_file_name
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.task_progress import attach_progress_dialog
from ui.lib.tooltip_qt import ToolTip, create_tooltip
from ui.main_thread_bridge import MainThreadBridge
from ui.task_context import LongTaskContext
from ui.task_runner import start_task

logger = get_logger(__name__)

# Tab index of the Wildcard command picker (see :meth:`DataPage._build_wildcard_tab`).
WILDCARD_TAB_INDEX = 5

# Loggers for ``mb data`` steps (see ``setup_logging(script_name=...)`` in each module). Also
# ``modelbuilder.mb.space_estimate`` (see :func:`mb.utils.logging_setup.get_logger`) for disk
# heuristics during convert / create-dataset — same logger as console/file when running the GUI.
_DATA_COMMAND_LOGGERS: dict[ModelBuildStepCommand, tuple[str, ...]] = {
    ModelBuildStepCommand.GATHER: ("modelbuilder.mb.gather",),
    ModelBuildStepCommand.CONVERT: ("modelbuilder.mb.convert", "modelbuilder.mb.space_estimate"),
    ModelBuildStepCommand.DEDUPLICATE: ("modelbuilder.mb.deduplicate_images",),
    ModelBuildStepCommand.UPSCALE: ("modelbuilder.mb.upscale_small_images",),
    ModelBuildStepCommand.CREATE_DATASET: ("modelbuilder.mb.create_datasets", "modelbuilder.mb.space_estimate"),
    ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH: ("modelbuilder.mb.fix_jpeg_mismatch",),
}


class DataPage(QWidget):
    """UI scaffold for data command forms."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._title = QLabel()
        root.addWidget(self._title)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        self._last_space_estimate_msg: str | None = None
        self._space_estimate_status = QLabel()
        self._space_estimate_status.setObjectName("space_estimate_status")
        self._space_estimate_status.setWordWrap(True)
        root.addWidget(self._space_estimate_status)

        self._pipeline_group_tooltips: list[ToolTip] | None = None
        self._intro_tooltip: ToolTip | None = None

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_gather_tab(), "")
        self.tabs.addTab(self._build_convert_tab(), "")
        self.tabs.addTab(self._build_dedup_tab(), "")
        self.tabs.addTab(self._build_upscale_tab(), "")
        self.tabs.addTab(self._build_dataset_tab(), "")
        self.tabs.addTab(self._build_wildcard_tab(), "")
        root.addWidget(self.tabs, 1)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton()
        self.btn_run = QPushButton()
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_run)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        root.addWidget(self.output, 1)

        # Marshals dialogs / label updates from QThreadPool workers onto the GUI thread (see
        # :meth:`_worker_precheck_convert_or_dataset` — heavy space estimates no longer run on the main thread).
        self._gui_bridge = MainThreadBridge(self)

        self._data_log_bridge = QtLogBridge(self)
        self._data_log_bridge.line.connect(self._append, Qt.ConnectionType.QueuedConnection)
        self._duplicates_resolver_dialog: DuplicatesResolverDialog | None = None
        self._syncing_shared_pipeline_fields = False

        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_run.clicked.connect(self._run_current_command)
        self.tabs.currentChanged.connect(self._validate_inputs)
        self.wildcard_command_combo.currentIndexChanged.connect(self._on_wildcard_command_changed)
        self.wildcard_extra_args.textChanged.connect(self._sync_data_tab_run_without_log)

        self._align_shared_raw_data_dir_to_gather_default()
        for ed in self._shared_raw_data_dir_edits():
            ed.textChanged.connect(partial(self._on_shared_raw_data_dir_changed, ed))
        for ed in self._shared_run_id_edits():
            ed.textChanged.connect(partial(self._on_shared_run_id_changed, ed))
        self.dataset_data_dir.textChanged.connect(self._sync_data_tab_run_without_log)

        self.retranslate_ui(refresh_output=False)
        self._sync_data_tab_run_without_log()

    def retranslate_ui(self, *, refresh_output: bool = True) -> None:
        self._title.setText(f"<h2>{_('Data')}</h2>")
        self._intro.setText(
            _(
                "Typical order: Gather → Convert → Deduplicate (optional) → Upscale (optional) → Create Dataset. "
                "If class folders are already under your raw data root, skip Gather and start at Convert. "
                "If sources were mislabeled as .jpg, run fix-jpeg-extension-mismatch before Create Dataset "
                "(it repairs CONVERTED/ and updates the unified snapshot—no separate full Convert needed for that). "
                "Create Dataset reads training JPEGs from each class’s CONVERTED/ folder only—not from "
                "small_images_review, upscaled_small_images, or visual_media_review. "
                "See docs/DATA_PIPELINE.md for storage on external drives and large vs small images."
            )
        )
        self.tabs.setTabText(0, _("Gather"))
        self.tabs.setTabText(1, _("Convert"))
        self.tabs.setTabText(2, _("Deduplicate"))
        self.tabs.setTabText(3, _("Upscale"))
        self.tabs.setTabText(4, _("Create Dataset"))
        self.tabs.setTabText(WILDCARD_TAB_INDEX, _("Wildcard"))
        self.btn_validate.setText(_("Validate Inputs"))
        self.btn_run.setText(_("Run Data Command"))
        self.output.setPlaceholderText(_("Validation and run results will appear here."))
        self.wildcard_extra_args.setPlaceholderText(
            _(
                "Example:\n"
                "--dry-run --json\n"
                "--raw-data-dir C:\\path\\to\\raw_data"
            )
        )
        if self._last_space_estimate_msg:
            self._space_estimate_status.setText(
                _("Latest space check") + ": " + self._last_space_estimate_msg
            )
        else:
            self._space_estimate_status.setText(
                _(
                    "No space estimate yet. It updates after the disk-space check finishes "
                    "(first phase of Convert or Create Dataset)."
                )
            )
        apply_qform_label_column(
            self._gather_form,
            [
                _("Source dir"),
                _("Subdirs (space-separated)"),
                _("Target count"),
                _("Target dir"),
                _("Rejected dir"),
                _("Subdir weights"),
                _("Raw data dir"),
            ],
        )
        apply_qform_label_column(
            self._convert_form,
            [_("Raw data dir"), _("Format (jpeg/jpg)"), _("Run ID (optional)"), ""],
        )
        apply_qform_label_column(
            self._dedup_form,
            [_("Raw data dir"), "", _("Snapshot Run ID (optional)"), ""],
        )
        self.dedup_list_only.setText(
            _(
                "Review cross-class duplicates only: same-class CONVERTED duplicates are removed "
                "automatically; JSON + resolver list cross-class groups after run"
            )
        )
        apply_qform_label_column(
            self._upscale_form,
            [_("Raw data dir"), _("Review dir (optional)")],
        )
        apply_qform_label_column(
            self._dataset_form,
            [
                _("Model type"),
                _("Raw data dir"),
                _("Output data dir"),
                _("Test items per class"),
                _("Test split mode"),
                _("Small-class threshold (weighted)"),
                _("Seed (optional)"),
                _("Run ID (optional)"),
                _("Max train per class"),
                "",
                "",
                "",
            ],
        )
        self.dataset_max_train.setSpecialValueText(_("None"))
        self.dataset_balance_train.setText(_("Balance train set to smallest class"))
        self.dataset_allow_external.setText(_("Allow external/removable storage"))
        self._dataset_prereq_note.setText(
            _(
                "Before Create Dataset: (1) Raw data must use one folder per class under the raw data root "
                "(run Gather first if you do not already have that layout). "
                "(2) Run Convert so each class has normalized JPEGs under CONVERTED/ and the unified snapshot "
                "matches your sources. "
                "(3) If any files had the wrong extension for their actual image bytes, run "
                "mb data fix-jpeg-extension-mismatch (or the Wildcard tab with that subcommand)—it rewrites "
                "CONVERTED/ and patches the unified snapshot for those files; you do not need a second full Convert "
                "just to refresh the snapshot. "
                "(4) Optional: Deduplicate may move borderline-small files out of CONVERTED/ into each class’s "
                "small_images_review/; Upscale is optional. Create Dataset reads JPEGs from CONVERTED/ only—put "
                "any images you still want trained back under CONVERTED/ (from small_images_review or from "
                "upscaled_small_images after upscale) before Create Dataset. "
                "(5) Choose an output data directory with enough free space; check “Allow external/removable "
                "storage” when writing to removable drives."
            )
        )
        self._apply_pipeline_tab_tooltips()
        self._apply_pipeline_group_tooltips()
        self._apply_intro_tooltip()
        self._refresh_run_id_field_tooltips()
        self.btn_convert_run_id_latest.setText(_("Latest"))
        self.btn_dedup_resolver_run_id_latest.setText(_("Latest"))
        self.btn_dataset_run_id_latest.setText(_("Latest"))
        _tip_latest_under_raw = _(
            "Set Run ID from the newest snapshot_*.json file under the shared raw data directory "
            "(by file modification time)."
        )
        self.btn_convert_run_id_latest.setToolTip(_tip_latest_under_raw)
        self.btn_dedup_resolver_run_id_latest.setToolTip(_tip_latest_under_raw)
        self.btn_dataset_run_id_latest.setToolTip(
            _(
                "Set Run ID from the newest snapshot_*.json found across the same search paths "
                "as manual Run ID entry (raw dir, its parent, output data dir, and class folders)."
            )
        )
        self.btn_dedup_open_resolver_from_snapshot.setText(
            _("Open duplicate resolver from snapshot…")
        )
        self._dedup_resolver_fallback_note.setText(
            _(
                "If the resolver did not open after a list-only run (for example after a UI crash), "
                "use the button above to load duplicate groups from the snapshot on disk."
            )
        )
        self._dedup_create_dataset_note.setText(
            _(
                "Training sources for Create Dataset are JPEGs under each class’s CONVERTED/ only. "
                "Deduplicate moves borderline-small images from CONVERTED/ into that class’s small_images_review/ "
                "(and removes very small ones). Files that stay only in small_images_review are not used by "
                "Create Dataset unless you copy or move them back into CONVERTED/. "
                "visual_media_review holds duplicate copies of video/GIF frame extracts for review; the canonical "
                "frame JPEG is already in CONVERTED/, so you do not merge visual_media_review for Create Dataset."
            )
        )
        self._upscale_help_note.setText(
            _(
                "Upscale is optional. It enlarges images from the review tree; outputs go under "
                "<review-dir>/upscaled_small_images/. Default review dir when empty is "
                "<raw-data-dir>/small_images_review (one subfolder per class)—set Review dir explicitly if "
                "deduplicate placed files under raw_data/<class>/small_images_review/ instead. "
                "Create Dataset still reads only from CONVERTED/; copy any upscaled files you want trained into "
                "the right CONVERTED/ folder before Create Dataset. If you skip upscale, accept that images left "
                "only in small_images_review stay out of the split unless you return them to CONVERTED/ manually."
            )
        )
        self._refresh_dedup_resolver_run_id_tooltip()
        for edit in (
            self.gather_source,
            self.gather_target_dir,
            self.gather_rejected_dir,
            self.gather_raw_data_dir,
            self.convert_raw_data_dir,
            self.dedup_raw_data_dir,
            self.upscale_raw_data_dir,
            self.upscale_review_dir,
            self.dataset_raw_data_dir,
            self.dataset_data_dir,
        ):
            row = edit.parentWidget()
            if row is not None:
                btn = row.findChild(QPushButton)
                if btn is not None:
                    btn.setText(_("Browse..."))
        if refresh_output:
            self._run_startup_validation()

    def _run_startup_validation(self) -> None:
        """Called from :meth:`MainWindow._run_page_startup_validation` after cache restore."""
        self.output.clear()
        self._validate_inputs()

    def _shared_raw_data_dir_edits(self) -> tuple[QLineEdit, ...]:
        return (
            self.gather_raw_data_dir,
            self.convert_raw_data_dir,
            self.dedup_raw_data_dir,
            self.upscale_raw_data_dir,
            self.dataset_raw_data_dir,
        )

    def _shared_run_id_edits(self) -> tuple[QLineEdit, ...]:
        return (self.convert_run_id, self.dedup_resolver_run_id, self.dataset_run_id)

    def _align_shared_raw_data_dir_to_gather_default(self) -> None:
        """After tabs are built, match every raw-data field to Gather’s pipeline default."""
        text = self.gather_raw_data_dir.text()
        self._propagate_shared_raw_data_dir_from_text(text, source=None)

    def _propagate_shared_raw_data_dir_from_text(self, text: str, *, source: QLineEdit | None) -> None:
        if self._syncing_shared_pipeline_fields:
            return
        self._syncing_shared_pipeline_fields = True
        try:
            for ed in self._shared_raw_data_dir_edits():
                if ed is source:
                    continue
                ed.blockSignals(True)
                ed.setText(text)
                ed.blockSignals(False)
        finally:
            self._syncing_shared_pipeline_fields = False

    def _propagate_shared_run_id_from_text(self, text: str, *, source: QLineEdit | None) -> None:
        if self._syncing_shared_pipeline_fields:
            return
        self._syncing_shared_pipeline_fields = True
        try:
            for ed in self._shared_run_id_edits():
                if ed is source:
                    continue
                ed.blockSignals(True)
                ed.setText(text)
                ed.blockSignals(False)
        finally:
            self._syncing_shared_pipeline_fields = False

    def _on_shared_raw_data_dir_changed(self, source: QLineEdit, *_args: object) -> None:
        # QLineEdit.textChanged emits the new string; partial(..., source) forwards it as an extra arg.
        self._propagate_shared_raw_data_dir_from_text(source.text(), source=source)
        self._sync_data_tab_run_without_log()

    def _on_shared_run_id_changed(self, source: QLineEdit, *_args: object) -> None:
        self._propagate_shared_run_id_from_text(source.text(), source=source)
        self._sync_data_tab_run_without_log()

    def _pipeline_raw_path(self) -> Path:
        return Path(self.gather_raw_data_dir.text().strip() or "raw_data")

    def _pipeline_shared_dict(self) -> dict[str, str]:
        return {
            "raw_data_dir": self.gather_raw_data_dir.text(),
            "run_id": self.convert_run_id.text(),
        }

    def _pipeline_shared_for_restore(self, state: dict) -> tuple[str, str]:
        p = state.get("pipeline")
        if isinstance(p, dict):
            return str(p.get("raw_data_dir") or ""), str(p.get("run_id") or "")
        raw_first = ""
        for key in ("gather", "convert", "dedup", "upscale", "dataset"):
            d = state.get(key) or {}
            if isinstance(d, dict):
                v = d.get("raw_data_dir")
                if isinstance(v, str) and v.strip():
                    raw_first = v.strip()
                    break
        run_id = ""
        c = state.get("convert") or {}
        if isinstance(c, dict):
            run_id = str(c.get("run_id", ""))
        if not run_id.strip():
            ds = state.get("dataset") or {}
            if isinstance(ds, dict):
                run_id = str(ds.get("run_id", ""))
        if not run_id.strip():
            d = state.get("dedup") or {}
            if isinstance(d, dict):
                run_id = str(d.get("resolver_run_id", ""))
        return raw_first, run_id

    def _apply_shared_pipeline_text(self, raw_data_dir: str, run_id: str) -> None:
        self._syncing_shared_pipeline_fields = True
        try:
            for ed in self._shared_raw_data_dir_edits():
                ed.blockSignals(True)
                ed.setText(raw_data_dir)
                ed.blockSignals(False)
            for ed in self._shared_run_id_edits():
                ed.blockSignals(True)
                ed.setText(run_id)
                ed.blockSignals(False)
        finally:
            self._syncing_shared_pipeline_fields = False

    def _wildcard_command_value_for_persist(self) -> str:
        """CLI subcommand string for the Wildcard tab combo (stable across ``currentData()`` quirks)."""
        idx = self.wildcard_command_combo.currentIndex()
        if idx < 0:
            return ModelBuildStepCommand.GATHER.value
        data = self.wildcard_command_combo.itemData(idx)
        if isinstance(data, ModelBuildStepCommand):
            return data.value
        parsed = ModelBuildStepCommand.try_from(data)
        if parsed is not None:
            return parsed.value
        return ModelBuildStepCommand.GATHER.value

    def _apply_wildcard_section_from_state(self, w: object) -> None:
        """Restore Wildcard command + extra args from cached ``wildcard`` dict."""
        if not isinstance(w, dict):
            return
        wc = w.get("command")
        if wc is not None:
            parsed = ModelBuildStepCommand.try_from(wc)
            if parsed is not None:
                want = parsed.value
                for i in range(self.wildcard_command_combo.count()):
                    data = self.wildcard_command_combo.itemData(i)
                    cand: ModelBuildStepCommand | None
                    if isinstance(data, ModelBuildStepCommand):
                        cand = data
                    else:
                        cand = ModelBuildStepCommand.try_from(data)
                    if cand is not None and cand.value == want:
                        self.wildcard_command_combo.setCurrentIndex(i)
                        break
        ex = w.get("extra_args")
        if isinstance(ex, str):
            self.wildcard_extra_args.setPlainText(ex)

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        pipe = self._pipeline_shared_dict()
        return {
            "tab": int(self.tabs.currentIndex()),
            "pipeline": pipe,
            "gather": {
                "source": self.gather_source.text(),
                "subdirs": self.gather_subdirs.text(),
                "target_count": int(self.gather_target_count.value()),
                "target_dir": self.gather_target_dir.text(),
                "rejected_dir": self.gather_rejected_dir.text(),
                "subdir_weights": self.gather_subdir_weights.text(),
            },
            "convert": {
                "format": self.convert_format.text(),
                "skip_space": bool(self.convert_skip_space.isChecked()),
            },
            "dedup": {
                "list_only": bool(self.dedup_list_only.isChecked()),
            },
            "upscale": {
                "review_dir": self.upscale_review_dir.text(),
            },
            "dataset": {
                "model_type": self.dataset_model_type.currentText(),
                "data_dir": self.dataset_data_dir.text(),
                "test_per_class": int(self.dataset_test_per_class.value()),
                "test_split_mode": self.dataset_test_split_mode.currentData(),
                "test_small_class_threshold": int(self.dataset_test_small_threshold.value()),
                "seed": int(self.dataset_seed.value()),
                "max_train": int(self.dataset_max_train.value()),
                "balance_train": bool(self.dataset_balance_train.isChecked()),
                "allow_external": bool(self.dataset_allow_external.isChecked()),
                "skip_space": bool(self.dataset_skip_space.isChecked()),
            },
            "wildcard": {
                "command": self._wildcard_command_value_for_persist(),
                "extra_args": self.wildcard_extra_args.toPlainText(),
            },
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        self.tabs.blockSignals(True)
        try:
            tab = state.get("tab")
            if isinstance(tab, int) and 0 <= tab < self.tabs.count():
                self.tabs.setCurrentIndex(tab)

            self._apply_wildcard_section_from_state(state.get("wildcard"))

            raw_g, run_g = self._pipeline_shared_for_restore(state)
            self._apply_shared_pipeline_text(raw_g, run_g)

            g = state.get("gather") or {}
            if isinstance(g, dict):
                self.gather_source.setText(str(g.get("source", "")))
                self.gather_subdirs.setText(str(g.get("subdirs", "")))
                tc = g.get("target_count")
                if isinstance(tc, int):
                    self.gather_target_count.setValue(tc)
                self.gather_target_dir.setText(str(g.get("target_dir", "")))
                self.gather_rejected_dir.setText(str(g.get("rejected_dir", "")))
                self.gather_subdir_weights.setText(str(g.get("subdir_weights", "")))

            c = state.get("convert") or {}
            if isinstance(c, dict):
                self.convert_format.setText(str(c.get("format", "jpeg")))
                self.convert_skip_space.setChecked(bool(c.get("skip_space", False)))

            d = state.get("dedup") or {}
            if isinstance(d, dict):
                self.dedup_list_only.setChecked(bool(d.get("list_only", False)))

            u = state.get("upscale") or {}
            if isinstance(u, dict):
                self.upscale_review_dir.setText(str(u.get("review_dir", "")))

            ds = state.get("dataset") or {}
            if isinstance(ds, dict):
                dmt = ds.get("model_type")
                if isinstance(dmt, str):
                    dmtix = self.dataset_model_type.findText(dmt)
                    if dmtix >= 0:
                        self.dataset_model_type.setCurrentIndex(dmtix)
                self.dataset_data_dir.setText(str(ds.get("data_dir", "")))
                tpc = ds.get("test_per_class")
                if isinstance(tpc, int):
                    self.dataset_test_per_class.setValue(tpc)
                tsm = ds.get("test_split_mode")
                if isinstance(tsm, str):
                    tix = self.dataset_test_split_mode.findData(tsm)
                    if tix >= 0:
                        self.dataset_test_split_mode.setCurrentIndex(tix)
                tst = ds.get("test_small_class_threshold")
                if isinstance(tst, int):
                    self.dataset_test_small_threshold.setValue(tst)
                sd = ds.get("seed")
                if isinstance(sd, int):
                    self.dataset_seed.setValue(sd)
                mt = ds.get("max_train")
                if isinstance(mt, int):
                    self.dataset_max_train.setValue(mt)
                self.dataset_balance_train.setChecked(bool(ds.get("balance_train", False)))
                self.dataset_allow_external.setChecked(bool(ds.get("allow_external", False)))
                self.dataset_skip_space.setChecked(bool(ds.get("skip_space", False)))
        except Exception:
            pass
        finally:
            self.tabs.blockSignals(False)
        self._on_dataset_model_type_changed()

    def _build_gather_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        gd = gather_pipeline_defaults()
        self._gather_group = QGroupBox("mb data gather")
        form = QFormLayout(self._gather_group)
        self._gather_form = form
        self.gather_source = QLineEdit()
        self.gather_subdirs = QLineEdit()
        self.gather_target_count = QSpinBox()
        self.gather_target_count.setRange(1, 5_000_000)
        self.gather_target_count.setValue(gd["target_count"])
        self.gather_target_dir = QLineEdit(str(gd["target_dir"]))
        self.gather_rejected_dir = QLineEdit(str(gd["rejected_dir"]))
        self.gather_subdir_weights = QLineEdit()
        self.gather_raw_data_dir = QLineEdit(str(gd["raw_data_dir"]))

        form.addRow(_("Source dir"), self._path_row(self.gather_source, select_dir=True))
        form.addRow(_("Subdirs (space-separated)"), self.gather_subdirs)
        form.addRow(_("Target count"), self.gather_target_count)
        form.addRow(_("Target dir"), self._path_row(self.gather_target_dir, select_dir=True))
        form.addRow(_("Rejected dir"), self._path_row(self.gather_rejected_dir, select_dir=True))
        form.addRow(_("Subdir weights"), self.gather_subdir_weights)
        form.addRow(_("Raw data dir"), self._path_row(self.gather_raw_data_dir, select_dir=True))
        v.addWidget(self._gather_group)
        v.addStretch(1)
        return tab

    def _build_convert_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self._convert_group = QGroupBox("mb data convert")
        form = QFormLayout(self._convert_group)
        self._convert_form = form
        self.convert_raw_data_dir = QLineEdit("raw_data")
        self.convert_format = QLineEdit("jpeg")
        self.convert_run_id = QLineEdit()
        convert_run_id_row, self.btn_convert_run_id_latest = self._run_id_row(
            self.convert_run_id, self._on_latest_run_id_under_shared_raw_dir
        )
        form.addRow(_("Raw data dir"), self._path_row(self.convert_raw_data_dir, select_dir=True))
        form.addRow(_("Format (jpeg/jpg)"), self.convert_format)
        form.addRow(_("Run ID (optional)"), convert_run_id_row)
        self.convert_skip_space = QCheckBox(
            _("Skip free-space check (not recommended; use if you accept the risk)")
        )
        form.addRow("", self.convert_skip_space)
        v.addWidget(self._convert_group)
        v.addStretch(1)
        return tab

    def _build_dedup_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self._dedup_group = QGroupBox("mb data deduplicate")
        form = QFormLayout(self._dedup_group)
        self._dedup_form = form
        self.dedup_raw_data_dir = QLineEdit("raw_data")
        form.addRow(_("Raw data dir"), self._path_row(self.dedup_raw_data_dir, select_dir=True))
        self.dedup_list_only = QCheckBox(
            _(
                "Review cross-class duplicates only: same-class CONVERTED duplicates are removed "
                "automatically; JSON + resolver list cross-class groups after run"
            )
        )
        form.addRow("", self.dedup_list_only)
        self.dedup_resolver_run_id = QLineEdit()
        dedup_resolver_rid_row, self.btn_dedup_resolver_run_id_latest = self._run_id_row(
            self.dedup_resolver_run_id,
            self._on_latest_run_id_under_shared_raw_dir,
        )
        form.addRow(_("Snapshot Run ID (optional)"), dedup_resolver_rid_row)
        self.btn_dedup_open_resolver_from_snapshot = QPushButton()
        self.btn_dedup_open_resolver_from_snapshot.clicked.connect(
            self._on_open_duplicate_resolver_from_snapshot
        )
        form.addRow("", self.btn_dedup_open_resolver_from_snapshot)
        v.addWidget(self._dedup_group)

        self._dedup_resolver_fallback_note = QLabel()
        self._dedup_resolver_fallback_note.setWordWrap(True)
        self._dedup_resolver_fallback_note.setObjectName("dedup_resolver_fallback_note")
        v.addWidget(self._dedup_resolver_fallback_note)

        self._dedup_scope_note = QLabel(
            _(
                "Scope note: Deduplicate only processes files inside each class folder's "
                "CONVERTED directory (raw_data/<class>/CONVERTED). Other folders, including "
                "IMAGES and root-level utility directories, are not scanned."
            )
        )
        self._dedup_scope_note.setWordWrap(True)
        self._dedup_scope_note.setObjectName("dedup_scope_note")
        v.addWidget(self._dedup_scope_note)

        self._dedup_create_dataset_note = QLabel()
        self._dedup_create_dataset_note.setWordWrap(True)
        self._dedup_create_dataset_note.setObjectName("dedup_create_dataset_note")
        v.addWidget(self._dedup_create_dataset_note)

        v.addStretch(1)
        return tab

    def _build_upscale_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self._upscale_help_note = QLabel()
        self._upscale_help_note.setWordWrap(True)
        self._upscale_help_note.setObjectName("upscale_help_note")
        v.addWidget(self._upscale_help_note)

        self._upscale_group = QGroupBox("mb data upscale")
        form = QFormLayout(self._upscale_group)
        self._upscale_form = form
        self.upscale_raw_data_dir = QLineEdit("raw_data")
        self.upscale_review_dir = QLineEdit()
        form.addRow(_("Raw data dir"), self._path_row(self.upscale_raw_data_dir, select_dir=True))
        form.addRow(
            _("Review dir (optional)"),
            self._path_row(self.upscale_review_dir, select_dir=True),
        )
        v.addWidget(self._upscale_group)
        v.addStretch(1)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self._dataset_prereq_note = QLabel()
        self._dataset_prereq_note.setWordWrap(True)
        self._dataset_prereq_note.setObjectName("dataset_prereq_note")
        v.addWidget(self._dataset_prereq_note)

        self._dataset_group = QGroupBox("mb data create-dataset")
        form = QFormLayout(self._dataset_group)
        self._dataset_form = form
        self.dataset_model_type = QComboBox()
        for mt in ModelType:
            self.dataset_model_type.addItem(mt.value)
        self.dataset_raw_data_dir = QLineEdit("raw_data")
        self.dataset_data_dir = QLineEdit("data")
        self.dataset_test_per_class = QSpinBox()
        self.dataset_test_per_class.setRange(1, 1_000_000)
        self.dataset_test_per_class.setValue(1000)
        self.dataset_seed = QSpinBox()
        self.dataset_seed.setRange(0, 2_147_483_647)
        self.dataset_run_id = QLineEdit()
        dataset_run_id_row, self.btn_dataset_run_id_latest = self._run_id_row(
            self.dataset_run_id, self._on_dataset_use_latest_run_id
        )
        self.dataset_max_train = QSpinBox()
        self.dataset_max_train.setRange(0, 1_000_000)
        self.dataset_max_train.setSpecialValueText(_("None"))
        self.dataset_balance_train = QCheckBox(_("Balance train set to smallest class"))
        self.dataset_allow_external = QCheckBox(_("Allow external/removable storage"))

        form.addRow(_("Model type"), self.dataset_model_type)
        form.addRow(_("Raw data dir"), self._path_row(self.dataset_raw_data_dir, select_dir=True))
        form.addRow(_("Output data dir"), self._path_row(self.dataset_data_dir, select_dir=True))
        form.addRow(_("Test items per class"), self.dataset_test_per_class)
        self.dataset_test_split_mode = QComboBox()
        self.dataset_test_split_mode.addItem(_("Fixed count per class"), "fixed")
        self.dataset_test_split_mode.addItem(_("Dataset-weighted (modulated)"), "dataset_weighted")
        form.addRow(_("Test split mode"), self.dataset_test_split_mode)
        self.dataset_test_small_threshold = QSpinBox()
        self.dataset_test_small_threshold.setRange(0, 10_000_000)
        self.dataset_test_small_threshold.setSpecialValueText(_("Same as test per class"))
        self.dataset_test_small_threshold.setValue(0)
        form.addRow(_("Small-class threshold (weighted)"), self.dataset_test_small_threshold)
        form.addRow(_("Seed (optional)"), self.dataset_seed)
        form.addRow(_("Run ID (optional)"), dataset_run_id_row)
        form.addRow(_("Max train per class"), self.dataset_max_train)
        form.addRow("", self.dataset_balance_train)
        form.addRow("", self.dataset_allow_external)
        self.dataset_skip_space = QCheckBox(
            _("Skip free-space check on output drive (not recommended)")
        )
        form.addRow("", self.dataset_skip_space)
        v.addWidget(self._dataset_group)
        v.addStretch(1)

        # image_generation_lora: LoraDatasetCreator only takes raw_data_dir/data_dir (a flat
        # copy, no split/balance/snapshot integration) — hide everything else on this tab.
        self._dataset_classification_only_rows = (
            self.dataset_test_per_class,
            self.dataset_test_split_mode,
            self.dataset_test_small_threshold,
            self.dataset_seed,
            dataset_run_id_row,
            self.dataset_max_train,
            self.dataset_balance_train,
            self.dataset_allow_external,
            self.dataset_skip_space,
        )
        self.dataset_model_type.currentIndexChanged.connect(self._on_dataset_model_type_changed)
        self._on_dataset_model_type_changed()
        return tab

    def _build_wildcard_tab(self) -> QWidget:
        """Run ``mb data <subcommand>`` with optional extra CLI tokens (same as the terminal)."""
        tab = QWidget()
        v = QVBoxLayout(tab)

        self._wildcard_group = QGroupBox("mb data (wildcard)")
        inner = QVBoxLayout(self._wildcard_group)
        self.wildcard_command_combo = QComboBox()
        for cmd in ModelBuildStepCommand.gui_wildcard_command_values():
            label = cmd.value.replace("-", " ")
            self.wildcard_command_combo.addItem(label, cmd)
        self.wildcard_extra_args = QPlainTextEdit()
        self.wildcard_extra_args.setObjectName("wildcard_extra_args")
        self.wildcard_extra_args.setMinimumHeight(100)
        self._wildcard_hint = QLabel(
            _(
                "Pick the subcommand, then type any arguments for ``mb data <subcommand> …`` "
                "(line breaks are fine). The active pipeline config path is passed as ``--config`` "
                "when you do not include it below. Cancellation is not applied to this path—use Ctrl+C in a terminal "
                "if you need to stop a long job."
            )
        )
        self._wildcard_hint.setWordWrap(True)
        inner.addWidget(QLabel(_("Command")))
        inner.addWidget(self.wildcard_command_combo)
        inner.addWidget(QLabel(_("Arguments (optional)")))
        inner.addWidget(self.wildcard_extra_args, 1)
        inner.addWidget(self._wildcard_hint)
        v.addWidget(self._wildcard_group)
        v.addStretch(1)
        return tab

    def _on_wildcard_command_changed(self, *_args: object) -> None:
        self._sync_data_tab_run_without_log()

    def _parse_wildcard_extra_argv(self) -> list[str]:
        raw = self.wildcard_extra_args.toPlainText().strip()
        if not raw:
            return []
        try:
            return shlex.split(raw, posix=os.name != "nt")
        except ValueError as e:
            raise ValueError(_("Could not parse arguments (check quotes and line breaks): {err}").format(err=e)) from e

    def _collect_wildcard_cli_inputs(self) -> dict:
        cmd = ModelBuildStepCommand.try_from(self._wildcard_command_value_for_persist())
        if cmd is None:
            cmd = ModelBuildStepCommand.GATHER
        return {
            "wildcard_cli": True,
            "data_subcommand": cmd,
            "extra_argv": self._parse_wildcard_extra_argv(),
        }

    def _apply_pipeline_tab_tooltips(self) -> None:
        """Hover hints for each Data tab (native tab bar tooltips)."""
        tips = [
            _(
                "Copy samples from source subdirectories into class folders under the raw data root. "
                "Skip this tab if your data is already organized as one folder per class."
            ),
            _(
                "Normalize images to JPEG under each class’s CONVERTED/ folder. "
                "Video/GIF extracts: the frame JPEG is written to CONVERTED/ and a duplicate copy to "
                "visual_media_review/ for inspection only. "
                "Point Raw data dir at the same tree on an internal or external drive; outputs stay beside sources."
            ),
            _(
                "Remove duplicates and quarantine very small images from class CONVERTED/ folders only "
                "(raw_data/<class>/CONVERTED); other folders are ignored. "
                "Borderline-small files move to each class’s small_images_review/—Create Dataset does not read "
                "that folder unless you copy back into CONVERTED/."
            ),
            _(
                "Optional: enlarge undersized images from the review area. Outputs go under "
                "upscaled_small_images/; Create Dataset still uses CONVERTED/ only—merge copies there if needed. "
                "Default review dir when empty is raw_data/small_images_review (class subfolders); override Review "
                "dir if your review files live under raw_data/<class>/small_images_review/."
            ),
            _(
                "After initial Convert (and any JPEG extension repair if needed), optionally Dedup/Upscale: "
                "copies from each class’s CONVERTED/ into train/ and test/ under the output data directory. "
                "Does not pull from small_images_review, upscaled_small_images, or visual_media_review. "
                "A good place to put the final dataset on your main drive while raw data stays external."
            ),
            _(
                "Choose a data subcommand and optional ``mb data …`` arguments in the text box "
                "(same flags as the CLI)."
            ),
        ]
        for i, text in enumerate(tips):
            if i < self.tabs.count():
                self.tabs.setTabToolTip(i, text)

    def _apply_pipeline_group_tooltips(self) -> None:
        """Delayed tooltips on each step’s group box (see ui.lib.tooltip_qt)."""
        groups = [
            self._gather_group,
            self._convert_group,
            self._dedup_group,
            self._upscale_group,
            self._dataset_group,
            self._wildcard_group,
        ]
        texts = [
            _(
                "Gather copies files into timestamped runs under the target dir and tracks hashes so reruns skip "
                "already-seen files. Set raw data dir to the pipeline root used by later steps."
            ),
            _(
                "Convert walks class folders, writes CONVERTED/ JPEGs, and records a unified snapshot. "
                "For image classification, videos and multi-frame GIFs may yield a random frame JPEG in CONVERTED/ "
                "plus a duplicate in visual_media_review/ for review only. "
                "Very large images are downscaled (max edge 4000px)."
            ),
            _(
                "Deduplicate scans only class CONVERTED/ folders, removes duplicate images, and handles tiny "
                "dimensions: removes very small images and moves borderline-small ones to each class’s "
                "small_images_review/ for optional upscale. Create Dataset sources CONVERTED/ JPEGs only."
            ),
            _(
                "Optional: reads the small-image review tree and writes upscaled copies under "
                "upscaled_small_images/. Create Dataset does not read that output—copy wanted files into CONVERTED/ "
                "before create-dataset if they should appear in train/test."
            ),
            _(
                "Prerequisite: Convert has been run (CONVERTED/ + snapshot). JPEG extension repair updates "
                "that snapshot for repaired files—you do not need another full Convert for snapshot alignment. "
                "Copies only from each class’s CONVERTED/ (not small_images_review or visual_media_review). "
                "Writes train/ and test/ to the output data dir. Set output on a spacious internal disk if raw "
                "data lives on external storage."
            ),
            _(
                "Runs ``mb data <subcommand>`` with the optional argument text you provide "
                "(parsed like the shell). The GUI injects ``--config`` when it is not already present."
            ),
        ]
        if self._pipeline_group_tooltips is None or len(self._pipeline_group_tooltips) != len(
            groups
        ):
            self._pipeline_group_tooltips = [
                create_tooltip(g, t) for g, t in zip(groups, texts)
            ]
        else:
            for tip, t in zip(self._pipeline_group_tooltips, texts):
                tip.set_text(t)

    def _apply_intro_tooltip(self) -> None:
        """Extra detail on hover over the intro paragraph."""
        detail = _(
            "Storage: convert reads and writes under the raw data path you choose (including on removable drives). "
            "Create Dataset copies into the output data directory—often set that to your main disk. "
            "Large images: convert downscales long edges; tiny images: deduplicate moves small files to "
            "small_images_review/. JPEG extension mismatches: run fix-jpeg-extension-mismatch (updates CONVERTED/ "
            "and the unified snapshot for affected files; no extra Convert pass required for that). "
            "Create Dataset globs JPEGs from each class’s CONVERTED/ only. "
            "visual_media_review/ is a duplicate of video/GIF frame outputs for human review—not a second input. "
            "After deduplicate, files that remain only under small_images_review/ (or only under upscaled_small_images/ "
            "after upscale) are not used unless you copy the ones you want back into CONVERTED/. "
            "Full discussion: docs/DATA_PIPELINE.md."
        )
        if self._intro_tooltip is None:
            self._intro_tooltip = create_tooltip(self._intro, detail)
        else:
            self._intro_tooltip.set_text(detail)

    def _run_id_row(self, edit: QLineEdit, use_latest_slot) -> tuple[QWidget, QPushButton]:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        latest = QPushButton()
        latest.setObjectName("run_id_latest_button")
        latest.clicked.connect(use_latest_slot)
        h.addWidget(edit, 1)
        h.addWidget(latest, 0)
        return row, latest

    def _path_row(self, edit: QLineEdit, select_dir: bool = True) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, select_dir=select_dir))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, select_dir: bool = True) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if select_dir:
            value = get_existing_directory(
                self,
                _("Select directory"),
                start,
            )
            if value:
                edit.setText(value)
        else:
            value = get_open_file_name(
                self,
                _("Select file"),
                start,
                _("All files (*.*)"),
            )
            if value:
                edit.setText(value)
        self._validate_inputs()

    def _on_latest_run_id_under_shared_raw_dir(self) -> None:
        raw = self._pipeline_raw_path()
        rid = run_id_from_latest_unified_snapshot([raw], quiet=True)
        if not rid:
            qt_alert(
                self,
                _("No snapshot found"),
                _("No snapshot_*.json files were found under:\n{path}").format(path=raw),
                kind="warning",
            )
            return
        self.convert_run_id.setText(rid)
        self._sync_data_tab_run_without_log()

    def _on_dataset_use_latest_run_id(self) -> None:
        raw_d = self._pipeline_raw_path()
        data_d = Path(self.dataset_data_dir.text().strip() or "data")
        paths = unified_snapshot_search_paths_for_dataset(raw_d, data_d)
        rid = run_id_from_latest_unified_snapshot(paths, quiet=True)
        if not rid:
            qt_alert(
                self,
                _("No snapshot found"),
                _("No snapshot_*.json files were found in the search paths for the current directories."),
                kind="warning",
            )
            return
        self.dataset_run_id.setText(rid)
        self._sync_data_tab_run_without_log()

    def _on_dataset_model_type_changed(self) -> None:
        mt = ModelType.from_pipeline_value(self.dataset_model_type.currentText())
        apply_model_type_field_visibility(
            self._dataset_form,
            mt,
            {
                widget: (
                    ModelType.IMAGE_CLASSIFICATION,
                    ModelType.OBJECT_DETECTION,
                    ModelType.IMAGE_GENERATION,
                )
                for widget in self._dataset_classification_only_rows
            },
        )

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _refresh_run_id_field_tooltips(self) -> None:
        """Explain expected snapshot filename and search paths from current form values."""
        raw_shared = self._pipeline_raw_path()
        self.convert_run_id.setToolTip(
            _(
                "Optional. Leave empty to start a new conversion run and write a new snapshot file.\n\n"
                "If set, the job resumes that run: a unified snapshot file must exist and load correctly, named:\n"
                "  snapshot_<run_id>.json\n"
                "placed directly under the raw data directory:\n"
                "  {raw}\n"
                "Example: snapshot_20260408_abc123.json"
            ).format(raw=raw_shared)
        )
        raw_d = raw_shared
        data_d = Path(self.dataset_data_dir.text().strip() or "data")
        if raw_d.exists():
            paths = unified_snapshot_search_paths_for_dataset(raw_d, data_d)
            path_lines = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(paths))
        else:
            path_lines = _(
                "  (Raw data directory must exist to list class folders; until then, "
                "search still includes the raw dir, its parent, and the output data dir.)"
            )
        self.dataset_run_id.setToolTip(
            _(
                "Optional. Leave empty to use the latest unified snapshot found along the search order below.\n\n"
                "If set, the file must exist, load correctly, and be named:\n"
                "  snapshot_<run_id>.json\n"
                "The job searches these directories in order (first match wins):\n"
                "{paths}\n"
                "Output data directory (for context): {data_dir}"
            ).format(paths=path_lines, data_dir=data_d)
        )
        self._refresh_dedup_resolver_run_id_tooltip()

    def _refresh_dedup_resolver_run_id_tooltip(self) -> None:
        raw = self._pipeline_raw_path()
        self.dedup_resolver_run_id.setToolTip(
            _(
                "Optional. Same Run ID as Convert / Create Dataset (shared across the Data page). "
                "Leave empty to load the latest loadable snapshot under:\n"
                "  {raw}\n\n"
                "If set, the file must exist and be named:\n"
                "  snapshot_<run_id>.json\n"
                "in that directory. Used when opening the duplicate resolver from disk; deduplicate "
                "Run does not require a Run ID."
            ).format(raw=raw)
        )

    def _sync_data_tab_run_without_log(self) -> None:
        """Update Run ID tooltips and Run button state without appending to the output log."""
        self._refresh_run_id_field_tooltips()
        command = self._current_command()
        try:
            self._collect_inputs(command)
            self.btn_run.setEnabled(True)
            self.btn_run.setToolTip("")
        except ValueError as exc:
            self.btn_run.setEnabled(False)
            self.btn_run.setToolTip(str(exc))

    def _set_busy(self, busy: bool) -> None:
        self.btn_validate.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and self._can_run())
        self.tabs.setEnabled(not busy)

    def _current_command(self) -> ModelBuildStepCommand:
        if self.tabs.currentIndex() == WILDCARD_TAB_INDEX:
            parsed = ModelBuildStepCommand.try_from(self._wildcard_command_value_for_persist())
            if parsed is not None:
                return parsed
            return ModelBuildStepCommand.GATHER
        tabs = ModelBuildStepCommand.data_page_tab_values()
        i = self.tabs.currentIndex()
        if 0 <= i < len(tabs):
            return tabs[i]
        return tabs[0]

    def _can_run(self) -> bool:
        command = self._current_command()
        try:
            self._collect_inputs(command)
            return True
        except ValueError:
            return False

    def _validate_inputs(self) -> None:
        self._refresh_run_id_field_tooltips()
        command = self._current_command()
        try:
            self._collect_inputs(command)
            self.btn_run.setEnabled(True)
            self.btn_run.setToolTip("")
            self._append(_("[ok] {cmd}: inputs look valid").format(cmd=command.value))
        except ValueError as exc:
            self.btn_run.setEnabled(False)
            self.btn_run.setToolTip(str(exc))
            self._append(_("[invalid] {cmd}: {err}").format(cmd=command.value, err=exc))

    def _collect_inputs(self, command: ModelBuildStepCommand) -> dict:
        if self.tabs.currentIndex() == WILDCARD_TAB_INDEX:
            return self._collect_wildcard_cli_inputs()
        if command == ModelBuildStepCommand.GATHER:
            gd = gather_pipeline_defaults()
            source_dir = Path(self.gather_source.text().strip())
            if not source_dir.exists():
                raise ValueError(_("Source dir does not exist."))
            subdirs = [x for x in self.gather_subdirs.text().strip().split() if x]
            if not subdirs:
                raise ValueError(_("Provide at least one subdirectory."))
            weights_raw = self.gather_subdir_weights.text().strip()
            weights = {}
            if weights_raw:
                for pair in weights_raw.split(","):
                    if ":" not in pair:
                        raise ValueError(_("Subdir weights must be in subdir:weight format."))
                    name, value = pair.split(":", 1)
                    weights[name.strip()] = float(value.strip())
            return {
                "source_dir": source_dir,
                "subdirs": subdirs,
                "target_count": int(self.gather_target_count.value()),
                "target_dir": Path(
                    self.gather_target_dir.text().strip() or str(gd["target_dir"])
                ),
                "rejected_dir": Path(self.gather_rejected_dir.text().strip())
                if self.gather_rejected_dir.text().strip()
                else None,
                "subdir_weights": weights if weights else None,
                "raw_data_dir": Path(
                    self.gather_raw_data_dir.text().strip() or str(gd["raw_data_dir"])
                ),
            }
        if command == ModelBuildStepCommand.CONVERT:
            fmt = self.convert_format.text().strip().lower() or "jpeg"
            if fmt not in {"jpeg", "jpg"}:
                raise ValueError(_("Format must be jpeg or jpg."))
            raw_data_dir = self._pipeline_raw_path()
            run_id = self.convert_run_id.text().strip() or None
            if run_id:
                if not raw_data_dir.exists():
                    raise ValueError(
                        _("Raw data directory must exist when a Run ID is set (snapshot is loaded from there).")
                    )
                if find_loadable_unified_snapshot_path_for_run_id([raw_data_dir], run_id) is None:
                    raise ValueError(
                        _("No loadable unified snapshot for Run ID {rid!r}. Expected {name} under:\n  {dir}").format(
                            rid=run_id,
                            name=f"snapshot_{run_id}.json",
                            dir=raw_data_dir,
                        )
                    )
            return {
                "raw_data_dir": raw_data_dir,
                "format": fmt,
                "run_id": run_id,
                "skip_space_check": bool(self.convert_skip_space.isChecked()),
            }
        if command == ModelBuildStepCommand.DEDUPLICATE:
            return {
                "raw_data_dir": self._pipeline_raw_path(),
                "list_only": bool(self.dedup_list_only.isChecked()),
            }
        if command == ModelBuildStepCommand.UPSCALE:
            review_raw = self.upscale_review_dir.text().strip()
            return {
                "raw_data_dir": self._pipeline_raw_path(),
                "review_dir": Path(review_raw) if review_raw else None,
            }
        if command == ModelBuildStepCommand.CREATE_DATASET:
            mt = ModelType.from_pipeline_value(self.dataset_model_type.currentText())
            raw_data_dir = self._pipeline_raw_path()
            data_dir = Path(self.dataset_data_dir.text().strip() or "data")
            if not raw_data_dir.exists():
                raise ValueError(_("Raw data dir does not exist."))
            if mt == ModelType.IMAGE_GENERATION_LORA:
                # LoraDatasetCreator only takes raw_data_dir/data_dir (flat copy, no split/
                # balance/snapshot integration) — see mb/data/lora_dataset.py.
                return {
                    "model_type": mt,
                    "raw_data_dir": raw_data_dir,
                    "data_dir": data_dir,
                }
            run_id = self.dataset_run_id.text().strip() or None
            if run_id:
                search_paths = unified_snapshot_search_paths_for_dataset(raw_data_dir, data_dir)
                if find_loadable_unified_snapshot_path_for_run_id(search_paths, run_id) is None:
                    raise ValueError(
                        _(
                            "No loadable unified snapshot for Run ID {rid!r} in the search paths "
                            "(see the Run ID field tooltip for locations checked)."
                        ).format(rid=run_id)
                    )
            tpc = int(self.dataset_test_per_class.value())
            tst_raw = int(self.dataset_test_small_threshold.value())
            # Spinbox 0 is the special "Same as test per class" value. Sending None here made
            # :class:`~mb.data.dataset.DatasetCreator` fall through to pipeline
            # ``data.test_small_class_threshold`` instead of using this tab's anchor — pass the
            # anchor explicitly so GUI and pipeline stay aligned.
            test_small_thr = tst_raw if tst_raw > 0 else tpc
            return {
                "model_type": mt,
                "raw_data_dir": raw_data_dir,
                "data_dir": data_dir,
                "test_per_class": tpc,
                "test_split_mode": self.dataset_test_split_mode.currentData(),
                "test_small_class_threshold": test_small_thr,
                "seed": int(self.dataset_seed.value()) if self.dataset_seed.value() > 0 else None,
                "run_id": run_id,
                "balance_train": bool(self.dataset_balance_train.isChecked()),
                "max_train_per_class": int(self.dataset_max_train.value()) if self.dataset_max_train.value() > 0 else None,
                "allow_external_storage": bool(self.dataset_allow_external.isChecked()),
                "skip_space_check": bool(self.dataset_skip_space.isChecked()),
            }
        raise ValueError(_("Unknown command: {cmd}").format(cmd=command.value))

    def _snapshot_path_after_successful_data_run(self) -> str | None:
        """Best-effort path to the newest ``snapshot_*.json`` after convert or create-dataset."""
        cmd = getattr(self, "_pending_command", None)
        payload = getattr(self, "_pending_payload", None) or {}
        try:
            if cmd == ModelBuildStepCommand.CONVERT:
                raw = payload.get("raw_data_dir")
                p = find_latest_unified_snapshot_path([raw]) if raw else None
                return str(p.resolve()) if p else None
            if cmd == ModelBuildStepCommand.CREATE_DATASET:
                data_dir = payload.get("data_dir")
                raw = payload.get("raw_data_dir")
                paths = [x for x in (data_dir, raw) if x]
                p = find_latest_unified_snapshot_path(paths) if paths else None
                return str(p.resolve()) if p else None
        except OSError as exc:
            logger.warning("Could not resolve latest snapshot path: %s", exc)
        return None

    def _apply_space_estimate_to_ui(self, message: str) -> None:
        self._last_space_estimate_msg = message
        self._space_estimate_status.setText(_("Latest space check") + ": " + message)
        self._append(f"[space] {message}")

    def _run_space_estimate(self, command: ModelBuildStepCommand, payload: dict):
        """Disk-heavy; call from worker thread or from tests on the GUI thread."""
        reload_pipeline_config(payload.get("pipeline_config_path"), force=True)
        mt = ModelType.from_pipeline_value(get_pipeline_config().get("model.default_type"))
        if command == ModelBuildStepCommand.CONVERT:
            return run_convert_estimate(payload["raw_data_dir"], mt)
        return run_create_dataset_estimate(payload["raw_data_dir"], payload["data_dir"])

    def _space_precheck_ui(self, command: ModelBuildStepCommand, payload: dict) -> bool:
        """
        Run heuristic disk estimate (main-thread / tests). For the Run button, precheck runs
        in the worker via :meth:`_worker_precheck_convert_or_dataset` so the UI stays responsive.
        """
        if command not in (ModelBuildStepCommand.CONVERT, ModelBuildStepCommand.CREATE_DATASET):
            return True
        if payload.get("skip_space_check"):
            self._apply_space_estimate_to_ui(_("Skipped (free-space estimate not run)."))
            return True
        r = self._run_space_estimate(command, payload)
        self._apply_space_estimate_to_ui(r.message)
        if r.ok or payload.get("skip_space_check"):
            return True
        if qt_alert(
            self,
            _("Insufficient disk space (estimate)"),
            _("{msg}\n\nContinue anyway?").format(msg=r.message),
            kind="askyesno",
        ):
            payload["skip_space_check"] = True
            return True
        return False

    def _worker_precheck_convert_or_dataset(self, command: ModelBuildStepCommand, payload: dict) -> bool:
        """
        Space estimate + optional dialogs for convert / create-dataset, from QThreadPool worker.
        Heavy work runs here; UI updates and ``qt_alert`` use :attr:`_gui_bridge`.
        """
        if command not in (ModelBuildStepCommand.CONVERT, ModelBuildStepCommand.CREATE_DATASET):
            return True
        if payload.get("skip_space_check"):
            self._gui_bridge.invoke(
                lambda: self._apply_space_estimate_to_ui(_("Skipped (free-space estimate not run)."))
            )
        else:
            r = self._run_space_estimate(command, payload)
            self._gui_bridge.invoke(lambda: self._apply_space_estimate_to_ui(r.message))
            if not r.ok:

                def _ask_low_space() -> bool:
                    return bool(
                        qt_alert(
                            self,
                            _("Insufficient disk space (estimate)"),
                            _("{msg}\n\nContinue anyway?").format(msg=r.message),
                            kind="askyesno",
                        )
                    )

                if not self._gui_bridge.invoke(_ask_low_space):
                    return False
                payload["skip_space_check"] = True

        if command == ModelBuildStepCommand.CREATE_DATASET:

            def _create_dataset_guards() -> bool:
                if check_target_external_storage(
                    logger, payload["data_dir"], override=payload["allow_external_storage"]
                ):
                    qt_alert(
                        self,
                        _("External storage"),
                        _(
                            'The target data directory is on external or removable storage, and '
                            '"Allow external storage" is not checked. Enable it on the Create Dataset tab '
                            "or choose a different data directory."
                        ),
                        kind="warning",
                    )
                    return False
                if check_same_drive(payload["raw_data_dir"], payload["data_dir"]):
                    if not qt_alert(
                        self,
                        _("Confirm"),
                        _("Source and target are on the same drive. Continue dataset creation?"),
                        kind="askyesno",
                    ):
                        return False
                return True

            return bool(self._gui_bridge.invoke(_create_dataset_guards))

        return True

    def _run_current_command(self) -> None:
        command = self._current_command()
        payload = self._collect_inputs(command)
        w = self.window()
        payload["pipeline_config_path"] = (
            w._effective_pipeline_config_path()
            if w is not None and hasattr(w, "_effective_pipeline_config_path")
            else None
        )

        self._append(f"[run] mb data {command.value}")
        self._pending_run_summary = f"mb data {command.value}"
        self._pending_command = command
        self._pending_payload = payload
        self._pending_data_subcommand = DataPipelineSubcommand.try_from(command)
        self._set_busy(True)
        handle = start_task(
            self._execute_data_command,
            self._on_run_success,
            self._on_run_error,
            lambda: self._set_busy(False),
            command,
            payload,
            pass_context=True,
            on_cancelled=self._on_run_cancelled,
        )
        attach_progress_dialog(self, _("Data: {cmd}").format(cmd=command.value), handle, cancellable=True)

    def _execute_data_command(self, ctx: LongTaskContext, command: ModelBuildStepCommand, payload: dict) -> bool:
        log_names = _DATA_COMMAND_LOGGERS.get(command, ())
        if log_names:
            with tee_logger_to_qt(self._data_log_bridge, *log_names):
                return self._execute_data_command_impl(ctx, command, payload)
        return self._execute_data_command_impl(ctx, command, payload)

    def _execute_data_command_impl(
        self, ctx: LongTaskContext, command: ModelBuildStepCommand, payload: dict
    ) -> bool:
        if payload.get("wildcard_cli"):
            return self._execute_wildcard_mb_data_cli(ctx, command, payload)

        ce = ctx.cancel_event
        if command in (ModelBuildStepCommand.CONVERT, ModelBuildStepCommand.CREATE_DATASET):
            if not payload.get("skip_space_check"):
                ctx.progress(_("Checking free space…"))
            if not self._worker_precheck_convert_or_dataset(command, payload):
                return False
            if command == ModelBuildStepCommand.CONVERT:
                ctx.progress(_("Converting…"))
            else:
                ctx.progress(_("Creating dataset…"))

        if command == ModelBuildStepCommand.GATHER:
            layout = data_class_layout_defaults()
            mt = ModelType.from_pipeline_value(get_pipeline_config().get("model.default_type"))
            gatherer = ImageGatherer(
                source_dir=str(payload["source_dir"]),
                valid_subdirs=payload["subdirs"],
                target_dir=payload["target_dir"],
                target_count=payload["target_count"],
                rejected_dir=payload["rejected_dir"],
                subdir_weights=payload["subdir_weights"],
                class_qualifying_subdir=layout.get("class_qualifying_subdir"),
                model_type=mt,
            )
            gatherer.raw_data_dir = payload["raw_data_dir"]
            return bool(gatherer.run(cancel_event=ce))
        if command == ModelBuildStepCommand.CONVERT:
            mt = ModelType.from_pipeline_value(get_pipeline_config().get("model.default_type"))
            converter = ImageConverter(raw_data_dir=payload["raw_data_dir"], model_type=mt)
            return bool(
                converter.run(
                    cancel_event=ce,
                    skip_space_check=payload.get("skip_space_check", False),
                    run_id=payload.get("run_id"),
                )
            )
        if command == ModelBuildStepCommand.DEDUPLICATE:
            deduplicator = ImageDeduplicator(raw_data_dir=payload["raw_data_dir"])
            return bool(
                deduplicator.run(
                    cancel_event=ce,
                    list_only=payload.get("list_only", False),
                )
            )
        if command == ModelBuildStepCommand.UPSCALE:
            review_dir = payload["review_dir"] or (payload["raw_data_dir"] / "small_images_review")
            upscaler = ImageUpscaler(review_dir=review_dir)
            return bool(upscaler.run(cancel_event=ce))
        if command == ModelBuildStepCommand.CREATE_DATASET:
            if payload.get("model_type") == ModelType.IMAGE_GENERATION_LORA:
                from mb.data.lora_dataset import LoraDatasetCreator

                lora_creator = LoraDatasetCreator(
                    raw_data_dir=payload["raw_data_dir"], data_dir=payload["data_dir"]
                )
                return bool(lora_creator.run(cancel_event=ce))
            if payload["seed"] is not None:
                random.seed(payload["seed"])
            creator = DatasetCreator(
                raw_data_dir=payload["raw_data_dir"],
                data_dir=payload["data_dir"],
                test_per_class=payload["test_per_class"],
                balance_train=payload["balance_train"],
                max_train_per_class=payload["max_train_per_class"],
                run_id=payload["run_id"],
                skip_space_check=payload.get("skip_space_check", False),
                test_split_mode=payload.get("test_split_mode"),
                test_small_class_threshold=payload.get("test_small_class_threshold"),
            )
            return bool(creator.run(cancel_event=ce))
        raise ValueError(_("Unknown command: {cmd}").format(cmd=command.value))

    def _execute_wildcard_mb_data_cli(
        self, ctx: LongTaskContext, command: ModelBuildStepCommand, payload: dict
    ) -> bool:
        """Invoke :func:`mb.cli.main` with ``data <subcommand>`` and user-supplied tokens."""
        from mb.cli import main

        sub = payload["data_subcommand"]
        extra = list(payload["extra_argv"])
        pc = payload.get("pipeline_config_path")
        # ``--config`` is defined on the *root* parser; it must appear before ``data``, not after the
        # subcommand name (otherwise argparse reports "unrecognized arguments: --config …").
        if pc and "--config" not in extra and "-c" not in extra:
            argv = ["--config", str(pc), "data", sub.value, *extra]
        else:
            argv = ["data", sub.value, *extra]
        ctx.progress(_("Running: {line}").format(line="mb " + " ".join(argv)))
        code = main(argv)
        return code == 0

    def _on_run_success(self, success: bool) -> None:
        summary = getattr(self, "_pending_run_summary", "mb data")
        sub = getattr(self, "_pending_data_subcommand", None)
        snap = self._snapshot_path_after_successful_data_run() if success else None
        if success:
            # Gather/Convert/etc. can add or remove class-folder directories under the
            # shared raw data dir; drop the cached discovery so the next check reflects it.
            clear_class_discovery_cache()
            self._append(_("[done] Data command completed successfully."))
            if snap:
                self._append(f"[snapshot] {snap}")
            if (
                getattr(self, "_pending_command", None) == ModelBuildStepCommand.DEDUPLICATE
                and bool(getattr(self, "_pending_payload", {}).get("list_only", False))
            ):
                self._maybe_open_duplicates_resolver()
            append_recent_run(
                ModelBuilderTaskType.DATA,
                summary,
                True,
                data_subcommand=sub,
                snapshot_path=snap,
            )
        else:
            self._append(_("[failed] Data command reported failure."))
            append_recent_run(
                ModelBuilderTaskType.DATA,
                summary,
                False,
                "reported failure",
                data_subcommand=sub,
            )

    def _on_run_cancelled(self) -> None:
        self._append(
            _(
                "[stopped] Data command cancelled — partial copies or snapshot updates may exist; check folders before re-running."
            )
        )
        append_recent_run(
            ModelBuilderTaskType.DATA,
            getattr(self, "_pending_run_summary", "mb data"),
            False,
            "cancelled",
            data_subcommand=getattr(self, "_pending_data_subcommand", None),
        )

    def _on_run_error(self, message: str) -> None:
        self._append(f"[error] {message}")
        append_recent_run(
            ModelBuilderTaskType.DATA,
            getattr(self, "_pending_run_summary", "mb data"),
            False,
            message,
            data_subcommand=getattr(self, "_pending_data_subcommand", None),
        )
        qt_operation_error(
            self,
            _("Data command failed"),
            _("A data pipeline step reported an error. See Details for the full message."),
            detail=message,
        )

    def _present_duplicates_resolver_window(
        self,
        items: list,
        *,
        confirm: bool,
        title: str = "",
        message: str = "",
    ) -> None:
        if confirm:
            if not qt_alert(
                self,
                title,
                message,
                kind="askyesno",
            ):
                return
        self._duplicates_resolver_dialog = DuplicatesResolverDialog(items, parent=self)
        self._duplicates_resolver_dialog.show()
        self._duplicates_resolver_dialog.raise_()
        self._duplicates_resolver_dialog.activateWindow()

    def _maybe_open_duplicates_resolver(self) -> None:
        """Offer opening a resolver window after list-only dedup completes."""
        payload = getattr(self, "_pending_payload", None) or {}
        raw_data_dir = payload.get("raw_data_dir")
        if not isinstance(raw_data_dir, Path):
            return
        snap, items = get_duplicate_review_context_from_raw_data_dir(
            raw_data_dir, run_id=None, logger=logger
        )
        if snap is None:
            self._append(_("[dedup] No loadable snapshot found for duplicate resolver."))
            return
        if not items:
            self._append(_("[dedup] No potential duplicates were recorded in the snapshot."))
            return
        self._present_duplicates_resolver_window(
            items,
            confirm=True,
            title=_("Duplicates found"),
            message=_("Open duplicate resolver window now?"),
        )

    def _on_open_duplicate_resolver_from_snapshot(self) -> None:
        raw = self._pipeline_raw_path()
        if not raw.exists():
            qt_alert(
                self,
                _("Invalid raw data directory"),
                _("Raw data directory does not exist:\n{path}").format(path=raw),
                kind="warning",
            )
            return
        run_id = self.convert_run_id.text().strip() or None
        snap, items = get_duplicate_review_context_from_raw_data_dir(
            raw, run_id=run_id, logger=logger
        )
        if snap is None:
            qt_alert(
                self,
                _("No snapshot"),
                _(
                    "No loadable unified snapshot was found under:\n{path}\n\n"
                    "Expected snapshot_<run_id>.json (leave Run ID empty for the latest file tried in sorted name order)."
                ).format(path=raw),
                kind="warning",
            )
            return
        if not items:
            qt_alert(
                self,
                _("Nothing to review"),
                _(
                    "A snapshot loaded, but it does not list any cross-class duplicate groups for the resolver. "
                    "Run Deduplicate with list-only if you have not yet, or pick another Run ID."
                ),
                kind="info",
            )
            return
        self._present_duplicates_resolver_window(items, confirm=False)
