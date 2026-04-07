"""Data operations page mirroring `mb data` subcommands."""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.data.convert import ImageConverter
from mb.data.dataset import DatasetCreator
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
from mb.utils.snapshot import find_latest_unified_snapshot_path
from mb.utils.translations import _
from ui.lib.qt_log_bridge import QtLogBridge, tee_logger_to_qt
from ui.lib.fast_directory_picker_qt import get_existing_directory, get_open_file_name
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.task_progress import attach_progress_dialog
from ui.lib.tooltip_qt import ToolTip, create_tooltip
from ui.main_thread_bridge import MainThreadBridge
from ui.task_context import LongTaskContext
from ui.task_runner import start_task

logger = logging.getLogger(__name__)

# Loggers for ``mb data`` steps (see ``setup_logging(script_name=...)`` in each module). Also
# ``modelbuilder.mb.space_estimate`` (see :func:`mb.utils.logging_setup.get_logger`) for disk
# heuristics during convert / create-dataset — same logger as console/file when running the GUI.
_DATA_COMMAND_LOGGERS: dict[ModelBuildStepCommand, tuple[str, ...]] = {
    ModelBuildStepCommand.GATHER: ("modelbuilder.mb.gather",),
    ModelBuildStepCommand.CONVERT: ("modelbuilder.mb.convert", "modelbuilder.mb.space_estimate"),
    ModelBuildStepCommand.DEDUPLICATE: ("modelbuilder.mb.deduplicate_images",),
    ModelBuildStepCommand.UPSCALE: ("modelbuilder.mb.upscale_small_images",),
    ModelBuildStepCommand.CREATE_DATASET: ("modelbuilder.mb.create_datasets", "modelbuilder.mb.space_estimate"),
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

        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_run.clicked.connect(self._run_current_command)
        self.tabs.currentChanged.connect(self._validate_inputs)

        self.retranslate_ui(refresh_output=False)

    def retranslate_ui(self, *, refresh_output: bool = True) -> None:
        self._title.setText(f"<h2>{_('Data')}</h2>")
        self._intro.setText(
            _(
                "Typical order: Gather → Convert → Deduplicate (optional) → Upscale (optional) → Create Dataset. "
                "If class folders are already under your raw data root, skip Gather and start at Convert. "
                "See docs/DATA_PIPELINE.md for storage on external drives and large vs small images."
            )
        )
        self.tabs.setTabText(0, _("Gather"))
        self.tabs.setTabText(1, _("Convert"))
        self.tabs.setTabText(2, _("Deduplicate"))
        self.tabs.setTabText(3, _("Upscale"))
        self.tabs.setTabText(4, _("Create Dataset"))
        self.btn_validate.setText(_("Validate Inputs"))
        self.btn_run.setText(_("Run Data Command"))
        self.output.setPlaceholderText(_("Validation and run results will appear here."))
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
            [_("Raw data dir"), _("Format (jpeg/jpg)"), ""],
        )
        apply_qform_label_column(self._dedup_form, [_("Raw data dir")])
        apply_qform_label_column(
            self._upscale_form,
            [_("Raw data dir"), _("Review dir (optional)")],
        )
        apply_qform_label_column(
            self._dataset_form,
            [
                _("Raw data dir"),
                _("Output data dir"),
                _("Test items per class"),
                _("Seed (optional)"),
                _("Run ID (optional)"),
                _("Max train per class"),
                "",
                "",
            ],
        )
        self.dataset_max_train.setSpecialValueText(_("None"))
        self.dataset_balance_train.setText(_("Balance train set to smallest class"))
        self.dataset_allow_external.setText(_("Allow external/removable storage"))
        self._apply_pipeline_tab_tooltips()
        self._apply_pipeline_group_tooltips()
        self._apply_intro_tooltip()
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

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        return {
            "tab": int(self.tabs.currentIndex()),
            "gather": {
                "source": self.gather_source.text(),
                "subdirs": self.gather_subdirs.text(),
                "target_count": int(self.gather_target_count.value()),
                "target_dir": self.gather_target_dir.text(),
                "rejected_dir": self.gather_rejected_dir.text(),
                "subdir_weights": self.gather_subdir_weights.text(),
                "raw_data_dir": self.gather_raw_data_dir.text(),
            },
            "convert": {
                "raw_data_dir": self.convert_raw_data_dir.text(),
                "format": self.convert_format.text(),
                "skip_space": bool(self.convert_skip_space.isChecked()),
            },
            "dedup": {"raw_data_dir": self.dedup_raw_data_dir.text()},
            "upscale": {
                "raw_data_dir": self.upscale_raw_data_dir.text(),
                "review_dir": self.upscale_review_dir.text(),
            },
            "dataset": {
                "raw_data_dir": self.dataset_raw_data_dir.text(),
                "data_dir": self.dataset_data_dir.text(),
                "test_per_class": int(self.dataset_test_per_class.value()),
                "seed": int(self.dataset_seed.value()),
                "run_id": self.dataset_run_id.text(),
                "max_train": int(self.dataset_max_train.value()),
                "balance_train": bool(self.dataset_balance_train.isChecked()),
                "allow_external": bool(self.dataset_allow_external.isChecked()),
                "skip_space": bool(self.dataset_skip_space.isChecked()),
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
                self.gather_raw_data_dir.setText(str(g.get("raw_data_dir", "")))

            c = state.get("convert") or {}
            if isinstance(c, dict):
                self.convert_raw_data_dir.setText(str(c.get("raw_data_dir", "")))
                self.convert_format.setText(str(c.get("format", "jpeg")))
                self.convert_skip_space.setChecked(bool(c.get("skip_space", False)))

            d = state.get("dedup") or {}
            if isinstance(d, dict):
                self.dedup_raw_data_dir.setText(str(d.get("raw_data_dir", "")))

            u = state.get("upscale") or {}
            if isinstance(u, dict):
                self.upscale_raw_data_dir.setText(str(u.get("raw_data_dir", "")))
                self.upscale_review_dir.setText(str(u.get("review_dir", "")))

            ds = state.get("dataset") or {}
            if isinstance(ds, dict):
                self.dataset_raw_data_dir.setText(str(ds.get("raw_data_dir", "")))
                self.dataset_data_dir.setText(str(ds.get("data_dir", "")))
                tpc = ds.get("test_per_class")
                if isinstance(tpc, int):
                    self.dataset_test_per_class.setValue(tpc)
                sd = ds.get("seed")
                if isinstance(sd, int):
                    self.dataset_seed.setValue(sd)
                self.dataset_run_id.setText(str(ds.get("run_id", "")))
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
        form.addRow(_("Raw data dir"), self._path_row(self.convert_raw_data_dir, select_dir=True))
        form.addRow(_("Format (jpeg/jpg)"), self.convert_format)
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
        v.addWidget(self._dedup_group)
        v.addStretch(1)
        return tab

    def _build_upscale_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

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

        self._dataset_group = QGroupBox("mb data create-dataset")
        form = QFormLayout(self._dataset_group)
        self._dataset_form = form
        self.dataset_raw_data_dir = QLineEdit("raw_data")
        self.dataset_data_dir = QLineEdit("data")
        self.dataset_test_per_class = QSpinBox()
        self.dataset_test_per_class.setRange(1, 1_000_000)
        self.dataset_test_per_class.setValue(1000)
        self.dataset_seed = QSpinBox()
        self.dataset_seed.setRange(0, 2_147_483_647)
        self.dataset_run_id = QLineEdit()
        self.dataset_max_train = QSpinBox()
        self.dataset_max_train.setRange(0, 1_000_000)
        self.dataset_max_train.setSpecialValueText(_("None"))
        self.dataset_balance_train = QCheckBox(_("Balance train set to smallest class"))
        self.dataset_allow_external = QCheckBox(_("Allow external/removable storage"))

        form.addRow(_("Raw data dir"), self._path_row(self.dataset_raw_data_dir, select_dir=True))
        form.addRow(_("Output data dir"), self._path_row(self.dataset_data_dir, select_dir=True))
        form.addRow(_("Test items per class"), self.dataset_test_per_class)
        form.addRow(_("Seed (optional)"), self.dataset_seed)
        form.addRow(_("Run ID (optional)"), self.dataset_run_id)
        form.addRow(_("Max train per class"), self.dataset_max_train)
        form.addRow("", self.dataset_balance_train)
        form.addRow("", self.dataset_allow_external)
        self.dataset_skip_space = QCheckBox(
            _("Skip free-space check on output drive (not recommended)")
        )
        form.addRow("", self.dataset_skip_space)
        v.addWidget(self._dataset_group)
        v.addStretch(1)
        return tab

    def _apply_pipeline_tab_tooltips(self) -> None:
        """Hover hints for each Data tab (native tab bar tooltips)."""
        tips = [
            _(
                "Copy samples from source subdirectories into class folders under the raw data root. "
                "Skip this tab if your data is already organized as one folder per class."
            ),
            _(
                "Normalize images to JPEG under each class’s CONVERTED/ folder. "
                "Point Raw data dir at the same tree on an internal or external drive; outputs stay beside sources."
            ),
            _(
                "Remove duplicates and quarantine very small images (by pixel size) to small_images_review/ under the raw root."
            ),
            _(
                "Upscale images previously moved to the small-image review area (after deduplicate). "
                "Default review dir is raw_data/small_images_review when left empty."
            ),
            _(
                "Build train/ and test/ under the output data directory from the raw tree. "
                "A good place to put the final dataset on your main drive while raw data stays external."
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
        ]
        texts = [
            _(
                "Gather copies files into timestamped runs under the target dir and tracks hashes so reruns skip "
                "already-seen files. Set raw data dir to the pipeline root used by later steps."
            ),
            _(
                "Convert walks class folders, writes CONVERTED/ JPEGs, and records a unified snapshot. "
                "For image classification, videos and multi-frame GIFs may yield a random frame JPEG. "
                "Very large images are downscaled (max edge 4000px)."
            ),
            _(
                "Deduplicate removes duplicate images and handles tiny dimensions: removes very small images and "
                "moves borderline-small ones to small_images_review/ for manual review before upscale."
            ),
            _(
                "Upscale processes the review tree produced by deduplicate for undersized images you choose to keep."
            ),
            _(
                "Create-dataset reads from raw data (including CONVERTED/) and writes the train/test split to the "
                "output data dir. Set output on a spacious internal disk if raw data lives on external storage."
            ),
        ]
        if self._pipeline_group_tooltips is None:
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
            "small_images_review/. Full discussion: docs/DATA_PIPELINE.md."
        )
        if self._intro_tooltip is None:
            self._intro_tooltip = create_tooltip(self._intro, detail)
        else:
            self._intro_tooltip.set_text(detail)

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

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_validate.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and self._can_run())
        self.tabs.setEnabled(not busy)

    def _current_command(self) -> ModelBuildStepCommand:
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
            return {
                "raw_data_dir": Path(self.convert_raw_data_dir.text().strip() or "raw_data"),
                "format": fmt,
                "skip_space_check": bool(self.convert_skip_space.isChecked()),
            }
        if command == ModelBuildStepCommand.DEDUPLICATE:
            return {"raw_data_dir": Path(self.dedup_raw_data_dir.text().strip() or "raw_data")}
        if command == ModelBuildStepCommand.UPSCALE:
            review_raw = self.upscale_review_dir.text().strip()
            return {
                "raw_data_dir": Path(self.upscale_raw_data_dir.text().strip() or "raw_data"),
                "review_dir": Path(review_raw) if review_raw else None,
            }
        if command == ModelBuildStepCommand.CREATE_DATASET:
            raw_data_dir = Path(self.dataset_raw_data_dir.text().strip() or "raw_data")
            data_dir = Path(self.dataset_data_dir.text().strip() or "data")
            if not raw_data_dir.exists():
                raise ValueError(_("Raw data dir does not exist."))
            return {
                "raw_data_dir": raw_data_dir,
                "data_dir": data_dir,
                "test_per_class": int(self.dataset_test_per_class.value()),
                "seed": int(self.dataset_seed.value()) if self.dataset_seed.value() > 0 else None,
                "run_id": self.dataset_run_id.text().strip() or None,
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
                )
            )
        if command == ModelBuildStepCommand.DEDUPLICATE:
            deduplicator = ImageDeduplicator(raw_data_dir=payload["raw_data_dir"])
            return bool(deduplicator.run(cancel_event=ce))
        if command == ModelBuildStepCommand.UPSCALE:
            review_dir = payload["review_dir"] or (payload["raw_data_dir"] / "small_images_review")
            upscaler = ImageUpscaler(review_dir=review_dir)
            return bool(upscaler.run(cancel_event=ce))
        if command == ModelBuildStepCommand.CREATE_DATASET:
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
            )
            return bool(creator.run(cancel_event=ce))
        raise ValueError(_("Unknown command: {cmd}").format(cmd=command.value))

    def _on_run_success(self, success: bool) -> None:
        summary = getattr(self, "_pending_run_summary", "mb data")
        sub = getattr(self, "_pending_data_subcommand", None)
        snap = self._snapshot_path_after_successful_data_run() if success else None
        if success:
            self._append(_("[done] Data command completed successfully."))
            if snap:
                self._append(f"[snapshot] {snap}")
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
