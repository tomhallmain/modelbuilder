"""Data operations page mirroring `mb data` subcommands."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
from mb.utils.storage import check_same_drive, check_target_external_storage
from ui.lib.qt_alert import qt_alert, qt_operation_error
from mb.utils.constants import DataPipelineSubcommand, ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.translations import _
from ui.lib.task_progress import attach_progress_dialog
from ui.task_context import LongTaskContext
from ui.task_runner import start_task

logger = logging.getLogger(__name__)


class DataPage(QWidget):
    """UI scaffold for data command forms."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel(f"<h2>{_('Data')}</h2>")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                _("Prepare datasets using gather, convert, deduplicate, upscale, and split flows.")
            )
        )

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_gather_tab(), _("Gather"))
        self.tabs.addTab(self._build_convert_tab(), _("Convert"))
        self.tabs.addTab(self._build_dedup_tab(), _("Deduplicate"))
        self.tabs.addTab(self._build_upscale_tab(), _("Upscale"))
        self.tabs.addTab(self._build_dataset_tab(), _("Create Dataset"))
        root.addWidget(self.tabs, 1)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton(_("Validate Inputs"))
        self.btn_run = QPushButton(_("Run Data Command"))
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_run)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(_("Validation and run results will appear here."))
        root.addWidget(self.output, 1)

        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_run.clicked.connect(self._run_current_command)
        self.tabs.currentChanged.connect(self._validate_inputs)

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
        except Exception:
            pass
        finally:
            self.tabs.blockSignals(False)

    def _build_gather_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data gather")
        form = QFormLayout(group)
        self.gather_source = QLineEdit()
        self.gather_subdirs = QLineEdit()
        self.gather_target_count = QSpinBox()
        self.gather_target_count.setRange(1, 5_000_000)
        self.gather_target_count.setValue(16000)
        self.gather_target_dir = QLineEdit("raw_data/coherent")
        self.gather_rejected_dir = QLineEdit()
        self.gather_subdir_weights = QLineEdit()
        self.gather_raw_data_dir = QLineEdit("raw_data")

        form.addRow(_("Source dir"), self._path_row(self.gather_source, select_dir=True))
        form.addRow(_("Subdirs (space-separated)"), self.gather_subdirs)
        form.addRow(_("Target count"), self.gather_target_count)
        form.addRow(_("Target dir"), self._path_row(self.gather_target_dir, select_dir=True))
        form.addRow(_("Rejected dir"), self._path_row(self.gather_rejected_dir, select_dir=True))
        form.addRow(_("Subdir weights"), self.gather_subdir_weights)
        form.addRow(_("Raw data dir"), self._path_row(self.gather_raw_data_dir, select_dir=True))
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_convert_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data convert")
        form = QFormLayout(group)
        self.convert_raw_data_dir = QLineEdit("raw_data")
        self.convert_format = QLineEdit("jpeg")
        form.addRow(_("Raw data dir"), self._path_row(self.convert_raw_data_dir, select_dir=True))
        form.addRow(_("Format (jpeg/jpg)"), self.convert_format)
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_dedup_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data deduplicate")
        form = QFormLayout(group)
        self.dedup_raw_data_dir = QLineEdit("raw_data")
        form.addRow(_("Raw data dir"), self._path_row(self.dedup_raw_data_dir, select_dir=True))
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_upscale_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data upscale")
        form = QFormLayout(group)
        self.upscale_raw_data_dir = QLineEdit("raw_data")
        self.upscale_review_dir = QLineEdit()
        form.addRow(_("Raw data dir"), self._path_row(self.upscale_raw_data_dir, select_dir=True))
        form.addRow(
            _("Review dir (optional)"),
            self._path_row(self.upscale_review_dir, select_dir=True),
        )
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data create-dataset")
        form = QFormLayout(group)
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
        form.addRow(_("Test images per class"), self.dataset_test_per_class)
        form.addRow(_("Seed (optional)"), self.dataset_seed)
        form.addRow(_("Run ID (optional)"), self.dataset_run_id)
        form.addRow(_("Max train per class"), self.dataset_max_train)
        form.addRow("", self.dataset_balance_train)
        form.addRow("", self.dataset_allow_external)
        v.addWidget(group)
        v.addStretch(1)
        return tab

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
            value = QFileDialog.getExistingDirectory(
                self,
                _("Select directory"),
                start,
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        else:
            value, _ = QFileDialog.getOpenFileName(
                self,
                _("Select file"),
                start,
                _("All files (*.*)"),
                options=QFileDialog.Option.DontUseNativeDialog,
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

    def _current_command(self) -> str:
        return ["gather", "convert", "deduplicate", "upscale", "create-dataset"][self.tabs.currentIndex()]

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
            self._append(_("[ok] {cmd}: inputs look valid").format(cmd=command))
        except ValueError as exc:
            self.btn_run.setEnabled(False)
            self.btn_run.setToolTip(str(exc))
            self._append(_("[invalid] {cmd}: {err}").format(cmd=command, err=exc))

    def _collect_inputs(self, command: str) -> dict:
        if command == "gather":
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
                "target_dir": Path(self.gather_target_dir.text().strip() or "raw_data/coherent"),
                "rejected_dir": Path(self.gather_rejected_dir.text().strip()) if self.gather_rejected_dir.text().strip() else None,
                "subdir_weights": weights if weights else None,
                "raw_data_dir": Path(self.gather_raw_data_dir.text().strip() or "raw_data"),
            }
        if command == "convert":
            fmt = self.convert_format.text().strip().lower() or "jpeg"
            if fmt not in {"jpeg", "jpg"}:
                raise ValueError(_("Format must be jpeg or jpg."))
            return {
                "raw_data_dir": Path(self.convert_raw_data_dir.text().strip() or "raw_data"),
                "format": fmt,
            }
        if command == "deduplicate":
            return {"raw_data_dir": Path(self.dedup_raw_data_dir.text().strip() or "raw_data")}
        if command == "upscale":
            review_raw = self.upscale_review_dir.text().strip()
            return {
                "raw_data_dir": Path(self.upscale_raw_data_dir.text().strip() or "raw_data"),
                "review_dir": Path(review_raw) if review_raw else None,
            }
        if command == "create-dataset":
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
            }
        raise ValueError(_("Unknown command: {cmd}").format(cmd=command))

    def _run_current_command(self) -> None:
        command = self._current_command()
        payload = self._collect_inputs(command)

        if command == "create-dataset":
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
                return
            if check_same_drive(payload["raw_data_dir"], payload["data_dir"]):
                if not qt_alert(
                    self,
                    _("Confirm"),
                    _("Source and target are on the same drive. Continue dataset creation?"),
                    kind="askyesno",
                ):
                    return

        self._append(f"[run] mb data {command}")
        self._pending_run_summary = f"mb data {command}"
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
        attach_progress_dialog(self, _("Data: {cmd}").format(cmd=command), handle, cancellable=True)

    def _execute_data_command(self, ctx: LongTaskContext, command: str, payload: dict) -> bool:
        ce = ctx.cancel_event
        if command == "gather":
            gatherer = ImageGatherer(
                source_dir=str(payload["source_dir"]),
                valid_subdirs=payload["subdirs"],
                target_dir=payload["target_dir"],
                target_count=payload["target_count"],
                rejected_dir=payload["rejected_dir"],
                subdir_weights=payload["subdir_weights"],
            )
            gatherer.raw_data_dir = payload["raw_data_dir"]
            return bool(gatherer.run(cancel_event=ce))
        if command == "convert":
            converter = ImageConverter(raw_data_dir=payload["raw_data_dir"])
            return bool(converter.run(cancel_event=ce))
        if command == "deduplicate":
            deduplicator = ImageDeduplicator(raw_data_dir=payload["raw_data_dir"])
            return bool(deduplicator.run(cancel_event=ce))
        if command == "upscale":
            review_dir = payload["review_dir"] or (payload["raw_data_dir"] / "small_images_review")
            upscaler = ImageUpscaler(review_dir=review_dir)
            return bool(upscaler.run(cancel_event=ce))
        if command == "create-dataset":
            if payload["seed"] is not None:
                random.seed(payload["seed"])
            creator = DatasetCreator(
                raw_data_dir=payload["raw_data_dir"],
                data_dir=payload["data_dir"],
                test_images_per_class=payload["test_per_class"],
                balance_train=payload["balance_train"],
                max_train_per_class=payload["max_train_per_class"],
                run_id=payload["run_id"],
            )
            return bool(creator.run(cancel_event=ce))
        raise ValueError(_("Unknown command: {cmd}").format(cmd=command))

    def _on_run_success(self, success: bool) -> None:
        summary = getattr(self, "_pending_run_summary", "mb data")
        sub = getattr(self, "_pending_data_subcommand", None)
        if success:
            self._append(_("[done] Data command completed successfully."))
            append_recent_run(ModelBuilderTaskType.DATA, summary, True, data_subcommand=sub)
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
