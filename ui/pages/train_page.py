"""Training page scaffold aligned with `mb train` arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.models.generation_architectures import BaseGenerationArchitecture
from mb.models.types import ArchitectureType, FrameworkType, ModelType
from mb.pipeline_config import get_pipeline_config
from mb.training.lora_diffusion_trainer import LoraTrainingConfig
from mb.training.run_args import TrainingRunArgs
from mb.utils.logging_setup import setup_logging
from ui.controllers.model_type_field_visibility import apply_model_type_field_visibility
from ui.controllers.train_page_controller import TrainPageFieldValues, build_training_request
from ui.lib.fast_directory_picker_qt import get_existing_directory, get_open_file_name
from ui.lib.qt_alert import qt_alert, qt_operation_error
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.snapshot import find_latest_unified_snapshot_path, run_id_from_latest_unified_snapshot
from mb.utils.translations import _
from ui.lib.training_progress_dialog import attach_training_progress_dialog
from ui.spawn_mb_train import spawn_mb_train_subprocess
from ui.task_context import LongTaskContext
from ui.task_runner import start_task
from ui.workspace import Workspace, default_settings, effective_pipeline_config_path
from ui.lib.form_layout_i18n import apply_qform_label_column


class TrainPage(QWidget):
    """UI scaffold for training configuration."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("train_page")
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        self._head.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        core_group = QGroupBox()
        self._core_group = core_group
        core_form = QFormLayout(core_group)
        self._core_form = core_form
        self.model_type = QComboBox()
        for mt in ModelType:
            self.model_type.addItem(mt.value)
        self.framework = QComboBox()
        # ``registered_values`` returns a set-like container; sort for deterministic UI order.
        self.framework.addItems(sorted(FrameworkType.registered_values()))
        self.architecture = QLineEdit(ArchitectureType.get_default().value)
        self.architecture.setObjectName("train_architecture_edit")
        self.data_dir = QLineEdit("data")
        self.data_dir.setObjectName("train_data_dir_edit")
        self.output_dir = QLineEdit("data/models")
        self.output_dir.setObjectName("train_output_dir_edit")
        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 8192)
        self.batch_size.setSpecialValueText(_("Auto"))
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setValue(224)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 128)
        self.num_workers.setSpecialValueText(_("Config default"))
        self.resume_from = QLineEdit()
        self.run_id = QLineEdit()
        train_run_id_row, self.btn_run_id_latest = self._run_id_row(
            self.run_id, self._on_train_use_latest_run_id
        )
        self.skip_snapshot = QCheckBox(_("Skip unified snapshot update"))

        core_form.addRow(_("Model type"), self.model_type)
        core_form.addRow(_("Framework"), self.framework)
        core_form.addRow(_("Architecture"), self.architecture)
        core_form.addRow(_("Data dir"), self._path_row(self.data_dir, select_file=False))
        core_form.addRow(_("Output dir"), self._path_row(self.output_dir, select_file=False))
        core_form.addRow(_("Batch size"), self.batch_size)
        core_form.addRow(_("Image size"), self.image_size)
        core_form.addRow(_("Num workers"), self.num_workers)
        resume_from_row = self._path_row(self.resume_from, select_file=True)
        core_form.addRow(_("Resume checkpoint"), resume_from_row)
        core_form.addRow(_("Run ID (optional)"), train_run_id_row)
        core_form.addRow("", self.skip_snapshot)
        self.train_subprocess = QCheckBox(
            _("Run training in a separate process (survives closing this app; see log file)")
        )
        core_form.addRow("", self.train_subprocess)
        root.addWidget(core_group)

        # Rows only meaningful for image_classification (LoRA has no resume/snapshot/detached-
        # subprocess support yet — mb train --train-args-json rejects image_generation_lora).
        # QFormLayout.setRowVisible needs the exact widget passed to addRow — for wrapped rows
        # (browse-button path rows, the run-id-with-"Latest"-button row) that's the wrapper,
        # not the QLineEdit inside it.
        self._classification_only_core_rows = (
            resume_from_row,
            train_run_id_row,
            self.skip_snapshot,
            self.train_subprocess,
        )

        hp_group = QGroupBox()
        self._hp_group = hp_group
        hp_form = QFormLayout(hp_group)
        self._hp_form = hp_form
        self.frozen_epochs = QSpinBox()
        self.frozen_epochs.setRange(0, 10000)
        self.frozen_epochs.setValue(5)
        self.unfrozen_epochs = QSpinBox()
        self.unfrozen_epochs.setRange(0, 10000)
        self.unfrozen_epochs.setValue(20)
        self.frozen_lr = self._lr_spin(0.001)
        self.unfrozen_lr_max = self._lr_spin(0.0003)
        self.unfrozen_lr_min = self._lr_spin(0.00001)

        hp_form.addRow(_("Frozen epochs"), self.frozen_epochs)
        hp_form.addRow(_("Unfrozen epochs"), self.unfrozen_epochs)
        hp_form.addRow(_("Frozen LR"), self.frozen_lr)
        hp_form.addRow(_("Unfrozen LR max"), self.unfrozen_lr_max)
        hp_form.addRow(_("Unfrozen LR min"), self.unfrozen_lr_min)
        root.addWidget(hp_group)

        lora_group = QGroupBox()
        self._lora_group = lora_group
        lora_form = QFormLayout(lora_group)
        self._lora_form = lora_form
        self.base_model_architecture = QComboBox()
        self.base_model_architecture.addItem(_("Auto-detect"), "")
        for arch in BaseGenerationArchitecture:
            self.base_model_architecture.addItem(arch.value, arch.value)
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 1024)
        self.lora_rank.setValue(16)
        self.lora_alpha = QSpinBox()
        self.lora_alpha.setRange(0, 1024)
        self.lora_alpha.setSpecialValueText(_("Same as rank"))
        self.learning_rate = self._lr_spin(1e-4)
        self.max_train_steps = QSpinBox()
        self.max_train_steps.setRange(1, 1_000_000)
        self.max_train_steps.setValue(1000)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText(_("optional, integer"))

        lora_form.addRow(_("Base model architecture"), self.base_model_architecture)
        lora_form.addRow(_("LoRA rank"), self.lora_rank)
        lora_form.addRow(_("LoRA alpha"), self.lora_alpha)
        lora_form.addRow(_("Learning rate"), self.learning_rate)
        lora_form.addRow(_("Max train steps"), self.max_train_steps)
        lora_form.addRow(_("Seed"), self.seed)
        root.addWidget(lora_group)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton(_("Validate Training Config"))
        self.btn_validate.setObjectName("train_validate_btn")
        self.btn_start = QPushButton(_("Start Training"))
        self.btn_start.setObjectName("train_start_btn")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_start)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setObjectName("train_output_log")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(
            _("Training validation and execution messages will appear here.")
        )
        root.addWidget(self.output, 1)

        self.model_type.currentIndexChanged.connect(self._on_model_type_changed)
        self.framework.currentTextChanged.connect(self._refresh_architecture_hint)
        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_start.clicked.connect(self._start_training)

        self.retranslate_ui(refresh_output=False)
        # Visibility only here — MainWindow unconditionally calls _run_startup_validation()
        # right after construction (post cache-restore), which refreshes the architecture
        # hint and validates; doing that here too would construct the (potentially
        # heavyweight, e.g. PyTorch/Keras) architecture-listing trainer redundantly.
        self._apply_model_type_visibility()

    def retranslate_ui(self, *, refresh_output: bool = True) -> None:
        self._head.setText(f"<h2>{_('Train')}</h2>")
        self._intro.setText(
            _("Configure framework, architecture, paths, and hyperparameters (training CLI: {cmd}).").format(
                cmd="mb train"
            )
        )
        self._core_group.setTitle(_("Core configuration"))
        apply_qform_label_column(
            self._core_form,
            [
                _("Model type"),
                _("Framework"),
                _("Architecture"),
                _("Data dir"),
                _("Output dir"),
                _("Batch size"),
                _("Image size"),
                _("Num workers"),
                _("Resume checkpoint"),
                _("Run ID (optional)"),
                "",
                "",
            ],
        )
        self.skip_snapshot.setText(_("Skip unified snapshot update"))
        self.train_subprocess.setText(
            _("Run training in a separate process (survives closing this app; see log file)")
        )
        self._hp_group.setTitle(_("Hyperparameters (image classification)"))
        apply_qform_label_column(
            self._hp_form,
            [
                _("Frozen epochs"),
                _("Unfrozen epochs"),
                _("Frozen LR"),
                _("Unfrozen LR max"),
                _("Unfrozen LR min"),
            ],
        )
        self._lora_group.setTitle(_("Hyperparameters (image generation LoRA)"))
        apply_qform_label_column(
            self._lora_form,
            [
                _("Base model architecture"),
                _("LoRA rank"),
                _("LoRA alpha"),
                _("Learning rate"),
                _("Max train steps"),
                _("Seed"),
            ],
        )
        self.base_model_architecture.setItemText(0, _("Auto-detect"))
        self.lora_alpha.setSpecialValueText(_("Same as rank"))
        self.seed.setPlaceholderText(_("optional, integer"))
        self.batch_size.setSpecialValueText(_("Auto"))
        self.num_workers.setSpecialValueText(_("Config default"))
        self.btn_validate.setText(_("Validate Training Config"))
        self.btn_start.setText(_("Start Training"))
        self.output.setPlaceholderText(
            _("Training validation and execution messages will appear here.")
        )
        self.btn_run_id_latest.setText(_("Latest"))
        self.btn_run_id_latest.setToolTip(
            _(
                "Set Run ID from the newest snapshot_*.json under the data directory "
                "(by file modification time)."
            )
        )
        for edit, sel_file in (
            (self.data_dir, False),
            (self.output_dir, False),
            (self.resume_from, True),
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
        self._refresh_architecture_hint()
        self._validate_inputs()

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        return {
            "model_type_idx": int(self.model_type.currentIndex()),
            "model_type_value": self.model_type.currentText(),
            "framework_idx": int(self.framework.currentIndex()),
            "framework_value": self.framework.currentText(),
            "architecture": self.architecture.text(),
            "data_dir": self.data_dir.text(),
            "output_dir": self.output_dir.text(),
            "resume_from": self.resume_from.text(),
            "run_id": self.run_id.text(),
            "skip_snapshot": bool(self.skip_snapshot.isChecked()),
            "train_subprocess": bool(self.train_subprocess.isChecked()),
            "frozen_epochs": int(self.frozen_epochs.value()),
            "unfrozen_epochs": int(self.unfrozen_epochs.value()),
            "frozen_lr": float(self.frozen_lr.value()),
            "unfrozen_lr_max": float(self.unfrozen_lr_max.value()),
            "unfrozen_lr_min": float(self.unfrozen_lr_min.value()),
            "batch_size": int(self.batch_size.value()),
            "image_size": int(self.image_size.value()),
            "num_workers": int(self.num_workers.value()),
            "base_model_architecture_idx": int(self.base_model_architecture.currentIndex()),
            "base_model_architecture_data": self.base_model_architecture.currentData(),
            "lora_rank": int(self.lora_rank.value()),
            "lora_alpha": int(self.lora_alpha.value()),
            "learning_rate": float(self.learning_rate.value()),
            "max_train_steps": int(self.max_train_steps.value()),
            "seed": self.seed.text(),
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        self.framework.blockSignals(True)
        try:
            mtv = state.get("model_type_value")
            if isinstance(mtv, str):
                mtix = self.model_type.findText(mtv)
                if mtix >= 0:
                    self.model_type.setCurrentIndex(mtix)
                else:
                    mti = state.get("model_type_idx")
                    if isinstance(mti, int) and 0 <= mti < self.model_type.count():
                        self.model_type.setCurrentIndex(mti)
            else:
                mti = state.get("model_type_idx")
                if isinstance(mti, int) and 0 <= mti < self.model_type.count():
                    self.model_type.setCurrentIndex(mti)
            fv = state.get("framework_value")
            if isinstance(fv, str):
                fix = self.framework.findText(fv)
                if fix >= 0:
                    self.framework.setCurrentIndex(fix)
                else:
                    fi = state.get("framework_idx")
                    if isinstance(fi, int) and 0 <= fi < self.framework.count():
                        self.framework.setCurrentIndex(fi)
            else:
                fi = state.get("framework_idx")
                if isinstance(fi, int) and 0 <= fi < self.framework.count():
                    self.framework.setCurrentIndex(fi)
            self.architecture.setText(str(state.get("architecture", "")))
            self.data_dir.setText(str(state.get("data_dir", "")))
            self.output_dir.setText(str(state.get("output_dir", "")))
            self.resume_from.setText(str(state.get("resume_from", "")))
            self.run_id.setText(str(state.get("run_id", "")))
            self.skip_snapshot.setChecked(bool(state.get("skip_snapshot", False)))
            self.train_subprocess.setChecked(bool(state.get("train_subprocess", False)))
            for key, spin in (
                ("frozen_epochs", self.frozen_epochs),
                ("unfrozen_epochs", self.unfrozen_epochs),
            ):
                v = state.get(key)
                if isinstance(v, int):
                    spin.setValue(v)
            for key, spin in (
                ("frozen_lr", self.frozen_lr),
                ("unfrozen_lr_max", self.unfrozen_lr_max),
                ("unfrozen_lr_min", self.unfrozen_lr_min),
            ):
                v = state.get(key)
                if isinstance(v, (int, float)):
                    spin.setValue(float(v))
            for key, spin in (
                ("batch_size", self.batch_size),
                ("image_size", self.image_size),
                ("num_workers", self.num_workers),
            ):
                v = state.get(key)
                if isinstance(v, int):
                    spin.setValue(v)
            bma_data = state.get("base_model_architecture_data")
            if isinstance(bma_data, str):
                bix = self.base_model_architecture.findData(bma_data)
                if bix >= 0:
                    self.base_model_architecture.setCurrentIndex(bix)
            else:
                bmi = state.get("base_model_architecture_idx")
                if isinstance(bmi, int) and 0 <= bmi < self.base_model_architecture.count():
                    self.base_model_architecture.setCurrentIndex(bmi)
            for key, spin in (
                ("lora_rank", self.lora_rank),
                ("lora_alpha", self.lora_alpha),
                ("max_train_steps", self.max_train_steps),
            ):
                v = state.get(key)
                if isinstance(v, int):
                    spin.setValue(v)
            lr = state.get("learning_rate")
            if isinstance(lr, (int, float)):
                self.learning_rate.setValue(float(lr))
            self.seed.setText(str(state.get("seed", "")))
        except Exception:
            pass
        finally:
            self.framework.blockSignals(False)
        # Visibility only — MainWindow calls _run_startup_validation() right after
        # restore, which already refreshes the architecture hint and validates.
        self._apply_model_type_visibility()

    def _path_row(self, edit: QLineEdit, select_file: bool = False) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, select_file=select_file))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

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

    def _on_train_use_latest_run_id(self) -> None:
        data_dir = Path(self.data_dir.text().strip() or "data")
        rid = run_id_from_latest_unified_snapshot([data_dir], quiet=True)
        if not rid:
            qt_alert(
                self,
                _("No snapshot found"),
                _("No snapshot_*.json files were found under:\n{path}").format(path=data_dir),
                kind="warning",
            )
            return
        self.run_id.setText(rid)
        self._validate_inputs()

    def _lr_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(8)
        spin.setRange(0.0, 10.0)
        spin.setSingleStep(0.0001)
        spin.setValue(value)
        return spin

    def _browse(self, edit: QLineEdit, select_file: bool = False) -> None:
        start = edit.text().strip()
        if not start and select_file and edit is self.resume_from:
            start = self.output_dir.text().strip()
        if not start:
            start = str(Path.cwd())
        if select_file:
            value = get_open_file_name(
                self,
                _("Select checkpoint file"),
                start,
                _("Model/checkpoint files (*.pth *.pt *.h5 *.keras *.ckpt);;All files (*.*)"),
            )
            if value:
                edit.setText(value)
        else:
            value = get_existing_directory(
                self,
                _("Select directory"),
                start,
            )
            if value:
                edit.setText(value)
        self._validate_inputs()

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_validate.setEnabled(not busy)
        self.btn_start.setEnabled(not busy and self._can_run())

    def _current_model_type(self) -> ModelType:
        return ModelType.from_pipeline_value(self.model_type.currentText())

    def _apply_model_type_visibility(self) -> None:
        model_type = self._current_model_type()
        is_lora = model_type == ModelType.IMAGE_GENERATION_LORA
        self._hp_group.setVisible(not is_lora)
        self._lora_group.setVisible(is_lora)
        apply_model_type_field_visibility(
            self._core_form,
            model_type,
            {
                widget: (
                    ModelType.IMAGE_CLASSIFICATION,
                    ModelType.OBJECT_DETECTION,
                    ModelType.IMAGE_GENERATION,
                )
                for widget in self._classification_only_core_rows
            },
        )

    def _on_model_type_changed(self) -> None:
        """Live combo-change handler: visibility + a fresh architecture hint + re-validate.

        Only call this for an actual user-driven change. ``restore_gui_state`` calls
        :meth:`_apply_model_type_visibility` instead — ``MainWindow`` runs
        :meth:`_run_startup_validation` (which already refreshes the hint and validates)
        right after restoring, so calling this full handler there would construct the
        (potentially heavyweight, e.g. PyTorch/Keras) architecture-listing trainer twice.
        """
        self._apply_model_type_visibility()
        self._refresh_architecture_hint()
        self._validate_inputs()

    def _refresh_architecture_hint(self) -> None:
        if self._current_model_type() == ModelType.IMAGE_GENERATION_LORA:
            self.architecture.setPlaceholderText(
                _("local checkpoint path or hub id, e.g. black-forest-labs/FLUX.1-dev")
            )
            return
        framework = self.framework.currentText()
        fw = FrameworkType.try_from(framework)
        if fw is None:
            self.architecture.setPlaceholderText(_("Enter architecture manually"))
            self._append(
                _("[warn] Unsupported framework for architecture list: {fw}").format(fw=framework)
            )
            return
        try:
            from mb.training.trainer import ModelTrainer

            trainer = ModelTrainer(
                framework=fw,
                model_type=ModelType.IMAGE_CLASSIFICATION,
                pipeline_config=get_pipeline_config(),
            )
            architectures = trainer.get_supported_architectures()
            if architectures:
                self.architecture.setPlaceholderText(", ".join(architectures[:8]))
                self._append(
                    _("[info] {fw} architectures: {names}").format(
                        fw=fw.value, names=", ".join(architectures)
                    )
                )
        except Exception as exc:
            self.architecture.setPlaceholderText(_("Enter architecture manually"))
            self._append(
                _("[warn] Could not load architecture list for {fw}: {err}").format(
                    fw=framework, err=exc
                )
            )

    def _can_run(self) -> bool:
        try:
            self._collect_inputs()
            return True
        except ValueError:
            return False

    def _collect_inputs(self) -> Union[TrainingRunArgs, LoraTrainingConfig]:
        values = TrainPageFieldValues(
            model_type=self._current_model_type(),
            framework_text=self.framework.currentText(),
            architecture_text=self.architecture.text(),
            data_dir_text=self.data_dir.text(),
            output_dir_text=self.output_dir.text(),
            batch_size=int(self.batch_size.value()),
            image_size=int(self.image_size.value()),
            num_workers=int(self.num_workers.value()),
            resume_from_text=self.resume_from.text(),
            run_id_text=self.run_id.text(),
            skip_snapshot=bool(self.skip_snapshot.isChecked()),
            frozen_epochs=int(self.frozen_epochs.value()),
            unfrozen_epochs=int(self.unfrozen_epochs.value()),
            frozen_lr=float(self.frozen_lr.value()),
            unfrozen_lr_max=float(self.unfrozen_lr_max.value()),
            unfrozen_lr_min=float(self.unfrozen_lr_min.value()),
            base_model_architecture_text=str(self.base_model_architecture.currentData() or ""),
            lora_rank=int(self.lora_rank.value()),
            lora_alpha=int(self.lora_alpha.value()),
            learning_rate=float(self.learning_rate.value()),
            max_train_steps=int(self.max_train_steps.value()),
            seed_text=self.seed.text(),
        )
        return build_training_request(values)

    def _validate_inputs(self) -> None:
        try:
            self._collect_inputs()
            self.btn_start.setEnabled(True)
            self.btn_start.setToolTip("")
            self._append(_("[ok] training inputs look valid"))
        except ValueError as exc:
            self.btn_start.setEnabled(False)
            self.btn_start.setToolTip(str(exc))
            self._append(_("[invalid] {err}").format(err=exc))

    def _start_training(self) -> None:
        # Ensure unsaved pipeline hyperparameters are persisted before launching training.
        mw = self.window()
        page_widgets = getattr(mw, "page_widgets", None)
        if isinstance(page_widgets, list):
            for page in page_widgets:
                save_if_dirty = getattr(page, "save_if_dirty", None)
                if callable(save_if_dirty):
                    if not save_if_dirty(show_success=False):
                        self._append(_("[error] Could not auto-save pipeline config; training cancelled."))
                        return
        request = self._collect_inputs()
        is_lora = isinstance(request, LoraTrainingConfig)
        args = request  # kept as `args` below for the (classification-only) detached-subprocess path
        summary_base = (
            f"image_generation_lora/{request.base_architecture.value}"
            if is_lora
            else f"{request.framework.value}/{request.architecture.value}"
        )
        if not is_lora and self.train_subprocess.isChecked():
            self._append(f"[run] mb train — detached subprocess ({summary_base})")
            self._set_busy(True)
            try:
                ws = Workspace.load(default_settings())
                pipe = effective_pipeline_config_path(ws)
                log_path = Path(args.output_dir) / "mb_train_subprocess.log"
                proc, json_path = spawn_mb_train_subprocess(
                    args,
                    pipeline_config=pipe,
                    log_file=log_path,
                )
                self._append(
                    _("[detached] PID {pid} — log: {log} — args JSON: {args}").format(
                        pid=proc.pid, log=log_path, args=json_path
                    )
                )
                append_recent_run(
                    ModelBuilderTaskType.TRAIN,
                    f"Train (detached) {summary_base}",
                    True,
                    f"PID {proc.pid} — {log_path}",
                )
                QMessageBox.information(
                    self,
                    _("Training started (separate process)"),
                    _(
                        "Training is running outside this window. Closing the app does not stop it.\n\n"
                        "Process ID: {pid}\n"
                        "Log file:\n{log}\n\n"
                        "Training arguments were written to:\n{args}\n\n"
                        "Stop the process from Task Manager or your OS if you need to abort."
                    ).format(pid=proc.pid, log=log_path, args=json_path),
                )
            except Exception as exc:
                self._append(_("[error] {err}").format(err=exc))
                append_recent_run(
                    ModelBuilderTaskType.TRAIN,
                    f"Train (detached) {summary_base}",
                    False,
                    str(exc),
                )
                qt_operation_error(
                    self,
                    _("Could not start detached training"),
                    _("Failed to launch the training subprocess. See Details for the error."),
                    detail=str(exc),
                )
            finally:
                self._set_busy(False)
            return

        self._append(f"[run] mb train ({summary_base})")
        self._pending_train_summary = f"mb train ({summary_base})"
        self._pending_training_args = request
        self._set_busy(True)
        worker = self._execute_training_lora if is_lora else self._execute_training_classification
        handle = start_task(
            worker,
            self._on_training_success,
            self._on_training_error,
            lambda: self._set_busy(False),
            request,
            pass_context=True,
            on_cancelled=self._on_training_cancelled,
        )
        attach_training_progress_dialog(self, _("Training"), handle, cancellable=True)

    def _execute_training_lora(self, ctx: LongTaskContext, config: LoraTrainingConfig) -> str:
        from mb.training.lora_diffusion_trainer import train_image_generation_lora

        # Ensure training modules using stdlib loggers also write to shared log file.
        setup_logging(script_name="train_gui")
        ctx.progress(_("Training LoRA adapter…"), None, True)
        output_dir = train_image_generation_lora(config, cancel_event=ctx.cancel_event)
        return str(output_dir)

    def _execute_training_classification(self, ctx: LongTaskContext, args: TrainingRunArgs) -> str:
        from mb.training.trainer import ModelTrainer

        # Ensure training modules using stdlib loggers also write to shared log file.
        setup_logging(script_name="train_gui")

        trainer = ModelTrainer(
            framework=args.framework,
            model_type=ModelType.IMAGE_CLASSIFICATION,
            pipeline_config=get_pipeline_config(),
        )
        supported = trainer.get_supported_architectures()
        if args.architecture.value not in supported:
            raise ValueError(
                _(
                    "Architecture '{arch}' not supported for {fw}. Supported: {sup}"
                ).format(
                    arch=args.architecture.value,
                    fw=args.framework.value,
                    sup=supported,
                )
            )
        model_path = trainer.train(
            args,
            cancel_event=ctx.cancel_event,
            progress_cb=lambda m, p: ctx.progress(m, p),
        )
        return str(model_path)

    def _on_training_success(self, model_path: str) -> None:
        self._append(_("[done] Training complete. Model saved: {path}").format(path=model_path))
        snap: str | None = None
        args = getattr(self, "_pending_training_args", None)
        if args is not None and getattr(args, "update_snapshot", False):
            p = find_latest_unified_snapshot_path([args.data_dir])
            if p is not None:
                snap = str(p.resolve())
                self._append(f"[snapshot] {snap}")
        append_recent_run(
            ModelBuilderTaskType.TRAIN,
            getattr(self, "_pending_train_summary", "mb train"),
            True,
            model_path or "",
            snapshot_path=snap,
        )

    def _on_training_cancelled(self) -> None:
        self._append(
            _(
                "[stopped] Training cancelled — partial checkpoints may exist; check the output folder before re-running."
            )
        )
        append_recent_run(
            ModelBuilderTaskType.TRAIN,
            getattr(self, "_pending_train_summary", "mb train"),
            False,
            "cancelled",
        )

    def _on_training_error(self, message: str) -> None:
        self._append(_("[error] {err}").format(err=message))
        append_recent_run(
            ModelBuilderTaskType.TRAIN,
            getattr(self, "_pending_train_summary", "mb train"),
            False,
            message,
        )
        qt_operation_error(
            self,
            _("Training failed"),
            _("Training stopped with an error. See Details for the message from the trainer."),
            detail=message,
        )
