"""Training page scaffold aligned with `mb train` arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
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

from mb.models.types import ModelType
from mb.pipeline_config import get_pipeline_config
from mb.training.run_args import TrainingRunArgs
from ui.lib.qt_alert import qt_operation_error
from ui.lib.task_progress import attach_progress_dialog
from ui.spawn_mb_train import spawn_mb_train_subprocess
from ui.task_context import LongTaskContext
from ui.task_runner import start_task
from ui.workspace import Workspace, default_settings, effective_pipeline_config_path


class TrainPage(QWidget):
    """UI scaffold for training configuration."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("train_page")
        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(QLabel("<h2>Train</h2>"))
        root.addWidget(QLabel("Configure framework, architecture, paths, and hyperparameters for `mb train`."))

        core_group = QGroupBox("Core configuration")
        core_form = QFormLayout(core_group)
        self.model_type = QComboBox()
        self.model_type.addItems(["image_classification"])
        self.framework = QComboBox()
        self.framework.addItems(["pytorch", "keras"])
        self.architecture = QLineEdit("resnet34")
        self.architecture.setObjectName("train_architecture_edit")
        self.data_dir = QLineEdit("data")
        self.data_dir.setObjectName("train_data_dir_edit")
        self.output_dir = QLineEdit("data/models")
        self.output_dir.setObjectName("train_output_dir_edit")
        self.resume_from = QLineEdit()
        self.run_id = QLineEdit()
        self.skip_snapshot = QCheckBox("Skip unified snapshot update")

        core_form.addRow("Model type", self.model_type)
        core_form.addRow("Framework", self.framework)
        core_form.addRow("Architecture", self.architecture)
        core_form.addRow("Data dir", self._path_row(self.data_dir, select_file=False))
        core_form.addRow("Output dir", self._path_row(self.output_dir, select_file=False))
        core_form.addRow("Resume checkpoint", self._path_row(self.resume_from, select_file=True))
        core_form.addRow("Run ID (optional)", self.run_id)
        core_form.addRow("", self.skip_snapshot)
        self.train_subprocess = QCheckBox(
            "Run training in a separate process (survives closing this app; see log file)"
        )
        core_form.addRow("", self.train_subprocess)
        root.addWidget(core_group)

        hp_group = QGroupBox("Hyperparameters")
        hp_form = QFormLayout(hp_group)
        self.frozen_epochs = QSpinBox()
        self.frozen_epochs.setRange(0, 10000)
        self.frozen_epochs.setValue(5)
        self.unfrozen_epochs = QSpinBox()
        self.unfrozen_epochs.setRange(0, 10000)
        self.unfrozen_epochs.setValue(20)
        self.frozen_lr = self._lr_spin(0.001)
        self.unfrozen_lr_max = self._lr_spin(0.0003)
        self.unfrozen_lr_min = self._lr_spin(0.00001)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 8192)
        self.batch_size.setSpecialValueText("Auto")
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setValue(224)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 128)
        self.num_workers.setSpecialValueText("Config default")

        hp_form.addRow("Frozen epochs", self.frozen_epochs)
        hp_form.addRow("Unfrozen epochs", self.unfrozen_epochs)
        hp_form.addRow("Frozen LR", self.frozen_lr)
        hp_form.addRow("Unfrozen LR max", self.unfrozen_lr_max)
        hp_form.addRow("Unfrozen LR min", self.unfrozen_lr_min)
        hp_form.addRow("Batch size", self.batch_size)
        hp_form.addRow("Image size", self.image_size)
        hp_form.addRow("Num workers", self.num_workers)
        root.addWidget(hp_group)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton("Validate Training Config")
        self.btn_validate.setObjectName("train_validate_btn")
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setObjectName("train_start_btn")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_start)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setObjectName("train_output_log")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Training validation and execution messages will appear here.")
        root.addWidget(self.output, 1)

        self.framework.currentTextChanged.connect(self._refresh_architecture_hint)
        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_start.clicked.connect(self._start_training)
        self._refresh_architecture_hint()
        self._validate_inputs()

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        return {
            "model_type_idx": int(self.model_type.currentIndex()),
            "framework_idx": int(self.framework.currentIndex()),
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
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        try:
            mti = state.get("model_type_idx")
            if isinstance(mti, int) and 0 <= mti < self.model_type.count():
                self.model_type.setCurrentIndex(mti)
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
        except Exception:
            pass
        self._refresh_architecture_hint()
        self._validate_inputs()

    def _path_row(self, edit: QLineEdit, select_file: bool = False) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(edit, select_file=select_file))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _lr_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(8)
        spin.setRange(0.0, 10.0)
        spin.setSingleStep(0.0001)
        spin.setValue(value)
        return spin

    def _browse(self, edit: QLineEdit, select_file: bool = False) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if select_file:
            value, _ = QFileDialog.getOpenFileName(
                self,
                "Select checkpoint file",
                start,
                "Model/checkpoint files (*.pth *.pt *.h5 *.keras *.ckpt);;All files (*.*)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        else:
            value = QFileDialog.getExistingDirectory(
                self,
                "Select directory",
                start,
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        self._validate_inputs()

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_validate.setEnabled(not busy)
        self.btn_start.setEnabled(not busy and self._can_run())

    def _refresh_architecture_hint(self) -> None:
        framework = self.framework.currentText()
        try:
            from mb.training.trainer import ModelTrainer

            trainer = ModelTrainer(
                framework=framework,
                model_type=ModelType.IMAGE_CLASSIFICATION,
                pipeline_config=get_pipeline_config(),
            )
            architectures = trainer.get_supported_architectures()
            if architectures:
                self.architecture.setPlaceholderText(", ".join(architectures[:8]))
                self._append(f"[info] {framework} architectures: {', '.join(architectures)}")
        except Exception as exc:
            self.architecture.setPlaceholderText("Enter architecture manually")
            self._append(f"[warn] Could not load architecture list for {framework}: {exc}")

    def _can_run(self) -> bool:
        try:
            self._collect_inputs()
            return True
        except ValueError:
            return False

    def _collect_inputs(self) -> TrainingRunArgs:
        data_dir = Path(self.data_dir.text().strip() or "data")
        output_dir = Path(self.output_dir.text().strip() or "data/models")
        architecture = self.architecture.text().strip()
        if not architecture:
            raise ValueError("Architecture is required.")
        if not data_dir.exists():
            raise ValueError("Data directory does not exist.")
        resume_raw = self.resume_from.text().strip()
        resume_path = Path(resume_raw) if resume_raw else None
        if resume_path and not resume_path.exists():
            raise ValueError("Resume checkpoint path does not exist.")

        cli_hyperparams: dict[str, Any] = {
            "frozen_epochs": int(self.frozen_epochs.value()),
            "unfrozen_epochs": int(self.unfrozen_epochs.value()),
            "frozen_lr": float(self.frozen_lr.value()),
            "unfrozen_lr_max": float(self.unfrozen_lr_max.value()),
            "unfrozen_lr_min": float(self.unfrozen_lr_min.value()),
            "image_size": int(self.image_size.value()),
        }
        if self.batch_size.value() > 0:
            cli_hyperparams["batch_size"] = int(self.batch_size.value())
        if self.num_workers.value() > 0:
            cli_hyperparams["num_workers"] = int(self.num_workers.value())

        return TrainingRunArgs(
            framework=self.framework.currentText(),
            architecture=architecture,
            data_dir=data_dir,
            output_dir=output_dir,
            resume_from=resume_path,
            run_id=self.run_id.text().strip() or None,
            update_snapshot=not self.skip_snapshot.isChecked(),
            cli_hyperparams=cli_hyperparams,
        )

    def _validate_inputs(self) -> None:
        try:
            self._collect_inputs()
            self.btn_start.setEnabled(True)
            self.btn_start.setToolTip("")
            self._append("[ok] training inputs look valid")
        except ValueError as exc:
            self.btn_start.setEnabled(False)
            self.btn_start.setToolTip(str(exc))
            self._append(f"[invalid] {exc}")

    def _start_training(self) -> None:
        args = self._collect_inputs()
        if self.train_subprocess.isChecked():
            self._append(f"[run] mb train — detached subprocess ({args.framework}/{args.architecture})")
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
                    f"[detached] PID {proc.pid} — log: {log_path} — args JSON: {json_path}"
                )
                QMessageBox.information(
                    self,
                    "Training started (separate process)",
                    "Training is running outside this window. Closing the app does not stop it.\n\n"
                    f"Process ID: {proc.pid}\n"
                    f"Log file:\n{log_path}\n\n"
                    f"Training arguments were written to:\n{json_path}\n\n"
                    "Stop the process from Task Manager or your OS if you need to abort.",
                )
            except Exception as exc:
                self._append(f"[error] {exc}")
                qt_operation_error(
                    self,
                    "Could not start detached training",
                    "Failed to launch the training subprocess. See Details for the error.",
                    detail=str(exc),
                )
            finally:
                self._set_busy(False)
            return

        self._append(f"[run] mb train ({args.framework}/{args.architecture})")
        self._set_busy(True)
        handle = start_task(
            self._execute_training,
            self._on_training_success,
            self._on_training_error,
            lambda: self._set_busy(False),
            args,
            pass_context=True,
            on_cancelled=self._on_training_cancelled,
        )
        attach_progress_dialog(self, "Training", handle, cancellable=True)

    def _execute_training(self, ctx: LongTaskContext, args: TrainingRunArgs) -> str:
        from mb.training.trainer import ModelTrainer

        trainer = ModelTrainer(
            framework=args.framework,
            model_type=ModelType.IMAGE_CLASSIFICATION,
            pipeline_config=get_pipeline_config(),
        )
        supported = trainer.get_supported_architectures()
        if args.architecture not in supported:
            raise ValueError(
                f"Architecture '{args.architecture}' not supported for {args.framework}. Supported: {supported}"
            )
        model_path = trainer.train(
            args,
            cancel_event=ctx.cancel_event,
            progress_cb=lambda m, p: ctx.progress(m, p),
        )
        return str(model_path)

    def _on_training_success(self, model_path: str) -> None:
        self._append(f"[done] Training complete. Model saved: {model_path}")

    def _on_training_cancelled(self) -> None:
        self._append("[stopped] Training cancelled — partial checkpoints may exist; check the output folder before re-running.")

    def _on_training_error(self, message: str) -> None:
        self._append(f"[error] {message}")
        qt_operation_error(
            self,
            "Training failed",
            "Training stopped with an error. See Details for the message from the trainer.",
            detail=message,
        )
