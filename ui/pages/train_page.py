"""Training page scaffold aligned with `mb train` arguments."""

from __future__ import annotations

from pathlib import Path

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

from mb.config import get_config
from mb.models.types import ModelType
from mb.training.trainer import ModelTrainer
from ui.task_runner import start_task


class TrainPage(QWidget):
    """UI scaffold for training configuration."""

    def __init__(self) -> None:
        super().__init__()
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
        self.data_dir = QLineEdit("data")
        self.output_dir = QLineEdit("data/models")
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
        self.btn_start = QPushButton("Start Training")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_start)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Training validation and execution messages will appear here.")
        root.addWidget(self.output, 1)

        self.framework.currentTextChanged.connect(self._refresh_architecture_hint)
        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_start.clicked.connect(self._start_training)
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
            trainer = ModelTrainer(framework=framework, model_type=ModelType.IMAGE_CLASSIFICATION, config=get_config(None))
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

    def _collect_inputs(self) -> dict:
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

        cli_hyperparams = {
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

        return {
            "framework": self.framework.currentText(),
            "architecture": architecture,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "resume_from": resume_path,
            "run_id": self.run_id.text().strip() or None,
            "update_snapshot": not self.skip_snapshot.isChecked(),
            "cli_hyperparams": cli_hyperparams,
        }

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
        payload = self._collect_inputs()
        self._append(f"[run] mb train ({payload['framework']}/{payload['architecture']})")
        self._set_busy(True)
        start_task(
            self._execute_training,
            self._on_training_success,
            self._on_training_error,
            lambda: self._set_busy(False),
            payload,
        )

    def _execute_training(self, payload: dict) -> str:
        trainer = ModelTrainer(
            framework=payload["framework"],
            model_type=ModelType.IMAGE_CLASSIFICATION,
            config=get_config(None),
        )
        supported = trainer.get_supported_architectures()
        if payload["architecture"] not in supported:
            raise ValueError(f"Architecture '{payload['architecture']}' not supported for {payload['framework']}. Supported: {supported}")
        model_path = trainer.train(
            data_dir=payload["data_dir"],
            architecture=payload["architecture"],
            output_dir=payload["output_dir"],
            cli_hyperparams=payload["cli_hyperparams"],
            resume_from_checkpoint=payload["resume_from"],
            run_id=payload["run_id"],
            update_snapshot=payload["update_snapshot"],
        )
        return str(model_path)

    def _on_training_success(self, model_path: str) -> None:
        self._append(f"[done] Training complete. Model saved: {model_path}")

    def _on_training_error(self, message: str) -> None:
        self._append(f"[error] {message}")
        QMessageBox.critical(self, "Training failed", message)
