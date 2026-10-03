"""Hybrid editor for :mod:`mb.pipeline_config` (structured form + raw YAML)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mb.pipeline_config import (
    PIPELINE_ROOT_KEYS,
    default_pipeline_yaml_dict,
    gather_subpath_for_display,
    gather_subpath_for_storage,
    get_pipeline_config,
    reload_pipeline_config,
    save_pipeline_yaml,
)
from mb.models.types import (
    ArchitectureType,
    ClassWeightingMode,
    FrameworkType,
    LabelMode,
    ModelType,
    TextBackendType,
    TextCalibrationMethod,
    TextThresholdPolicy,
)
from mb.training.text_config import (
    CLASS_WEIGHT_BALANCED,
    CLASS_WEIGHT_NONE,
    PIPELINE_SECTION as TEXT_SECTION,
    TEXT_CLASSIFICATION_DEFAULTS,
    TextConfigError,
    parse_text_run_config,
)
from mb.utils.logging_setup import get_logger
from ui.lib.directory_line_edit_row import make_directory_line_edit_row
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.qt_alert import qt_alert
from ui.lib.tooltip_qt import create_tooltip
from mb.utils.translations import _
from ui.workspace import resolve_pipeline_save_path

if TYPE_CHECKING:
    from ui.main_window import MainWindow

logger = get_logger(__name__)


# Combo data for a loaded {0: w0, 1: w1} class_weight, which the form keeps as-is.
_CUSTOM_CLASS_WEIGHT = "__custom__"


def _ext_lines_to_list(text: str) -> list[str]:
    out: list[str] = []
    for part in text.replace(",", "\n").splitlines():
        s = part.strip()
        if s:
            if not s.startswith("."):
                s = "." + s.lstrip(".")
            out.append(s)
    return out


def _list_to_ext_lines(items: object) -> str:
    if not isinstance(items, list):
        return ""
    return "\n".join(str(x) for x in items)


def _parse_class_names(text: str) -> Any:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines


def _class_names_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)


class PipelineConfigPage(QWidget):
    """Structured pipeline fields plus an Advanced YAML tab."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pipeline_config_page")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._page_title = QLabel()
        tf = QFont(self._page_title.font())
        tf.setPointSizeF(tf.pointSizeF() + 3)
        tf.setBold(True)
        self._page_title.setFont(tf)
        root.addWidget(self._page_title)

        self._page_desc = QLabel()
        self._page_desc.setWordWrap(True)
        root.addWidget(self._page_desc)

        self._path_label = QLabel()
        self._path_label.setWordWrap(True)
        root.addWidget(self._path_label)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_model_tab(), "")
        self._tabs.addTab(self._build_data_tab(), "")
        self._tabs.addTab(self._build_training_tab(), "")
        self._tabs.addTab(self._build_paths_tab(), "")
        self._tabs.addTab(self._build_text_tab(), "")
        self._tabs.addTab(self._build_advanced_tab(), "")
        root.addWidget(self._tabs, 1)

        row = QHBoxLayout()
        self._btn_reload = QPushButton()
        self._btn_reload.clicked.connect(self._refresh_from_disk)
        self._btn_apply_yaml = QPushButton()
        self._btn_apply_yaml.clicked.connect(self._apply_advanced_yaml)
        self._btn_save = QPushButton()
        self._btn_save.clicked.connect(self._on_save)
        self._btn_default = QPushButton()
        self._btn_default.clicked.connect(self._on_set_default)
        row.addWidget(self._btn_reload)
        row.addWidget(self._btn_apply_yaml)
        row.addWidget(self._btn_save)
        row.addWidget(self._btn_default)
        row.addStretch(1)
        root.addLayout(row)

        self.retranslate_ui()
        # MainWindow already called reload_pipeline_config with the same path; avoid a second load.
        # Defer until after this page is parented under MainWindow: during __init__ self.window()
        # is still None, so _refresh_from_disk would call reload_pipeline_config(None) and replace
        # the user pipeline with packaged defaults.
        QTimer.singleShot(0, self._startup_pipeline_refresh)
        self._last_saved_cfg: dict[str, Any] = {}
        # Last loaded text_classification section; keys without a form field pass through.
        self._loaded_text_cfg: dict[str, Any] = copy.deepcopy(TEXT_CLASSIFICATION_DEFAULTS)

    def collect_gui_state(self) -> dict:
        return {}

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return

    def _main_window(self) -> MainWindow | None:
        from ui.main_window import MainWindow

        w = self.window()
        return w if isinstance(w, MainWindow) else None

    def retranslate_ui(self) -> None:
        self._page_title.setText(_("Pipeline"))
        self._page_desc.setText(
            _(
                "Machine-learning defaults (<code>model</code>, <code>data</code>, "
                "<code>training</code>, <code>paths</code>, <code>text_classification</code>). "
                "Save writes to your active pipeline file, "
                "or to <code>pipeline.yaml</code> beside <code>application.yaml</code> in app data when "
                "only packaged defaults are loaded."
            )
        )
        self._tabs.setTabText(0, _("Model"))
        self._tabs.setTabText(1, _("Data"))
        self._tabs.setTabText(2, _("Training"))
        self._tabs.setTabText(3, _("Paths"))
        self._tabs.setTabText(4, _("Text classification"))
        self._tabs.setTabText(5, _("Advanced (YAML)"))
        self._model_group.setTitle(_("Model"))
        self._data_group.setTitle(_("Data"))
        self._gather_group.setTitle(_("Gather defaults"))
        self._training_group.setTitle(_("Training"))
        self._paths_group.setTitle(_("Paths"))
        apply_qform_label_column(
            self._model_form,
            [_("Default model type"), _("Default framework"), _("Default architecture")],
        )
        apply_qform_label_column(self._data_form, self._data_form_labels())
        apply_qform_label_column(
            self._gather_form,
            [
                _("Default gather target count"),
                _("Gather target subfolder"),
                _("Rejected subfolder"),
            ],
        )
        apply_qform_label_column(
            self._training_form,
            [
                _("Frozen epochs"),
                _("Unfrozen epochs"),
                _("Frozen learning rate"),
                _("Unfrozen LR (max)"),
                _("Unfrozen LR (min)"),
                _("DataLoader workers"),
                _("Class weighting"),
                _("Max class weight"),
                "",
            ],
        )
        apply_qform_label_column(
            self._paths_form,
            [_("Models directory"), _("Logs directory"), _("Timing data directory")],
        )
        self._text_group.setTitle(_("Text classification"))
        apply_qform_label_column(self._text_form, self._text_form_labels())
        self._x_pass_hint.setText(
            _(
                "tier_weight, gold_tiers, backend_options and an explicit class_weight mapping "
                "are kept as loaded; edit them in the Advanced (YAML) tab. Unknown keys and "
                "invalid values in this section are errors when training."
            )
        )
        self._x_model.setPlaceholderText(_("empty = backend default"))
        self._x_device.setPlaceholderText(_("empty = cuda when available, else cpu"))
        for spin in (self._x_lr, self._x_epochs, self._x_batch):
            spin.setSpecialValueText(_("Backend default"))
        self._x_subsample.setSpecialValueText(_("Off"))
        self._x_exclude.setText(_("Exclude label-conflict groups from training and metrics"))
        self._x_rescore.setText(_("Score unlabeled_cut.tsv when present"))
        self._yaml_hint.setText(
            _(
                "Full pipeline document. Click “Apply YAML” to parse and refresh the tabs, "
                "or edit the tabs and Save — the Advanced view updates on reload."
            )
        )
        self._btn_reload.setText(_("Reload from disk"))
        self._btn_apply_yaml.setText(_("Apply YAML"))
        self._btn_save.setText(_("Save"))
        self._btn_default.setText(_("Reset to shipped defaults…"))
        self._btn_default.setToolTip(
            _("Replace your user pipeline.yaml (app data) with the package default_pipeline.yaml.")
        )
        self._store_ck.setText(_("Save training checkpoints"))
        self._apply_pipeline_field_tooltips()

    def _apply_pipeline_field_tooltips(self) -> None:
        """Hover text for pipeline form fields (see :mod:`ui.lib.tooltip_qt`)."""

        def ensure(widget: QWidget, attr: str, text: str) -> None:
            tip = getattr(self, attr, None)
            if tip is None:
                setattr(self, attr, create_tooltip(widget, text))
            else:
                tip.set_text(text)

        ensure(
            self._m_type,
            "_tip_m_type",
            _("Task type selected when you start a new model project."),
        )
        ensure(
            self._m_framework,
            "_tip_m_framework",
            _("Deep learning framework for new runs (e.g. pytorch)."),
        )
        ensure(
            self._m_arch,
            "_tip_m_arch",
            _("Default backbone architecture name for new training runs."),
        )
        ensure(
            self._d_raw,
            "_tip_d_raw",
            _(
                "Root folder for raw images: one subfolder per class after gather/convert. "
                "Relative paths are resolved from the app working directory or project root."
            ),
        )
        ensure(
            self._d_out,
            "_tip_d_out",
            _(
                "Output of create-dataset: this folder holds train/ and test/ used for training and snapshots. "
                "Keep it on a fast internal disk when raw data lives on external storage."
            ),
        )
        ensure(
            self._g_target_dir,
            "_tip_g_target",
            _(
                "Subfolder under raw data for accepted gather output (e.g. coherent). "
                "Relative to Raw data directory unless you enter an absolute path."
            ),
        )
        ensure(
            self._g_rej,
            "_tip_g_rej",
            _(
                "Subfolder under raw data for files rejected during gather (e.g. rejected). "
                "Relative to Raw data directory unless you enter an absolute path."
            ),
        )

    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox()
        self._model_group = box
        form = QFormLayout(box)
        self._model_form = form
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._m_type = QComboBox()
        for mt in ModelType:
            self._m_type.addItem(mt.value.replace("_", " "), mt.value)
        form.addRow(_("Default model type"), self._m_type)

        self._m_framework = QLineEdit()
        form.addRow(_("Default framework"), self._m_framework)

        self._m_arch = QLineEdit()
        form.addRow(_("Default architecture"), self._m_arch)

        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox()
        self._data_group = box
        form = QFormLayout(box)
        self._data_form = form
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._d_raw = QLineEdit()
        self._d_out = QLineEdit()

        self._d_test_pc = QSpinBox()
        self._d_test_pc.setRange(0, 10_000_000)

        self._d_test_split_mode = QComboBox()
        self._d_test_split_mode.addItem(_("Fixed count per class"), "fixed")
        self._d_test_split_mode.addItem(_("Dataset-weighted (modulated)"), "dataset_weighted")

        self._d_test_small_thr = QSpinBox()
        self._d_test_small_thr.setRange(0, 10_000_000)
        self._d_test_small_thr.setSpecialValueText(_("Same as test anchor"))
        self._d_test_small_thr.setValue(0)

        self._d_seed = QSpinBox()
        self._d_seed.setRange(0, 2_147_483_647)
        self._d_seed.setSpecialValueText(_("None"))
        self._d_seed.setValue(0)

        self._d_im_size = QSpinBox()
        self._d_im_size.setRange(1, 4096)

        self._d_batch = QLineEdit()
        self._d_batch.setPlaceholderText(
            _("Leave empty for the model-type default batch size (omitted from saved YAML).")
        )

        self._d_img_types = QPlainTextEdit()
        self._d_img_types.setFixedHeight(100)

        self._d_vid_types = QPlainTextEdit()
        self._d_vid_types.setFixedHeight(80)

        self._d_class_names = QPlainTextEdit()
        self._d_class_names.setFixedHeight(72)
        self._d_class_names.setPlaceholderText(_("One class folder name per line; empty = discover"))

        self._d_qual = QLineEdit()
        self._d_qual.setPlaceholderText(_("empty = any"))

        self._d_label_mode = QComboBox()
        for mode in LabelMode:
            self._d_label_mode.addItem(mode.value, mode.value)

        widgets = [
            make_directory_line_edit_row(w, self._d_raw, dialog_title=_("Select raw data directory")),
            make_directory_line_edit_row(w, self._d_out, dialog_title=_("Select dataset directory")),
            self._d_test_pc,
            self._d_test_split_mode,
            self._d_test_small_thr,
            self._d_seed,
            self._d_im_size,
            self._d_batch,
            self._d_img_types,
            self._d_vid_types,
            self._d_class_names,
            self._d_qual,
            self._d_label_mode,
        ]
        labels = self._data_form_labels()
        assert len(labels) == len(widgets)
        for label, widget in zip(labels, widgets):
            form.addRow(label, widget)

        lay.addWidget(box)

        gbox = QGroupBox()
        self._gather_group = gbox
        gform = QFormLayout(gbox)
        self._gather_form = gform
        gform.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._g_target_n = QSpinBox()
        self._g_target_n.setRange(1, 100_000_000)
        gform.addRow(_("Default gather target count"), self._g_target_n)

        self._g_target_dir = QLineEdit()
        self._g_target_dir.setPlaceholderText(
            _("Relative to raw data directory (e.g. coherent); absolute path optional")
        )
        gform.addRow(
            _("Gather target subfolder"),
            make_directory_line_edit_row(
                w, self._g_target_dir, dialog_title=_("Select default gather target directory")
            ),
        )

        self._g_rej = QLineEdit()
        self._g_rej.setPlaceholderText(
            _("Relative to raw data directory (e.g. rejected); absolute path optional")
        )
        gform.addRow(
            _("Rejected subfolder"),
            make_directory_line_edit_row(
                w, self._g_rej, dialog_title=_("Select default rejected directory")
            ),
        )

        lay.addWidget(gbox)
        lay.addStretch(1)
        return w

    @staticmethod
    def _data_form_labels() -> list[str]:
        """Data-tab row labels, in row order; used at construction and on retranslate."""
        return [
            _("Raw data directory"),
            _("Dataset output directory"),
            _("Test images per class"),
            _("Test split mode"),
            _("Small-class threshold (weighted)"),
            _("Default create-dataset seed"),
            _("Image size (pixels)"),
            _("Training batch size"),
            _("Image file extensions (one per line)"),
            _("Video file extensions (one per line)"),
            _("Class folder names (empty = auto-discover)"),
            _("Class qualifying subfolder"),
            _("Label mode"),
        ]

    def _build_training_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox()
        self._training_group = box
        form = QFormLayout(box)
        self._training_form = form
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._t_frozen_e = QSpinBox()
        self._t_frozen_e.setRange(0, 1_000_000)
        form.addRow(_("Frozen epochs"), self._t_frozen_e)

        self._t_unfrozen_e = QSpinBox()
        self._t_unfrozen_e.setRange(0, 1_000_000)
        form.addRow(_("Unfrozen epochs"), self._t_unfrozen_e)

        self._t_frozen_lr = QDoubleSpinBox()
        self._t_frozen_lr.setRange(0.0, 1.0)
        self._t_frozen_lr.setDecimals(8)
        self._t_frozen_lr.setSingleStep(0.0001)
        form.addRow(_("Frozen learning rate"), self._t_frozen_lr)

        self._t_umax = QDoubleSpinBox()
        self._t_umax.setRange(0.0, 1.0)
        self._t_umax.setDecimals(8)
        form.addRow(_("Unfrozen LR (max)"), self._t_umax)

        self._t_umin = QDoubleSpinBox()
        self._t_umin.setRange(0.0, 1.0)
        self._t_umin.setDecimals(8)
        form.addRow(_("Unfrozen LR (min)"), self._t_umin)

        self._t_workers = QSpinBox()
        self._t_workers.setRange(0, 256)
        form.addRow(_("DataLoader workers"), self._t_workers)

        self._t_class_weighting = QComboBox()
        for mode in ClassWeightingMode:
            self._t_class_weighting.addItem(mode.value, mode.value)
        form.addRow(_("Class weighting"), self._t_class_weighting)

        self._t_class_weight_max = QDoubleSpinBox()
        self._t_class_weight_max.setRange(1.0, 10_000.0)
        self._t_class_weight_max.setDecimals(2)
        form.addRow(_("Max class weight"), self._t_class_weight_max)

        self._store_ck = QCheckBox()
        form.addRow("", self._store_ck)

        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _build_paths_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox()
        self._paths_group = box
        form = QFormLayout(box)
        self._paths_form = form
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._p_models = QLineEdit()
        form.addRow(
            _("Models directory"),
            make_directory_line_edit_row(w, self._p_models, dialog_title=_("Select models directory")),
        )

        self._p_logs = QLineEdit()
        form.addRow(
            _("Logs directory"),
            make_directory_line_edit_row(w, self._p_logs, dialog_title=_("Select logs directory")),
        )

        lay.addWidget(box)
        lay.addStretch(1)
        return w

    @staticmethod
    def _text_form_labels() -> list[str]:
        return [
            _("Dataset directory"),
            _("Runs directory"),
            _("Backend"),
            _("Model id"),
            _("Seed"),
            _("Max length (tokens)"),
            _("Class weight"),
            "",
            _("Subsample keep_unreviewed (x positives)"),
            _("Learning rate"),
            _("Epochs"),
            _("Batch size"),
            _("Warmup ratio"),
            _("Weight decay"),
            _("Early-stopping patience"),
            _("Calibration"),
            _("Threshold policy"),
            _("Threshold target"),
            _("Review queue size"),
            "",
            _("Device"),
            _("Model card notes (Markdown)"),
        ]

    def _build_text_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox()
        self._text_group = box
        form = QFormLayout(box)
        self._text_form = form
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def opt_double(decimals: int, step: float, hi: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setRange(0.0, hi)
            return spin

        self._x_data = QLineEdit()
        self._x_runs = QLineEdit()
        self._x_backend = QComboBox()
        for b in TextBackendType:
            self._x_backend.addItem(b.value, b.value)
        self._x_model = QLineEdit()
        self._x_seed = QSpinBox()
        self._x_seed.setRange(0, 2_147_483_647)
        self._x_max_len = QSpinBox()
        self._x_max_len.setRange(1, 100_000)
        self._x_class_weight = QComboBox()
        self._x_class_weight.addItem(CLASS_WEIGHT_BALANCED, CLASS_WEIGHT_BALANCED)
        self._x_class_weight.addItem(CLASS_WEIGHT_NONE, CLASS_WEIGHT_NONE)
        self._x_exclude = QCheckBox()
        self._x_subsample = opt_double(2, 0.5, 1_000.0)
        self._x_lr = opt_double(8, 0.00001, 1.0)
        self._x_epochs = QSpinBox()
        self._x_epochs.setRange(0, 100_000)
        self._x_batch = QSpinBox()
        self._x_batch.setRange(0, 1_000_000)
        self._x_warmup = opt_double(3, 0.01, 1.0)
        self._x_wd = opt_double(4, 0.001, 10.0)
        self._x_patience = QSpinBox()
        self._x_patience.setRange(0, 1000)
        self._x_calibration = QComboBox()
        for c in TextCalibrationMethod:
            self._x_calibration.addItem(c.value, c.value)
        self._x_threshold_kind = QComboBox()
        for k in TextThresholdPolicy:
            self._x_threshold_kind.addItem(k.value, k.value)
        self._x_threshold_value = opt_double(3, 0.01, 1.0)
        self._x_threshold_value.setMinimum(0.001)
        self._x_queue = QSpinBox()
        self._x_queue.setRange(0, 1_000_000)
        self._x_rescore = QCheckBox()
        self._x_device = QLineEdit()
        self._x_notes = QPlainTextEdit()
        self._x_notes.setFixedHeight(120)

        labels = self._text_form_labels()
        widgets = [
            make_directory_line_edit_row(w, self._x_data, dialog_title=_("Select text dataset directory")),
            make_directory_line_edit_row(w, self._x_runs, dialog_title=_("Select runs directory")),
            self._x_backend,
            self._x_model,
            self._x_seed,
            self._x_max_len,
            self._x_class_weight,
            self._x_exclude,
            self._x_subsample,
            self._x_lr,
            self._x_epochs,
            self._x_batch,
            self._x_warmup,
            self._x_wd,
            self._x_patience,
            self._x_calibration,
            self._x_threshold_kind,
            self._x_threshold_value,
            self._x_queue,
            self._x_rescore,
            self._x_device,
            self._x_notes,
        ]
        for label, widget in zip(labels, widgets):
            form.addRow(label, widget)
        lay.addWidget(box)
        self._x_pass_hint = QLabel()
        self._x_pass_hint.setWordWrap(True)
        lay.addWidget(self._x_pass_hint)
        lay.addStretch(1)
        return w

    def _text_dict_from_form(self) -> dict[str, Any]:
        out = copy.deepcopy(self._loaded_text_cfg)

        def opt(v: float, cast):
            return None if v <= 0 else cast(v)

        cw = self._x_class_weight.currentData()
        out.update(
            {
                "data_dir": self._x_data.text().strip() or TEXT_CLASSIFICATION_DEFAULTS["data_dir"],
                "runs_dir": self._x_runs.text().strip() or TEXT_CLASSIFICATION_DEFAULTS["runs_dir"],
                "backend": str(self._x_backend.currentData()),
                "model_id": self._x_model.text().strip() or None,
                "seed": int(self._x_seed.value()),
                "max_length": int(self._x_max_len.value()),
                # The "custom" entry stands for the loaded mapping, already in *out*.
                "class_weight": out.get("class_weight") if cw == _CUSTOM_CLASS_WEIGHT else str(cw),
                "exclude_conflicts": self._x_exclude.isChecked(),
                "subsample_keep_unreviewed": opt(float(self._x_subsample.value()), float),
                "calibration": str(self._x_calibration.currentData()),
                "model_card_notes": self._x_notes.toPlainText(),
                "device": self._x_device.text().strip() or None,
            }
        )
        optim = dict(out.get("optim") or {})
        optim.update(
            {
                "lr": opt(float(self._x_lr.value()), float),
                "epochs": opt(int(self._x_epochs.value()), int),
                "batch_size": opt(int(self._x_batch.value()), int),
                "warmup_ratio": float(self._x_warmup.value()),
                "weight_decay": float(self._x_wd.value()),
            }
        )
        out["optim"] = optim
        es = dict(out.get("early_stopping") or {})
        es["patience"] = int(self._x_patience.value())
        out["early_stopping"] = es
        out["threshold_policy"] = {
            **dict(out.get("threshold_policy") or {}),
            "kind": str(self._x_threshold_kind.currentData()),
            "value": float(self._x_threshold_value.value()),
        }
        out["reports"] = {
            **dict(out.get("reports") or {}),
            "review_queue_size": int(self._x_queue.value()),
            "rescore_unlabeled_cut": self._x_rescore.isChecked(),
        }
        return out

    def _apply_text_dict_to_form(self, raw: Any) -> None:
        d = copy.deepcopy(TEXT_CLASSIFICATION_DEFAULTS)
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict) and isinstance(d.get(key), dict):
                    d[key] = {**d[key], **value}
                else:
                    d[key] = copy.deepcopy(value)
        self._loaded_text_cfg = d

        def set_combo(combo: QComboBox, value: Any) -> None:
            idx = combo.findData(str(value))
            combo.setCurrentIndex(idx if idx >= 0 else 0)

        def num(v: Any, default: float = 0.0) -> float:
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default

        self._x_data.setText(str(d.get("data_dir") or ""))
        self._x_runs.setText(str(d.get("runs_dir") or ""))
        set_combo(self._x_backend, d.get("backend"))
        self._x_model.setText(str(d.get("model_id") or ""))
        self._x_seed.setValue(int(num(d.get("seed"))))
        self._x_max_len.setValue(int(num(d.get("max_length"), 64)))
        custom_ix = self._x_class_weight.findData(_CUSTOM_CLASS_WEIGHT)
        if custom_ix >= 0:
            self._x_class_weight.removeItem(custom_ix)
        cw = d.get("class_weight")
        if isinstance(cw, dict):
            self._x_class_weight.addItem(_("custom mapping (edit in YAML)"), _CUSTOM_CLASS_WEIGHT)
            self._x_class_weight.setCurrentIndex(self._x_class_weight.count() - 1)
        else:
            set_combo(self._x_class_weight, cw)
        self._x_exclude.setChecked(bool(d.get("exclude_conflicts")))
        self._x_subsample.setValue(num(d.get("subsample_keep_unreviewed")))
        optim = d.get("optim") or {}
        self._x_lr.setValue(num(optim.get("lr")))
        self._x_epochs.setValue(int(num(optim.get("epochs"))))
        self._x_batch.setValue(int(num(optim.get("batch_size"))))
        self._x_warmup.setValue(num(optim.get("warmup_ratio")))
        self._x_wd.setValue(num(optim.get("weight_decay")))
        self._x_patience.setValue(int(num((d.get("early_stopping") or {}).get("patience"))))
        set_combo(self._x_calibration, d.get("calibration"))
        tp = d.get("threshold_policy") or {}
        set_combo(self._x_threshold_kind, tp.get("kind"))
        self._x_threshold_value.setValue(num(tp.get("value"), 0.9))
        rep_cfg = d.get("reports") or {}
        self._x_queue.setValue(int(num(rep_cfg.get("review_queue_size"))))
        self._x_rescore.setChecked(bool(rep_cfg.get("rescore_unlabeled_cut")))
        self._x_device.setText(str(d.get("device") or ""))
        self._x_notes.setPlainText(str(d.get("model_card_notes") or ""))

    def _build_advanced_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self._yaml_hint = QLabel()
        self._yaml_hint.setWordWrap(True)
        lay.addWidget(self._yaml_hint)
        self._yaml_edit = QPlainTextEdit()
        self._yaml_edit.setFont(QFont("Consolas", 10))
        lay.addWidget(self._yaml_edit, 1)
        return w

    def _full_pipeline_dict_from_form(self) -> dict[str, Any]:
        mt = self._m_type.currentData()
        batch_raw = self._d_batch.text().strip()
        batch: Any
        if batch_raw == "":
            batch = None
        else:
            try:
                batch = int(batch_raw)
            except ValueError:
                batch = None

        cq = self._d_qual.text().strip()
        class_qual = None if cq == "" else cq

        raw_for_gather = self._d_raw.text().strip() or "raw_data"
        gather = {
            "default_target_count": int(self._g_target_n.value()),
            "default_target_dir": gather_subpath_for_storage(
                raw_for_gather, self._g_target_dir.text(), "coherent"
            ),
            "default_rejected_dir": gather_subpath_for_storage(
                raw_for_gather, self._g_rej.text(), "rejected"
            ),
        }

        tsm_data = self._d_test_split_mode.currentData()
        st_raw = int(self._d_test_small_thr.value())
        seed_v = int(self._d_seed.value())
        data: dict[str, Any] = {
            "raw_data_dir": self._d_raw.text().strip() or "raw_data",
            "data_dir": self._d_out.text().strip() or "data",
            "test_per_class": int(self._d_test_pc.value()),
            "test_split_mode": str(tsm_data) if tsm_data else "fixed",
            "test_small_class_threshold": None if st_raw <= 0 else st_raw,
            "seed": None if seed_v <= 0 else seed_v,
            "image_size": int(self._d_im_size.value()),
            "batch_size": batch,
            "image_types": _ext_lines_to_list(self._d_img_types.toPlainText()),
            "video_types": _ext_lines_to_list(self._d_vid_types.toPlainText()),
            "class_names": _parse_class_names(self._d_class_names.toPlainText()),
            "class_qualifying_subdir": class_qual,
            "label_mode": str(self._d_label_mode.currentData() or LabelMode.get_default().value),
            "gather": gather,
        }

        return {
            "model": {
                "default_type": str(mt) if mt else "image_classification",
                "default_framework": self._m_framework.text().strip()
                or FrameworkType.get_default().value,
                "default_architecture": self._m_arch.text().strip()
                or ArchitectureType.get_default().value,
            },
            "data": data,
            "training": {
                "frozen_epochs": int(self._t_frozen_e.value()),
                "unfrozen_epochs": int(self._t_unfrozen_e.value()),
                "frozen_lr": float(self._t_frozen_lr.value()),
                "unfrozen_lr_max": float(self._t_umax.value()),
                "unfrozen_lr_min": float(self._t_umin.value()),
                "num_workers": int(self._t_workers.value()),
                "class_weighting": str(
                    self._t_class_weighting.currentData()
                    or ClassWeightingMode.get_default().value
                ),
                "class_weight_max": float(self._t_class_weight_max.value()),
                "store_checkpoints": self._store_ck.isChecked(),
            },
            "paths": {
                "models_dir": self._p_models.text().strip() or "data/models",
                "logs_dir": self._p_logs.text().strip() or "logs",
            },
            TEXT_SECTION: self._text_dict_from_form(),
        }

    def _apply_dict_to_form(self, cfg: dict[str, Any]) -> None:
        m = cfg.get("model") or {}
        dtype = m.get("default_type") or "image_classification"
        idx = self._m_type.findData(str(dtype))
        self._m_type.setCurrentIndex(idx if idx >= 0 else 0)
        self._m_framework.setText(
            str(m.get("default_framework") or FrameworkType.get_default().value)
        )
        self._m_arch.setText(
            str(m.get("default_architecture") or ArchitectureType.get_default().value)
        )

        d = cfg.get("data") or {}
        raw_d = str(d.get("raw_data_dir") or "")
        self._d_raw.setText(raw_d)
        self._d_out.setText(str(d.get("data_dir") or ""))
        self._d_test_pc.setValue(int(d.get("test_per_class") or 0))
        tsm = str(d.get("test_split_mode") or "fixed")
        tix = self._d_test_split_mode.findData(tsm)
        self._d_test_split_mode.setCurrentIndex(tix if tix >= 0 else 0)
        st = d.get("test_small_class_threshold")
        self._d_test_small_thr.setValue(0 if st is None else int(st))
        sd = d.get("seed")
        self._d_seed.setValue(0 if sd is None else int(sd))
        self._d_im_size.setValue(int(d.get("image_size") or 224))
        bs = d.get("batch_size")
        self._d_batch.setText("" if bs is None else str(int(bs)))
        self._d_img_types.setPlainText(_list_to_ext_lines(d.get("image_types")))
        self._d_vid_types.setPlainText(_list_to_ext_lines(d.get("video_types")))
        self._d_class_names.setPlainText(_class_names_to_text(d.get("class_names")))
        cq = d.get("class_qualifying_subdir")
        self._d_qual.setText("" if cq is None else str(cq))
        lm = LabelMode.try_from(d.get("label_mode")) or LabelMode.get_default()
        lm_idx = self._d_label_mode.findData(lm.value)
        self._d_label_mode.setCurrentIndex(lm_idx if lm_idx >= 0 else 0)

        g = d.get("gather") or {}
        self._g_target_n.setValue(int(g.get("default_target_count") or 100000))
        self._g_target_dir.setText(
            gather_subpath_for_display(raw_d, str(g.get("default_target_dir") or ""))
        )
        self._g_rej.setText(
            gather_subpath_for_display(raw_d, str(g.get("default_rejected_dir") or ""))
        )

        t = cfg.get("training") or {}
        self._t_frozen_e.setValue(int(t.get("frozen_epochs") or 0))
        self._t_unfrozen_e.setValue(int(t.get("unfrozen_epochs") or 0))
        self._t_frozen_lr.setValue(float(t.get("frozen_lr") or 0.0))
        self._t_umax.setValue(float(t.get("unfrozen_lr_max") or 0.0))
        self._t_umin.setValue(float(t.get("unfrozen_lr_min") or 0.0))
        self._t_workers.setValue(int(t.get("num_workers") or 0))
        cw = ClassWeightingMode.try_from(t.get("class_weighting")) or ClassWeightingMode.get_default()
        cw_idx = self._t_class_weighting.findData(cw.value)
        self._t_class_weighting.setCurrentIndex(cw_idx if cw_idx >= 0 else 0)
        self._t_class_weight_max.setValue(float(t.get("class_weight_max") or 50.0))
        self._store_ck.setChecked(bool(t.get("store_checkpoints")))

        p = cfg.get("paths") or {}
        self._p_models.setText(str(p.get("models_dir") or ""))
        self._p_logs.setText(str(p.get("logs_dir") or ""))

        self._apply_text_dict_to_form(cfg.get(TEXT_SECTION))

    def _sync_yaml_editor(self, cfg: dict[str, Any]) -> None:
        sub = {k: cfg[k] for k in PIPELINE_ROOT_KEYS if k in cfg}
        try:
            text = yaml.dump(sub, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception:
            text = ""
        self._yaml_edit.setPlainText(text)

    def _startup_pipeline_refresh(self) -> None:
        self._refresh_from_disk(force=False)

    def _refresh_from_disk(self, *, force: bool = True) -> None:
        """
        Reload pipeline YAML into the global config and refresh this form.

        *force* should be False on first page open when :class:`~ui.main_window.MainWindow`
        has just loaded the same path — otherwise the pipeline file is read twice on startup.
        """
        from utils.config import get_user_pipeline_config_path

        mw = self._main_window()
        if mw is not None:
            reload_pipeline_config(mw._effective_pipeline_config_path(), force=force)
        else:
            reload_pipeline_config(None, force=force)

        pc = get_pipeline_config()
        path = pc.active_path
        save_hint = (
            resolve_pipeline_save_path(mw._workspace)
            if mw is not None
            else get_user_pipeline_config_path()
        )

        if path is not None and path.is_file():
            self._path_label.setText(
                _("Loaded: {path}\nSave will write to: {save}").format(
                    path=path, save=save_hint
                )
            )
        else:
            self._path_label.setText(
                _("Loaded: packaged defaults\nSave will write to: {path}").format(path=save_hint)
            )

        loaded = pc.to_dict()
        self._apply_dict_to_form(loaded)
        self._sync_yaml_editor(loaded)
        self._last_saved_cfg = self._full_pipeline_dict_from_form()

    def _apply_advanced_yaml(self) -> None:
        raw = self._yaml_edit.toPlainText()
        try:
            parsed = yaml.safe_load(raw) or {}
        except yaml.YAMLError as e:
            qt_alert(self, _("Pipeline"), _("Invalid YAML:\n{err}").format(err=e))
            return
        if not isinstance(parsed, dict):
            qt_alert(self, _("Pipeline"), _("Root value must be a mapping."))
            return
        extra = [k for k in parsed if k not in PIPELINE_ROOT_KEYS]
        if extra:
            r = QMessageBox.question(
                self,
                _("Pipeline"),
                _(
                    "The document contains keys outside model/data/training/paths/"
                    "text_classification: {keys}\n"
                    "They will be ignored when applying to the form. Continue?"
                ).format(keys=", ".join(extra)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        merged = {k: parsed[k] for k in PIPELINE_ROOT_KEYS if k in parsed}
        if len(merged) < 1:
            qt_alert(self, _("Pipeline"), _("No model/data/training/paths/text_classification keys found."))
            return
        self._apply_dict_to_form(merged)
        self._sync_yaml_editor(merged)

    def has_unsaved_changes(self) -> bool:
        return self._full_pipeline_dict_from_form() != self._last_saved_cfg

    def save_if_dirty(self, *, show_success: bool = False) -> bool:
        from utils.config import get_user_pipeline_config_path

        if not self.has_unsaved_changes():
            return True

        mw = self._main_window()
        cfg = self._full_pipeline_dict_from_form()
        target = (
            resolve_pipeline_save_path(mw._workspace) if mw is not None else get_user_pipeline_config_path()
        )
        try:
            save_pipeline_yaml(target, cfg)
        except OSError as e:
            qt_alert(self, _("Pipeline"), _("Could not write:\n{path}\n\n{err}").format(path=target, err=e))
            return False
        logger.info(
            "Pipeline config saved to %s (%s)",
            target,
            "manual" if show_success else "autosave",
        )

        if mw is not None:
            mw.reload_mb_yaml_config()
        else:
            reload_pipeline_config(target, force=True)
        self._refresh_from_disk(force=False)
        self._last_saved_cfg = self._full_pipeline_dict_from_form()
        if show_success:
            message = _("Saved to:\n{path}").format(path=target)
            try:
                parse_text_run_config(cfg[TEXT_SECTION])
            except TextConfigError as e:
                message += "\n\n" + str(e)
            qt_alert(self, _("Pipeline"), message)
        return True

    def _on_save(self) -> None:
        self.save_if_dirty(show_success=True)

    def hideEvent(self, event) -> None:
        # Best-effort autosave when navigating away from this page.
        self.save_if_dirty(show_success=False)
        super().hideEvent(event)

    def _on_set_default(self) -> None:
        from utils.config import get_user_pipeline_config_path

        target = get_user_pipeline_config_path()
        data = default_pipeline_yaml_dict()
        try:
            save_pipeline_yaml(target, data)
        except OSError as e:
            qt_alert(
                self,
                _("Pipeline"),
                _("Could not write defaults to:\n{path}\n\n{err}").format(path=target, err=e),
            )
            return
        reload_pipeline_config(target, force=True)
        mw = self._main_window()
        if mw is not None:
            mw.reload_mb_yaml_config()
        self._refresh_from_disk()
        qt_alert(self, _("Pipeline"), _("Reset user pipeline to shipped defaults:\n{path}").format(path=target))
