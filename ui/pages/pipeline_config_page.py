"""Hybrid editor for :mod:`mb.pipeline_config` (structured form + raw YAML)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from PySide6.QtCore import Qt
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
    get_pipeline_config,
    reload_pipeline_config,
    save_pipeline_yaml,
)
from mb.models.types import ModelType
from ui.lib.directory_line_edit_row import make_directory_line_edit_row
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.qt_alert import qt_alert
from mb.utils.translations import _
from ui.workspace import resolve_pipeline_save_path

if TYPE_CHECKING:
    from ui.main_window import MainWindow


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
        self._refresh_from_disk()

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
                "<code>training</code>, <code>paths</code>). Save writes to your active pipeline file, "
                "or to <code>pipeline.yaml</code> beside <code>application.yaml</code> in app data when "
                "only packaged defaults are loaded."
            )
        )
        self._tabs.setTabText(0, _("Model"))
        self._tabs.setTabText(1, _("Data"))
        self._tabs.setTabText(2, _("Training"))
        self._tabs.setTabText(3, _("Paths"))
        self._tabs.setTabText(4, _("Advanced (YAML)"))
        self._model_group.setTitle(_("model"))
        self._data_group.setTitle(_("data"))
        self._gather_group.setTitle(_("data.gather"))
        self._training_group.setTitle(_("training"))
        self._paths_group.setTitle(_("paths"))
        apply_qform_label_column(
            self._model_form,
            [_("default_type"), _("default_framework"), _("default_architecture")],
        )
        apply_qform_label_column(
            self._data_form,
            [
                _("raw_data_dir"),
                _("data_dir"),
                _("test_per_class"),
                _("image_size"),
                _("batch_size (empty = null)"),
                _("image_types (one extension per line)"),
                _("video_types (one extension per line)"),
                _("class_names (empty = auto-discover)"),
                _("class_qualifying_subdir (empty = any)"),
            ],
        )
        apply_qform_label_column(
            self._gather_form,
            [
                _("default_target_count"),
                _("default_target_dir"),
                _("default_rejected_dir"),
                _("default_raw_data_dir"),
            ],
        )
        apply_qform_label_column(
            self._training_form,
            [
                _("frozen_epochs"),
                _("unfrozen_epochs"),
                _("frozen_lr"),
                _("unfrozen_lr_max"),
                _("unfrozen_lr_min"),
                _("num_workers"),
                "",
            ],
        )
        apply_qform_label_column(
            self._paths_form,
            [_("models_dir"), _("logs_dir"), _("timing_dir")],
        )
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
        self._store_ck.setText(_("store_checkpoints"))

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
        form.addRow(_("default_type"), self._m_type)

        self._m_framework = QLineEdit()
        form.addRow(_("default_framework"), self._m_framework)

        self._m_arch = QLineEdit()
        form.addRow(_("default_architecture"), self._m_arch)

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
        form.addRow(
            _("raw_data_dir"),
            make_directory_line_edit_row(w, self._d_raw, dialog_title=_("Select raw data directory")),
        )

        self._d_out = QLineEdit()
        form.addRow(
            _("data_dir"),
            make_directory_line_edit_row(w, self._d_out, dialog_title=_("Select dataset directory")),
        )

        self._d_test_pc = QSpinBox()
        self._d_test_pc.setRange(0, 10_000_000)
        form.addRow(_("test_per_class"), self._d_test_pc)

        self._d_im_size = QSpinBox()
        self._d_im_size.setRange(1, 4096)
        form.addRow(_("image_size"), self._d_im_size)

        self._d_batch = QLineEdit()
        self._d_batch.setPlaceholderText(_("empty → null"))
        form.addRow(_("batch_size (empty = null)"), self._d_batch)

        self._d_img_types = QPlainTextEdit()
        self._d_img_types.setFixedHeight(100)
        form.addRow(_("image_types (one extension per line)"), self._d_img_types)

        self._d_vid_types = QPlainTextEdit()
        self._d_vid_types.setFixedHeight(80)
        form.addRow(_("video_types (one extension per line)"), self._d_vid_types)

        self._d_class_names = QPlainTextEdit()
        self._d_class_names.setFixedHeight(72)
        self._d_class_names.setPlaceholderText(_("One class folder name per line; empty = discover"))
        form.addRow(_("class_names (empty = auto-discover)"), self._d_class_names)

        self._d_qual = QLineEdit()
        self._d_qual.setPlaceholderText(_("empty → null"))
        form.addRow(_("class_qualifying_subdir (empty = any)"), self._d_qual)

        lay.addWidget(box)

        gbox = QGroupBox()
        self._gather_group = gbox
        gform = QFormLayout(gbox)
        self._gather_form = gform
        gform.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._g_target_n = QSpinBox()
        self._g_target_n.setRange(1, 100_000_000)
        gform.addRow(_("default_target_count"), self._g_target_n)

        self._g_target_dir = QLineEdit()
        gform.addRow(
            _("default_target_dir"),
            make_directory_line_edit_row(
                w, self._g_target_dir, dialog_title=_("Select default gather target directory")
            ),
        )

        self._g_rej = QLineEdit()
        gform.addRow(
            _("default_rejected_dir"),
            make_directory_line_edit_row(
                w, self._g_rej, dialog_title=_("Select default rejected directory")
            ),
        )

        self._g_raw = QLineEdit()
        gform.addRow(
            _("default_raw_data_dir"),
            make_directory_line_edit_row(
                w, self._g_raw, dialog_title=_("Select default raw data directory")
            ),
        )

        lay.addWidget(gbox)
        lay.addStretch(1)
        return w

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
        form.addRow(_("frozen_epochs"), self._t_frozen_e)

        self._t_unfrozen_e = QSpinBox()
        self._t_unfrozen_e.setRange(0, 1_000_000)
        form.addRow(_("unfrozen_epochs"), self._t_unfrozen_e)

        self._t_frozen_lr = QDoubleSpinBox()
        self._t_frozen_lr.setRange(0.0, 1.0)
        self._t_frozen_lr.setDecimals(8)
        self._t_frozen_lr.setSingleStep(0.0001)
        form.addRow(_("frozen_lr"), self._t_frozen_lr)

        self._t_umax = QDoubleSpinBox()
        self._t_umax.setRange(0.0, 1.0)
        self._t_umax.setDecimals(8)
        form.addRow(_("unfrozen_lr_max"), self._t_umax)

        self._t_umin = QDoubleSpinBox()
        self._t_umin.setRange(0.0, 1.0)
        self._t_umin.setDecimals(8)
        form.addRow(_("unfrozen_lr_min"), self._t_umin)

        self._t_workers = QSpinBox()
        self._t_workers.setRange(0, 256)
        form.addRow(_("num_workers"), self._t_workers)

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
            _("models_dir"),
            make_directory_line_edit_row(w, self._p_models, dialog_title=_("Select models directory")),
        )

        self._p_logs = QLineEdit()
        form.addRow(
            _("logs_dir"),
            make_directory_line_edit_row(w, self._p_logs, dialog_title=_("Select logs directory")),
        )

        self._p_timing = QLineEdit()
        form.addRow(
            _("timing_dir"),
            make_directory_line_edit_row(w, self._p_timing, dialog_title=_("Select timing data directory")),
        )

        lay.addWidget(box)
        lay.addStretch(1)
        return w

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

        gather = {
            "default_target_count": int(self._g_target_n.value()),
            "default_target_dir": self._g_target_dir.text().strip() or "raw_data/coherent",
            "default_rejected_dir": self._g_rej.text().strip() or "raw_data/rejected",
            "default_raw_data_dir": self._g_raw.text().strip() or "raw_data",
        }

        data: dict[str, Any] = {
            "raw_data_dir": self._d_raw.text().strip() or "raw_data",
            "data_dir": self._d_out.text().strip() or "data",
            "test_per_class": int(self._d_test_pc.value()),
            "image_size": int(self._d_im_size.value()),
            "batch_size": batch,
            "image_types": _ext_lines_to_list(self._d_img_types.toPlainText()),
            "video_types": _ext_lines_to_list(self._d_vid_types.toPlainText()),
            "class_names": _parse_class_names(self._d_class_names.toPlainText()),
            "class_qualifying_subdir": class_qual,
            "gather": gather,
        }

        return {
            "model": {
                "default_type": str(mt) if mt else "image_classification",
                "default_framework": self._m_framework.text().strip() or "pytorch",
                "default_architecture": self._m_arch.text().strip() or "resnet34",
            },
            "data": data,
            "training": {
                "frozen_epochs": int(self._t_frozen_e.value()),
                "unfrozen_epochs": int(self._t_unfrozen_e.value()),
                "frozen_lr": float(self._t_frozen_lr.value()),
                "unfrozen_lr_max": float(self._t_umax.value()),
                "unfrozen_lr_min": float(self._t_umin.value()),
                "num_workers": int(self._t_workers.value()),
                "store_checkpoints": self._store_ck.isChecked(),
            },
            "paths": {
                "models_dir": self._p_models.text().strip() or "data/models",
                "logs_dir": self._p_logs.text().strip() or "logs",
                "timing_dir": self._p_timing.text().strip() or "timing_data",
            },
        }

    def _apply_dict_to_form(self, cfg: dict[str, Any]) -> None:
        m = cfg.get("model") or {}
        dtype = m.get("default_type") or "image_classification"
        idx = self._m_type.findData(str(dtype))
        self._m_type.setCurrentIndex(idx if idx >= 0 else 0)
        self._m_framework.setText(str(m.get("default_framework") or "pytorch"))
        self._m_arch.setText(str(m.get("default_architecture") or "resnet34"))

        d = cfg.get("data") or {}
        self._d_raw.setText(str(d.get("raw_data_dir") or ""))
        self._d_out.setText(str(d.get("data_dir") or ""))
        self._d_test_pc.setValue(int(d.get("test_per_class") or 0))
        self._d_im_size.setValue(int(d.get("image_size") or 224))
        bs = d.get("batch_size")
        self._d_batch.setText("" if bs is None else str(int(bs)))
        self._d_img_types.setPlainText(_list_to_ext_lines(d.get("image_types")))
        self._d_vid_types.setPlainText(_list_to_ext_lines(d.get("video_types")))
        self._d_class_names.setPlainText(_class_names_to_text(d.get("class_names")))
        cq = d.get("class_qualifying_subdir")
        self._d_qual.setText("" if cq is None else str(cq))

        g = d.get("gather") or {}
        self._g_target_n.setValue(int(g.get("default_target_count") or 100000))
        self._g_target_dir.setText(str(g.get("default_target_dir") or ""))
        self._g_rej.setText(str(g.get("default_rejected_dir") or ""))
        self._g_raw.setText(str(g.get("default_raw_data_dir") or ""))

        t = cfg.get("training") or {}
        self._t_frozen_e.setValue(int(t.get("frozen_epochs") or 0))
        self._t_unfrozen_e.setValue(int(t.get("unfrozen_epochs") or 0))
        self._t_frozen_lr.setValue(float(t.get("frozen_lr") or 0.0))
        self._t_umax.setValue(float(t.get("unfrozen_lr_max") or 0.0))
        self._t_umin.setValue(float(t.get("unfrozen_lr_min") or 0.0))
        self._t_workers.setValue(int(t.get("num_workers") or 0))
        self._store_ck.setChecked(bool(t.get("store_checkpoints")))

        p = cfg.get("paths") or {}
        self._p_models.setText(str(p.get("models_dir") or ""))
        self._p_logs.setText(str(p.get("logs_dir") or ""))
        self._p_timing.setText(str(p.get("timing_dir") or ""))

    def _sync_yaml_editor(self, cfg: dict[str, Any]) -> None:
        sub = {k: cfg[k] for k in PIPELINE_ROOT_KEYS if k in cfg}
        try:
            text = yaml.dump(sub, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception:
            text = ""
        self._yaml_edit.setPlainText(text)

    def _refresh_from_disk(self) -> None:
        from utils.config import get_user_pipeline_config_path

        mw = self._main_window()
        if mw is not None:
            reload_pipeline_config(mw._effective_pipeline_config_path(), force=True)
        else:
            reload_pipeline_config(None, force=True)

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

        self._apply_dict_to_form(pc.to_dict())
        self._sync_yaml_editor(pc.to_dict())

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
                    "The document contains keys outside model/data/training/paths: {keys}\n"
                    "They will be ignored when applying to the form. Continue?"
                ).format(keys=", ".join(extra)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        merged = {k: parsed[k] for k in PIPELINE_ROOT_KEYS if k in parsed}
        if len(merged) < 1:
            qt_alert(self, _("Pipeline"), _("No model/data/training/paths keys found."))
            return
        self._apply_dict_to_form(merged)
        self._sync_yaml_editor(merged)

    def _on_save(self) -> None:
        from utils.config import get_user_pipeline_config_path

        mw = self._main_window()
        cfg = self._full_pipeline_dict_from_form()
        target = (
            resolve_pipeline_save_path(mw._workspace) if mw is not None else get_user_pipeline_config_path()
        )
        try:
            save_pipeline_yaml(target, cfg)
        except OSError as e:
            qt_alert(self, _("Pipeline"), _("Could not write:\n{path}\n\n{err}").format(path=target, err=e))
            return

        if mw is not None:
            mw.reload_mb_yaml_config()
        else:
            reload_pipeline_config(target, force=True)
        self._refresh_from_disk()
        qt_alert(self, _("Pipeline"), _("Saved to:\n{path}").format(path=target))

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
