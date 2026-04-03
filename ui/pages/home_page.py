"""Home page with summary, quick navigation, and recent run history."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mb.utils.constants import ModelBuilderTaskType
from mb.pipeline_config import get_pipeline_config
from mb.utils.recent_run_history import format_recent_runs_for_display, get_recent_runs
from mb.utils.snapshot import format_latest_unified_snapshot_summary
from mb.utils.translations import _


class HomePage(QWidget):
    """Landing page for the desktop UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("home_page")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._title = QLabel()
        self._title.setTextFormat(Qt.RichText)
        layout.addWidget(self._title)

        self._subtitle = QLabel()
        self._subtitle.setWordWrap(True)
        layout.addWidget(self._subtitle)

        self._summary = QGroupBox()
        summary_layout = QVBoxLayout(self._summary)
        self._bullet_data = QLabel()
        summary_layout.addWidget(self._bullet_data)
        self._bullet_train = QLabel()
        summary_layout.addWidget(self._bullet_train)
        self._bullet_convert = QLabel()
        summary_layout.addWidget(self._bullet_convert)
        self._bullet_config = QLabel()
        summary_layout.addWidget(self._bullet_config)
        layout.addWidget(self._summary)

        self._quick = QGroupBox()
        quick_layout = QHBoxLayout(self._quick)
        self._btn_data = QPushButton()
        self._btn_data.clicked.connect(
            lambda checked=False, r=ModelBuilderTaskType.DATA.nav_row_index: self._open_nav_row(r)
        )
        quick_layout.addWidget(self._btn_data)
        self._btn_train = QPushButton()
        self._btn_train.clicked.connect(
            lambda checked=False, r=ModelBuilderTaskType.TRAIN.nav_row_index: self._open_nav_row(r)
        )
        quick_layout.addWidget(self._btn_train)
        self._btn_convert = QPushButton()
        self._btn_convert.clicked.connect(
            lambda checked=False, r=ModelBuilderTaskType.CONVERT.nav_row_index: self._open_nav_row(r)
        )
        quick_layout.addWidget(self._btn_convert)
        layout.addWidget(self._quick)

        self._snapshot_box = QGroupBox()
        snapshot_layout = QVBoxLayout(self._snapshot_box)
        self._snapshot_detail = QPlainTextEdit()
        self._snapshot_detail.setObjectName("home_snapshot_detail")
        self._snapshot_detail.setReadOnly(True)
        self._snapshot_detail.setMinimumHeight(100)
        snapshot_layout.addWidget(self._snapshot_detail)
        layout.addWidget(self._snapshot_box)

        self._history = QGroupBox()
        history_layout = QVBoxLayout(self._history)
        self._recent_runs = QPlainTextEdit()
        self._recent_runs.setObjectName("home_recent_runs")
        self._recent_runs.setReadOnly(True)
        self._recent_runs.setMinimumHeight(180)
        history_layout.addWidget(self._recent_runs)
        layout.addWidget(self._history, 1)

        layout.addStretch(0)

        self.retranslate_ui()
        self.refresh_recent_runs()

    def retranslate_ui(self) -> None:
        self._title.setText(f"<h2>{_('Home')}</h2>")
        self._subtitle.setText(
            _(
                "Model Builder desktop shell for data prep, training, conversion, and info tools."
            )
        )
        self._summary.setTitle(_("Current scope"))
        self._bullet_data.setText(_("- Data pipeline: gather, convert, dedupe, upscale, create dataset"))
        self._bullet_train.setText(
            _("- Training: in-process or detached (CLI: {cmd})").format(cmd="mb train")
        )
        self._bullet_convert.setText(_("- Model conversion and dataset / model inspection (Info)"))
        self._bullet_config.setText(_("- Application shell settings (Config)"))
        self._quick.setTitle(_("Quick actions"))
        self._btn_data.setText(_("Open Data Page"))
        self._btn_train.setText(_("Open Train Page"))
        self._btn_convert.setText(_("Open Convert Page"))
        self._snapshot_box.setTitle(_("Latest unified snapshot"))
        self._snapshot_detail.setPlaceholderText(_("Loading…"))
        self._history.setTitle(_("Recent run history"))
        self._recent_runs.setPlaceholderText(_("Loading…"))
        self.refresh_recent_runs()
        self.refresh_snapshot_panel()

    def _open_nav_row(self, row: int) -> None:
        from ui.main_window import MainWindow

        w = self.window()
        if isinstance(w, MainWindow):
            w.nav_widget.setCurrentRow(row)

    def refresh_recent_runs(self) -> None:
        text = format_recent_runs_for_display(get_recent_runs())
        self._recent_runs.setPlainText(text)

    def refresh_snapshot_panel(self) -> None:
        pc = get_pipeline_config()
        raw = Path(str(pc.get("data.raw_data_dir", "raw_data")))
        data = Path(str(pc.get("data.data_dir", "data")))
        text = format_latest_unified_snapshot_summary([raw, data])
        if not text.strip():
            self._snapshot_detail.setPlainText(
                _(
                    "No snapshot_*.json found under the configured raw data or data directories.\n"
                    "(Run convert and create-dataset to create one.)"
                )
            )
        else:
            self._snapshot_detail.setPlainText(text)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_recent_runs()
        self.refresh_snapshot_panel()
