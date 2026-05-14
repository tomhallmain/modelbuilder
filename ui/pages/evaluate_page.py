"""Desktop shell placeholder for ``mb evaluate`` (metrics / validation flows TBD)."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from mb.utils.translations import _


class EvaluatePage(QWidget):
    """Skeleton page aligned with the ``evaluate`` top-level CLI command."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        root.addStretch(1)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self._head.setText(f"<h2>{_('Evaluate')}</h2>")
        self._intro.setText(
            _(
                "This area will host model and dataset evaluation once CLI subcommands "
                "and backends are connected."
            )
        )
        self._hint.setText(_("For now, use the terminal: {cmd}").format(cmd="mb evaluate run"))
