"""
Shared helper for showing/hiding ``QFormLayout`` rows based on the selected
:class:`~mb.models.types.ModelType`.

Introduced for LoRA support: multiple pages (Train, Data/Create Dataset) now have field
groups that only make sense for a subset of model types (e.g. frozen/unfrozen epoch
schedules for image classification vs. rank/alpha/base-model fields for
``image_generation_lora``). Centralizing the toggle mechanics here means each page only
has to declare *which* widget belongs to *which* model types, not re-derive the
show/hide logic itself.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from PySide6.QtWidgets import QFormLayout, QWidget

from mb.models.types import ModelType


def apply_model_type_field_visibility(
    form: QFormLayout,
    model_type: ModelType,
    field_visibility: Mapping[QWidget, Iterable[ModelType]],
) -> None:
    """
    For each ``(field_widget, allowed_model_types)`` pair, show *field_widget*'s row in
    *form* iff *model_type* is one of ``allowed_model_types`` — hide it otherwise.

    *field_widget* must be the widget originally passed to ``form.addRow(label, widget)``
    (``QFormLayout.setRowVisible`` accepts either the row's label or field widget and
    toggles both).
    """
    for field_widget, allowed in field_visibility.items():
        form.setRowVisible(field_widget, model_type in allowed)
