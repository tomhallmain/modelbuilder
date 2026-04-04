from __future__ import annotations

from enum import Enum

from mb.models.types import ModelBuildStepCommand
from mb.utils.translations import _


class ModelBuilderTaskType(str, Enum):
    """
    Categories of long-running work started from the desktop shell.

    Values are persisted in recent-run history. :meth:`nav_row_index` matches
    the order of :attr:`ui.main_window.MainWindow.NAV_PAGE_SPECS` (Home at 0).
    """

    DATA = "data"
    TRAIN = "train"
    CONVERT = "convert"

    @property
    def nav_row_index(self) -> int:
        """Stacked / sidebar row for this task's primary page."""
        return _MODEL_BUILDER_TASK_NAV_ROW[self]


# Must stay in sync with MainWindow.NAV_PAGE_SPECS (Home, Data, Train, Convert, …).
_MODEL_BUILDER_TASK_NAV_ROW: dict[ModelBuilderTaskType, int] = {
    ModelBuilderTaskType.DATA: 1,
    ModelBuilderTaskType.TRAIN: 2,
    ModelBuilderTaskType.CONVERT: 3,
}


# Backwards-compatible name: same as :class:`~mb.models.types.ModelBuildStepCommand`
# (includes ``estimate-space`` for CLI; Data page uses :meth:`~mb.models.types.ModelBuildStepCommand.data_page_tab_values`).
DataPipelineSubcommand = ModelBuildStepCommand


class AppInfo:
    SERVICE_NAME = "MyPersonalApplicationsService"
    APP_IDENTIFIER = "modelbuilder"


class ActionType(Enum):
    """Enumeration of supported action types for notifications."""
    SYSTEM = "system"  # For general system notifications

    def get_translation(self):
        """Get the translated string for this action type."""
        if self == ActionType.SYSTEM:
            return _("System")
        raise Exception("Unhandled action type translation: " + str(self))


class ProtectedActions(Enum):
    """Enumeration of actions that can be password protected."""
    ACCESS_ADMIN = "access_admin"
    
    @staticmethod
    def get_action(action_name: str):
        """Get the ProtectedActions enum value for a given action name."""
        try:
            return ProtectedActions(action_name.lower().replace(" ", "_"))
        except ValueError:
            return None

    def get_description(self):
        """Get the user-friendly description for this action."""
        descriptions = {
            ProtectedActions.ACCESS_ADMIN: _("Access Password Administration")
        }
        return descriptions.get(self, self.value)
