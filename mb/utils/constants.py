from __future__ import annotations

from enum import Enum

from mb.utils.translations import I18N

_ = I18N._


class ModelBuilderTaskType(str, Enum):
    """
    Categories of long-running work started from the desktop shell.

    Values are persisted in recent-run history. :meth:`nav_row_index` matches
    the order of :attr:`ui.main_window.MainWindow.NAV_ITEMS` (Home at 0).
    """

    DATA = "data"
    TRAIN = "train"
    CONVERT = "convert"

    @property
    def nav_row_index(self) -> int:
        """Stacked / sidebar row for this task's primary page."""
        return _MODEL_BUILDER_TASK_NAV_ROW[self]


# Must stay in sync with MainWindow.NAV_ITEMS (Home, Data, Train, Convert, …).
_MODEL_BUILDER_TASK_NAV_ROW: dict[ModelBuilderTaskType, int] = {
    ModelBuilderTaskType.DATA: 1,
    ModelBuilderTaskType.TRAIN: 2,
    ModelBuilderTaskType.CONVERT: 3,
}


class DataPipelineSubcommand(str, Enum):
    """
    ``mb data <subcommand>`` values (CLI and Data page tabs).

    Stored on recent-run history rows when :attr:`ModelBuilderTaskType` is ``DATA``.
    """

    GATHER = "gather"
    CONVERT = "convert"
    DEDUPLICATE = "deduplicate"
    UPSCALE = "upscale"
    CREATE_DATASET = "create-dataset"

    @classmethod
    def try_from(cls, raw: object) -> DataPipelineSubcommand | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return None


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
