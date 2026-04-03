from enum import Enum

from utils.translations import I18N
_ = I18N._


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
