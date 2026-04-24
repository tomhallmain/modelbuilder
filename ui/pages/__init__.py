"""Page modules for the Model Builder desktop UI."""

from ui.pages.config_page import ConfigPage
from ui.pages.convert_page import ConvertPage
from ui.pages.data_page import DataPage
from ui.pages.export_page import ExportPage
from ui.pages.home_page import HomePage
from ui.pages.info_page import InfoPage
from ui.pages.pipeline_config_page import PipelineConfigPage
from ui.pages.train_page import TrainPage

__all__ = [
    "HomePage",
    "DataPage",
    "TrainPage",
    "ConvertPage",
    "ExportPage",
    "ConfigPage",
    "PipelineConfigPage",
    "InfoPage",
]
