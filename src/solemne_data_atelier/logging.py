from dataclasses import dataclass
import logging
import logging.config

from typing import (
    TYPE_CHECKING,
    Literal,
    Union
)

__package_name__ = "solemne_data_atelier"

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


@dataclass
class ModuleLoggingConfig:
    default_level: Union[int, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = logging.INFO
    log_file: Union[str, "PathLike"] = f"{__package_name__}.log"

DEFAULT_LOGGING_CONFIG = ModuleLoggingConfig()


def setup_logging(
        config: ModuleLoggingConfig = DEFAULT_LOGGING_CONFIG,
):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": logging.INFO
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "level": logging.DEBUG,
                "filename": config.log_file,
                "mode": "a",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": config.default_level,
        },
    }

    logging.config.dictConfig(logging_config)
