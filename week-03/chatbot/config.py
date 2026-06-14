import logging
from dataclasses import dataclass
from enum import Enum


class Level(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ChatConfig:
    """Configuration settings for the Ollama chatbot."""

    model: str = "llama3"
    host: str = "http://localhost:11434"
    max_turns: int = 10
    system_prompt: str = (
    'You are a helpful assistant. '
    'Be concise. Answer in plain English.'
    )

    log_level: Level = Level.DEBUG
    log_to_file: bool = False
    log_file: str = "chatbot.log"

def setup_logging(config: ChatConfig) -> None:
    """Configures the global application logging framework.

    Initializes the root logger with custom formatting, specific stream and file
    output paths, and sets sensitivity levels based on the application runtime
    configuration. It also squelches noisy upstream HTTP library logs to prevent
    console clutter.

    Args:
        config: An instance of ChatConfig containing application settings such as
            log_level, log_to_file, and log_file path attributes.

    Raises:
        ValueError: If the log level string provided in the config object does
            not match a valid uppercase logging tier (DEBUG, INFO, WARNING,
            ERROR, CRITICAL).
    """

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)

    level = getattr(logging, config.log_level.value.upper())
    fmt = '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.log_to_file:
        handlers.append(logging.FileHandler(config.log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
