import logging
from dataclasses import dataclass

@dataclass
class ChatConfig:
    model: str = "llama3"
    host: str = "http://localhost:11434"
    max_turns: int = 10
    system_prompt: str = (
    'You are a helpful assistant. '
    'Be concise. Answer in plain English.'
    )

    log_level: str = "DEBUG"
    log_to_file: bool = False
    log_file: str = "chatbot.log"

def setup_logging(config: ChatConfig) -> None:
    """Sets up the logging configuration with ChatConfig."""
    level = getattr(logging, config.log_level.upper(), None)
    if level is None:
        raise ValueError(f"Invalid log level: {config.log_level}. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
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
