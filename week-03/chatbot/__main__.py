"""Entry point for python -m chatbot."""

from .config import ChatConfig, setup_logging
from .chatbot import run

if __name__ == "__main__":
    config = ChatConfig()
    setup_logging(config)
    run(config)
