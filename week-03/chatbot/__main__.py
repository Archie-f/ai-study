import logging

from .chatbot import run

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)-8s - %(name)-12s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)

if __name__ == "__main__":
    run()
