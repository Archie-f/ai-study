"""Chatbot module:
- Stores and manages the chat history
- Sends messages to Ollama
- Runs the chat loop and prints the conversation
"""
import logging
import time

import ollama

from .config import ChatConfig

logger = logging.getLogger(__name__)

type MessageParam = list[dict[str, str]]

EXIT_PHRASES = {'quit', 'exit', 'bye', 'q'}
AI_WELCOME_MESSAGE = "Session started. Welcome to Ollama. How can I help you?"
AI_GOODBYE_MESSAGE = "Goodbye!"
ASSISTANT = "assistant"
ENTER_MESSAGE_NOTIFICATION = "Assistant: Please enter a message"
USER = "user"

def add_message(history: MessageParam, role: str, content:str) -> None:
    """Adds a message to the chat history.

    Args:
        history: Messages stored in the chat history.
        role: Distinction of whom the message sent belongs to.
        content: Content of the message.

    """
    history.append({"role": role, "content": content})

def send(history: MessageParam, config: ChatConfig) -> str:
    """Sends the chat history to Ollama.

    Args:
        history: Messages stored in the chat history.
        config: ChatConfig object.

    Returns:
        The assistant's reply to the message as a string. On error, returns an error message string.
    """

    client = ollama.Client(config.host)
    start = time.monotonic()
    try:
        logger.debug("Sending '%d' messages to the model: '%s'.", len(history) ,config.model)
        response = client.chat(
            model=config.model,
            messages=history
        )
        elapsed = time.monotonic() - start
        reply: str = response["message"]["content"]
        logger.info("Ollama response in %.2fs, (%d chars).", elapsed, len(reply))
        return reply
    except ollama.ResponseError as err:
        logger.error("Sorry, Ollama has encountered a problem: %s", err)
        return "Error: Something went wrong with the model."
    except Exception as exc:
        logger.exception("Sorry, an error occurred: %s.", exc)
        return "Error: Something went wrong. Is Ollama still running?"

def run(config: ChatConfig) -> None:
    """Runs the chat loop. Reads user input, calls send(), and prints the conversation.

    Args:
        config: ChatConfig object.
    """

    message_history = [{"role": "system", "content": config.system_prompt}]
    count = 0

    logger.info(AI_WELCOME_MESSAGE)
    while count < config.max_turns:
        user_input = input("You: ").strip()

        if not user_input:
            print(ENTER_MESSAGE_NOTIFICATION)
            continue

        if user_input.lower() in EXIT_PHRASES:
            print(AI_GOODBYE_MESSAGE)
            break

        add_message(message_history, role=USER, content=user_input)
        ai_reply = send(message_history, config)
        add_message(message_history, role=ASSISTANT, content=ai_reply)
        print(f"{ASSISTANT.capitalize()}: {ai_reply}")
        print("---")

        count += 1
    else:
        print(f"Session reached to limit of {config.max_turns} turns.")

    logger.info("Session ended. %s", AI_GOODBYE_MESSAGE)