"""Chatbot module:
- Stores and manages the chat history
- Sends messages to Ollama
- Runs the chat loop and prints the conversation
"""
import logging
import ollama

logger = logging.getLogger(__name__)

type MessageParam = list[dict[str, str]]

EXIT_PHRASES = {'quit', 'exit', 'bye', 'q'}
AI_WELCOME_MESSAGE = "Session started. Welcome to Ollama. How can I help you?"
AI_GOODBYE_MESSAGE = "Goodbye!"
ASSISTANT = "assistant"
ENTER_MESSAGE_NOTIFICATION = "Assistant: Please enter a message"
USER = "user"
SYSTEM_PROMPT = (
    'You are a helpful assistant. '
    'Be concise. Answer in plain English.'
)

def add_message(history: MessageParam, role: str, content:str) -> None:
    """Adds a message to the chat history.

    Args:
        history: Messages stored in the chat history.
        role: Distinction of whom the message sent belongs to.
        content: Content of the message.

    """
    history.append({"role": role, "content": content})

def send(history: MessageParam, model: str = "llama3", host: str = "http://localhost:11434") -> str:
    """Sends the chat history to Ollama.

    Args:
        history: Messages stored in the chat history.
        model: Model name that is used to send the chat history.
        host: Model server url.

    Returns:
        The assistant's reply to the message as a string. On error, returns an error message string.
    """

    client = ollama.Client(host)

    try:
        logger.debug("Sending message: '%d' to the model: '%s'.", len(history) ,model)
        response = client.chat(
            model=model,
            messages=history
        )
        logger.info("Ollama response: %d", len(response["message"]['content']))
        return response["message"]['content']
    except ollama.ResponseError as err:
        logger.error("Sorry, Ollama has encountered a problem: %s", err)
        return "Error: Something went wrong with the model."
    except Exception as e:
        logger.error("Sorry, an error occurred: %s.", e)
        return "Error: Something went wrong. Is Ollama still running?"

def run(model: str = "llama3", host: str = "http://localhost:11434", max_turns: int = 50) -> None:
    """Runs the chat loop. Reads user input, calls send(), and prints the conversation.

    Args:
        model: Model name that is used to send the chat history.
        host: Model server url.
        max_turns: Maximum number of conversation turns before the loop ends.
    """

    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    count = 0

    logger.info(AI_WELCOME_MESSAGE)
    while count < max_turns:
        user_input = input("You: ").strip()

        if not user_input:
            print(ENTER_MESSAGE_NOTIFICATION)
            continue

        if user_input.lower() in EXIT_PHRASES:
            print(AI_GOODBYE_MESSAGE)
            break

        add_message(message_history, role=USER, content=user_input)
        ai_reply = send(message_history, model=model, host=host)
        add_message(message_history, role=ASSISTANT, content=ai_reply)
        print(f"{ASSISTANT.capitalize()}: {ai_reply}")
        print("---")

        count += 1
    else:
        print(f"Session reached to limit of {max_turns}.")

    logger.info("Session ended. %s", AI_GOODBYE_MESSAGE)