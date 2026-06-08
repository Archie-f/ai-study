"""Chatbot module:
- Stores and manages the chat history
- Sends messages to Ollama
- Runs the chat loop and prints the conversation
"""

type MessageParam = list[dict[str, str]]

def add_message(history: MessageParam, role: str, content:str) -> None:
    """Adds a message to the chat history.

    Args:
        history: Messages stored in the chat history.
        role: Distinction of whom the message sent belongs to.
        content: Content of the message.

    """

def send(history: MessageParam, model: str = "llama3", host: str = "http://localhost:11434") -> str:
    """Sends the chat history to Ollama.

    Args:
        history: Messages stored in the chat history.
        model: Model name that is used to send the chat history.
        host: Model server url.

    Returns:
        The assistant's reply to the message as a string.
    """

def run(model: str = "llama3", host: str = "http://localhost:11434", max_turns: int = 50) -> None:
    """Runs the chat loop. Reads user input, calls send(), and prints the conversation.

    Args:
        model: Model name that is used to send the chat history.
        host: Model server url.
        max_turns: Maximum number of conversation turns before the loop ends.
    """
