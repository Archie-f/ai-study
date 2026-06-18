import os

from typing import Literal, get_args

import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

load_dotenv()

class ConversationManager:
    """Manages multi-turn conversation history for the Anthropic API."""

    RoleTypes = Literal["user", "assistant"]
    _allowed_roles = get_args(RoleTypes)

    def __init__(self, system_prompt: str) -> None:
        """
        Initialise with a system prompt.
        The history starts empty.

            Arguments:
                system_prompt (str): The system prompt.
        """
        self._system_prompt: str = system_prompt
        self._history: list[MessageParam] = []

    def add_message(self, role: RoleTypes, content: str) -> None:
        """
        Append a message to history.
        role must be 'user' or 'assistant'.
        Raises ValueError if role is invalid.

            Arguments:
                role (str): The role of the message to be appended.
                content (str): The content of the message to be appended.
        """

        if role not in self._allowed_roles:
            raise ValueError(f"Invalid role {role}. Must be one of {self._allowed_roles}")

        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[MessageParam]:
        """
        Return the full message history as a list of dicts
        in the format expected by client.messages.create().
        """

        return self._history

    def clear(self) -> None:
        """Reset history to empty. System prompt is preserved."""

        self._history: list[MessageParam] = []

    def count_tokens(self, client) -> int:
        """
        Count tokens in current history using the Anthropic token-count API.
        Returns total input tokens if this history were sent now.
        """

        response = client.messages.count_tokens(
            model=os.environ["ANTHROPIC_MODEL_NAME"],
            system=self._system_prompt,
            messages=self.get_history()
        )
        return response.input_tokens

    def _trim_to_fit(self, client, max_tokens: int = 4000) -> None:
        """
        Remove oldest message pairs from history until token count
        is below max_tokens. Always removes in pairs (user + assistant)
        to preserve the alternating role constraint.
        """

        while self.count_tokens(client) > max_tokens and len(self._history) > 2:
            self._history = self._history[2:]

    def send(self, user_input: str, client, max_tokens: int = 512) -> str:
        """
        Add user_input to history, call the API, append the response,
        and return the assistant's reply as a plain string.
        """

        self.add_message("user", user_input)
        self._trim_to_fit(client)
        response = client.messages.create(
            model=os.environ["ANTHROPIC_MODEL_NAME"],
            max_tokens=max_tokens,
            system=self._system_prompt,
            messages=self.get_history(),
        )
        reply_text = response.content[0].text
        self.add_message("assistant", reply_text)
        return reply_text

if __name__ == "__main__":
    conversation_manager = ConversationManager("You are a concise Python tutor. Answer in 2-3 sentences maximum.")
    total_tokens: int = 0
    anthropic_client = anthropic.Anthropic()

    user_inputs: list[str] = [
        "What does the 'self' parameter do in Python?",
        "Is there an equivalent in Java?",
        "Show me a one-line example in Python."
    ]

    for inp in user_inputs:
        print(f"User: {inp}")
        print(f"Assistant: {conversation_manager.send(inp, anthropic_client)}")
        print(f"Tokens used: {conversation_manager.count_tokens(anthropic_client)}")
