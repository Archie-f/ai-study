import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
import typer

AI_READY_MESSAGE: str = "Chatbot ready. Type 'quit' or 'bye' to exit."
AI_GOODBYE_MESSAGE: str = "AI: Goodbye!"
SYSTEM_PROMPT: str = """
You are a helpful Python mentor.
Explain programming concepts clearly for beginners.
Use simple language and short examples.
""".strip()

type MessageParam = list[dict[str,str]]
type ChatMessages = list[ChatMessage]
app = typer.Typer()

@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class ChatConfig:
    url: str = "http://localhost:11434/api/chat"
    model: str = "llama3"
    system_prompt: str = SYSTEM_PROMPT
    max_history: int = 6

def get_user_input() -> str:
    return input("You: ")

def format_message(input_message: str, is_user_input: bool) -> ChatMessage:
    role = "assistant"
    if is_user_input:
        role = "user"

    return ChatMessage(role, input_message)

def append_message(
        message_history: ChatMessages,
        message_to_append: str,
        is_user_input: bool
) -> None:
    formatted_message = format_message(message_to_append, is_user_input)
    message_history.append(formatted_message)

def build_messages(config: ChatConfig, message_history: ChatMessages) -> MessageParam:
    system_msg: dict[str, str] = {"role": "system", "content": config.system_prompt}
    limited_history = message_history[-config.max_history:]
    messages = [message.to_dict() for message in limited_history]
    return [system_msg] + messages

async def send_request_to_model(config: ChatConfig, message_history: ChatMessages) -> str | None:
    prompt = build_messages(config, message_history)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config.url,
                json ={
                    "model": config.model,
                    "messages": prompt,
                    "stream": False
                },
                timeout = 60.0
            )
        # Catch HTTP errors like 404, 500
        response.raise_for_status()
        # Get content from the response. Raise a friendly message if there's no content.
        return response.json().get("message", {}).get("content", "Sorry, no response received. Please try again.")
    except httpx.ConnectError:
        return "Error: Could not connect to Ollama. Is it running?"
    except httpx.HTTPStatusError as error:
        return f"Error: Model returned error: '{error.response.text}."
    except httpx.ReadTimeout:
        return "Error: Ollama took too long to respond. Try a shorter prompt."

async def _chat(model: str, system_prompt: Optional[str], max_history: int):
    chat_config = ChatConfig()
    if model:
        chat_config.model = model
    if system_prompt:
        chat_config.system_prompt = system_prompt
    if max_history:
        chat_config.max_history = max_history

    messages = []
    typer.echo(f"Starting chat | model = {model}, max_history = {max_history}")
    typer.echo(AI_READY_MESSAGE)

    while True:
        user_input = get_user_input()

        if user_input.lower() == "quit" or user_input.lower() == "bye":
            typer.echo(AI_GOODBYE_MESSAGE)
            append_message(messages, user_input, is_user_input=True)
            append_message(messages, AI_GOODBYE_MESSAGE, is_user_input=False)
            break

        append_message(messages, user_input, is_user_input=True)

        ai_reply = await send_request_to_model(chat_config, messages)
        typer.echo(f"AI: {ai_reply}")
        append_message(messages, ai_reply, is_user_input=False)

@app.command()
def chat(
    model: str = typer.Option(ChatConfig.model, help="The model to use."),
    system_prompt: Optional[str] = typer.Option(None, help="Custom system prompt."),
    max_history: int = typer.Option(ChatConfig.max_history, help="The max number of messages to keep in the chat history.")
):
    asyncio.run(_chat(model, system_prompt, max_history))

if __name__ == "__main__":
    app()