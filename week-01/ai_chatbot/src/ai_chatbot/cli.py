from typing import Optional

import httpx
import typer

URL: str = "http://localhost:11434/api/"
MODEL: str = "llama3"
AI_READY_MESSAGE: str = "Chatbot ready. Type 'quit' or 'bye' to exit."
AI_GOODBYE_MESSAGE: str = "AI: Goodbye!"
SYSTEM_PROMPT: str = """
You are a helpful Python mentor.
Explain programming concepts clearly for beginners.
Use simple language and short examples.
""".strip()

type MessageParam = list[dict[str,str]]

def build_system_message(system_prompt: str) -> MessageParam:
    return [
    {
        "role": "system",
        "content": system_prompt
    }
]

app = typer.Typer()

def get_user_input() -> str:
    return input("You: ")

def format_message(input_message: str, is_user_input: bool) -> MessageParam:
    role = "assistant"
    if is_user_input:
        role = "user"

    return [{
        "role": role,
        "content": input_message
    }]

def append_message(
        message_history: MessageParam,
        message_to_append: str,
        is_user_input: bool
) -> None:
    formatted_message = format_message(message_to_append, is_user_input)
    message_history.extend(formatted_message)

def limit_message_history(message_history: MessageParam, limit: int) -> MessageParam:
    return message_history[-limit:]


@app.command()
def chat(model: str = typer.Option(MODEL, help="The model to use."),
    system_prompt: Optional[str] = typer.Option(None, help="Custom system prompt."),
    max_history: int = typer.Option(8, help="The max number of messages to keep in the chat history.")
    ):
    """Starts a chatbot conversation."""
    resolved_prompt = system_prompt or SYSTEM_PROMPT
    system_message = build_system_message(resolved_prompt)
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
        prompt = system_message + limit_message_history(messages, max_history)

        try:
            response = httpx.post(
                f"{URL}chat",
                json={
                    "model": model,
                    "messages": prompt,
                    "stream": False
                },
                timeout=60.0
            )
            # Catch HTTP errors like 404, 500
            response.raise_for_status()
            # Get content from the response. Raise a friendly message if there's none.
            ai_reply = response.json().get("message", {}).get("content", "Sorry, I didn't get a response. Please try again.")
            typer.echo(f"AI: {ai_reply}")
            append_message(messages, ai_reply, is_user_input=False)
        except httpx.ConnectError:
            return "Error: Could not connect to Ollama. Is it running?"
        except httpx.HTTPStatusError as error:
            return f"Error: Model returned error: '{error.response.text}."
        except httpx.ReadTimeout:
            typer.echo("Error: Ollama took too long to respond. Try a shorter prompt.")

@app.command()
def models():
    """List all downloaded Ollama models."""
    try:
        response = httpx.get(f"{URL}tags", timeout=60.0)
        # Catch HTTP errors like 404, 500
        response.raise_for_status()
        # Get content from the response. Raise a friendly message if there's none.
        model_list = response.json().get("models", [])

        if not model_list:
            typer.echo("No models downloaded yet. Run: ollama pull llama3")

        number_of_models = len(model_list)
        model_word = "model"
        if number_of_models > 1:
            f"{model_word}s"

        typer.echo(f"{len(model_list)} {model_word} downloaded in total.")
        for model in model_list:
            typer.echo(f" {model['name']}")

    except httpx.ConnectError:
        return "Error: Could not connect to Ollama. Is it running?"
    except httpx.HTTPStatusError as error:
        return f"Error: Model returned error: '{error.response.text}."
    except httpx.ReadTimeout:
        typer.echo("Error: Ollama took too long to respond. Try a shorter prompt.")


if __name__ == "__main__":
    app()