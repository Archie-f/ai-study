import os

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

providers: list[str] = ["ollama", "claude", "groq"]
prompt: str = "Say hello in one sentence."

def ask(content: str, provider: str = "claude") -> str:
    """Send a prompt to an LLM provider and return the reply.

        Supports three providers: "claude" (Anthropic), "groq" (Groq cloud),
        and "ollama" (local). Provider credentials are loaded from .env via
        python-dotenv.

        Args:
            content: The user prompt to send.
            provider: The LLM backend to use. One of "claude", "groq",
                "ollama". Defaults to "claude".

        Returns:
            A formatted string containing the provider name, the model reply,
            and token usage. On error, returns an error message string.

        Raises:
            ValueError: If an unsupported provider name is given.
        """
    max_token: int = 150
    message = [
        {"role": "user", "content": content}
    ]

    try:
        if provider.lower() == "ollama":
            client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL"),
                api_key=os.getenv("OLLAMA_API_KEY"),
            )
            response = client.chat.completions.create(
                model=os.getenv("OLLAMA_MODEL_NAME"),
                messages=message,
                max_tokens=max_token
            )
            ai_reply = response.choices[0].message.content
            token_data = response.usage.total_tokens
        elif provider.lower() == "groq":
            client = OpenAI(
                base_url=os.getenv("GROQ_BASE_URL"),
                api_key=os.getenv("GROQ_API_KEY"),
            )
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL_NAME"),
                messages=message,
                max_tokens=max_token
            )
            ai_reply = response.choices[0].message.content
            token_data = response.usage.total_tokens
        elif provider.lower() == "claude":
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL_NAME"),
                messages=message,
                max_tokens=max_token
            )
            ai_reply = response.content[0].text
            token_data = [response.usage.input_tokens, response.usage.output_tokens]
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return f"""
-----{provider}-----
assistant: \"{ai_reply}\"
Token usage data: {token_data if not provider == "claude" else f"Input: {token_data[0]}, output: {token_data[1]}"}
"""
    except Exception as e:
        return f"{provider}: Error | {e}"

if __name__ == "__main__":
    for provider_model in providers:
        print(ask(prompt, provider_model))
