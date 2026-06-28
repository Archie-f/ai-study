import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletion,
)

SYSTEM_PROMPT: str = "You are a helpful assistant."
USER_INPUT: str = "Explain what an API is in one sentence."

load_dotenv()
client = OpenAI(
    base_url=os.getenv("GROQ_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def build_prompt(prompt: str) -> list[ChatCompletionMessageParam]:
    system_turn: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    user_turn: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": prompt
    }
    return [system_turn, user_turn]

def call_groq(prompt: str, model: str = 'llama-3.1-8b-instant') -> ChatCompletion:
    """Call Groq API via OpenAI-compatible client and return the response.

        Args:
            prompt (str): The prompt to be sent to model.
            model (str, optional): The model to use. Defaults to llama-3.1-8b-instant.
    """
    return client.chat.completions.create(
        model=model,
        messages=build_prompt(prompt),
        max_tokens=256,
    )

def print_result(response: ChatCompletion) -> None:
    """Print response text and token counts.
    Pricing for Groq's llama-3.1-8b-instant is free.

        Args:
            response (ChatCompletion): The response to print.
    """
    assert response.choices[0].message.content is not None
    assert response.usage is not None
    text: str = response.choices[0].message.content
    tokens_in: int = response.usage.prompt_tokens
    tokens_out: int = response.usage.completion_tokens
    cost: float = 0.0

    print(f"""
Response: {text}
Tokens in: {tokens_in} | Tokens out: {tokens_out} | Cost: ${cost:.6f}""")

if __name__ == "__main__":
    print_result(call_groq(USER_INPUT))
