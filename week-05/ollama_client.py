import os
import time

import requests
from dotenv import load_dotenv


SYSTEM_PROMPT: str = "You are a helpful assistant."
USER_INPUT: str = "Explain what an API is in one sentence."

load_dotenv()

def build_prompt(user_input: str, system_prompt: str = '') -> list[dict[str, str]]:
    """Build a prompt from the user_input.
        Args:
            user_input (str): The user_input to build the prompt from.
            system_prompt (str, optional): The system_prompt to build the prompt from.

        Returns:
            Returns a list of dictionaries containing the user_input, system_prompt, and latency_ms.
    """
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_input,
        }
    ]

def call_ollama(message: str, model: str = "llama3") -> dict:
    """Make a single call to a local Ollama model.

        Args:
            message (str): The prompt to be sent to the model.
            model (str, optional): The model to use. Defaults to "llama3".

        Returns:
             dict with: text, tokens_in, tokens_out, latency_ms.
    """
    prompt: list[dict[str,str]] = build_prompt(user_input=message, system_prompt=SYSTEM_PROMPT)

    start_time: float = time.perf_counter()
    response = requests.post(
        f"{os.getenv("OLLAMA_BASE_URL")}/api/chat",
        json={
            "model": model,
            "messages": prompt,
            "stream": False
        }
    )
    elapsed_time: float = (time.perf_counter() - start_time) * 1000

    response.raise_for_status()

    data = response.json()
    return {
        "text": data["message"]["content"],
        "tokens_in": data.get("prompt_eval_count", 0),
        "tokens_out": data.get("eval_count", 0),
        "latency_ms": round(elapsed_time)
    }

if __name__ == "__main__":
    for key, value in call_ollama(USER_INPUT).items():
        print(f"{key}: {value}")
