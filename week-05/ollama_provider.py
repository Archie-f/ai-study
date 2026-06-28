import os
import time

import requests

from provider import *

class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3"):
        """Store model name."""
        self.model = model

    def ask(self, user_input: str, system_prompt: str = '') -> LLMResult:
        """Call to a local Ollama model and return unified LLMResult.

            Args:
                user_input: User input to be sent to the model.
                system_prompt: System prompt to be sent to the model.

            Returns:
                Unified LLMResult containing text, token counts, cost, and latency.
        """
        try:
            start_time: float = time.perf_counter()
            response = requests.post(
                f"{os.getenv("OLLAMA_BASE_URL")}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ],
                    "stream": False,
                },
                timeout=120
            )
            elapsed_time: float = (time.perf_counter() - start_time) * 1000
            response.raise_for_status()
            data = response.json()

            return LLMResult(
                provider="ollama",
                model=self.model,
                text=data["message"]["content"],
                tokens_in=data.get("prompt_eval_count", 0),
                tokens_out=data.get("eval_count", 0),
                latency_ms=round(elapsed_time),
            )
        except requests.exceptions.ConnectionError:
            return LLMResult(
                provider="ollama",
                model=self.model,
                text="ERROR: Ollama is not available (connection refused)",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
            )


def run(prompt: str) -> None:
    """Run the same prompt across all available providers and print results."""
    provider: LLMProvider = OllamaProvider()
    result = provider.ask(prompt)
    print(f"Provider: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Text: {result.text[:80]}...")
    print(f"Cost: ${result.cost_usd():.6f}")
    print(f"Latency: {result.latency_ms} ms")
    print()

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    inp: str = "Explain what an API is in one sentence."
    run(inp)