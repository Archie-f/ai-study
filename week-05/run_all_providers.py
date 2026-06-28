from dotenv import load_dotenv
load_dotenv()

from provider import LLMProvider, LLMResult
from anthropic_provider import AnthropicProvider
from openai_provider import OpenAIProvider
from ollama_provider import OllamaProvider
from groq_provider import GroqProvider


PROVIDERS: dict[str, LLMProvider] = {
    "claude": AnthropicProvider(),
    "gpt": OpenAIProvider(),
    "groq": GroqProvider(),
    "ollama": OllamaProvider(),
}

def run_all(prompt: str) -> dict[str, LLMResult]:
    """Run prompt through all registered providers.

    Returns a dict mapping provider name -> LLMResult.
    """
    results: dict[str, LLMResult] = {}
    for name, provider in PROVIDERS.items():
        results[name] = provider.ask(prompt)

    return results

def print_results(results: dict[str, LLMResult]) -> None:
    """Print raw results from all providers."""

    for name, result in results.items():
        print(f"""-------{name}-------
        Text: {result.text}
        Tokens in: {result.tokens_in}
        Tokens out: {result.tokens_out}
        Cost: ${result.cost_usd():.6f}
        Latency: {result.latency_ms} ms.
""")

if __name__ == "__main__":
    inp: str = "Explain what an API is in one sentence."
    print_results(run_all(inp))
