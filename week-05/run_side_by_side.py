from provider import LLMProvider
from anthropic_provider import AnthropicProvider
from openai_provider import OpenAIProvider
from groq_provider import GroqProvider


def run_side_by_side(prompt: str) -> None:
    """Run the same prompt across all available providers and print results."""
    providers: list[LLMProvider] = [AnthropicProvider(), OpenAIProvider(), GroqProvider()]
    for provider in providers:
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
    user_input: str = "Explain what an API is in one sentence."
    run_side_by_side(user_input)