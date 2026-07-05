import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent / 'week-05'))
from provider import LLMProvider
from anthropic_provider import AnthropicProvider
from openai_provider import OpenAIProvider
from ollama_provider import OllamaProvider
from groq_provider import GroqProvider
from compare import ComparisonResult, run_comparison, print_table

def main() -> None:
    inp: str = "Explain what an API is in one sentence."
    system_message: str = 'You are a helpful assistant.'
    providers_list: list[LLMProvider] = [
        AnthropicProvider(),
        OpenAIProvider(),
        GroqProvider(),
        OllamaProvider()
    ]

    results: ComparisonResult = run_comparison(inp, providers_list, system_message, GroqProvider())
    print_table(results)


if __name__ == '__main__':
    main()