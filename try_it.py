from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from llm_compare.cost_dashboard import run_comparison_batch
from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from llm_compare.providers.groq_provider import GroqProvider
from llm_compare.providers.ollama_provider import OllamaProvider


path_to_prompts = Path(__file__).parent / "tests" / "prompts.txt"
providers_list = [AnthropicProvider(), OpenAIProvider(), GroqProvider(), OllamaProvider()]

if __name__ == "__main__":
    run_comparison_batch(
        path=path_to_prompts,
        providers=providers_list,
        judge=OpenAIProvider(),
    )
