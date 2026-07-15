from dotenv import load_dotenv
load_dotenv()

from llm_compare.cost_dashboard import run_comparison_batch
from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from llm_compare.providers.groq_provider import GroqProvider
from llm_compare.providers.ollama_provider import OllamaProvider

PROMPTS = [
    "What year did the Berlin Wall fall, and who was the German chancellor at the time?",
    "Summarize this in two sentences: The README is the single most important artifact in a "
    "portfolio project, because it's often the only thing a hiring manager reads before deciding "
    "whether to look at the code at all.",
    "Write a Python function that checks if a string is a palindrome, ignoring case and spaces.",
    "A train leaves at 2pm going 60 mph. Another leaves the same station at 3pm going 90 mph in "
    "the same direction. At what time does the second train catch up?",
    "List exactly 3 pros and 3 cons of using a local LLM instead of a hosted API, as a numbered "
    "list, nothing else.",
]

providers_list = [AnthropicProvider(), OpenAIProvider(), GroqProvider(), OllamaProvider()]

if __name__ == "__main__":
    run_comparison_batch(
        prompts=PROMPTS,
        providers=providers_list,
        judge=OpenAIProvider(),
    )
