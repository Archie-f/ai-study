from dotenv import load_dotenv

from llm_compare.eval.types import EvalCase, EvalResult
from llm_compare.eval.harness import run_eval, print_report
from llm_compare.eval.dataset import load_eval_dataset
from llm_compare.eval.batch_runner import run_batch
from llm_compare.eval.report import generate_report

load_dotenv()

from llm_compare.providers.groq_provider import GroqProvider
from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from llm_compare.providers.ollama_provider import OllamaProvider
from llm_compare.providers.base import LLMProvider

SYSTEM_PROMPT = """
You are responsible to evaluate the user reviews. 
Evaluate the inputs and response with only one word as positive, negative or neutral.
"""

TASK_DESCRIPTION = "Classify the sentiment as exactly one word: positive, negative, or neutral."

CASES = [
    EvalCase(prompt="The product is great, I love it!", expected="positive", category="sentiment", task_description=TASK_DESCRIPTION),
    EvalCase(prompt="This is the worst experience I have ever had.", expected="negative", category="sentiment", task_description=TASK_DESCRIPTION),
    EvalCase(prompt="The package arrived on time.", expected="positive", category="sentiment", task_description=TASK_DESCRIPTION),
    EvalCase(prompt="I am so happy with my purchase!", expected="positive", category="sentiment", task_description=TASK_DESCRIPTION),
    EvalCase(prompt="The instructions were unclear and confusing.", expected="negative", category="sentiment", task_description=TASK_DESCRIPTION),
]

PROVIDERS: dict[str, LLMProvider] = {
    "claude": AnthropicProvider(),
    "open_ai": OpenAIProvider(),
    "groq": GroqProvider(),
    "ollama": OllamaProvider(),
}

def run_mini_eval(cases: list[EvalCase]) -> None:
    """Run a mini evaluation suite with exact-match and LLM-as-judge scorers and print both reports."""
    print("Running eval for exact-match...")
    results_exact_match: list[EvalResult] = run_eval(cases, provider=GroqProvider(), system_prompt=SYSTEM_PROMPT)
    print_report(results_exact_match)

    print('=' * 60)

    print("Running eval for llm-as-judge...")
    results_llm_as_judge: list[EvalResult] = run_eval(
        cases,
        provider=GroqProvider(),
        judge=AnthropicProvider(),
        system_prompt=SYSTEM_PROMPT
    )
    print_report(results_llm_as_judge)

def run_batch_runner(cases: list[EvalCase]):
    factual_system_prompt = "Answer with the requested value only. No explanation, no full sentences."

    all_results: dict[str, list[EvalResult]] = run_batch(
        cases,
        providers=PROVIDERS,
        judge=AnthropicProvider(),
        system_prompt=factual_system_prompt,
    )
    generate_report(all_results)

if __name__ == "__main__":
    dataset = load_eval_dataset()
    run_batch_runner(dataset)
