import sys
from pathlib import Path

from dotenv import load_dotenv

from eval_types import EvalCase, EvalResult
from eval_harness import run_eval, print_report
from llm_judge import score_with_llm

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent / "week-05"))
from groq_provider import GroqProvider
from anthropic_provider import AnthropicProvider

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
        scorer=score_with_llm,
        judge=AnthropicProvider(),
        system_prompt=SYSTEM_PROMPT
    )
    print_report(results_llm_as_judge)

if __name__ == "__main__":
    run_mini_eval(CASES)
