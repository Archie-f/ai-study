import sys
from pathlib import Path
from typing import Callable, Union

from eval_types import EvalCase, EvalResult
from exact_match import score_exact
from llm_judge import score_with_llm

sys.path.append(str(Path(__file__).parent.parent / "week-05"))
from provider import LLMProvider

SCORER_DISPATCH: dict[str, Callable] = {
    "factual":       score_exact,
    "sentiment":     score_exact,
    "summarization": score_with_llm,
}

def get_scorer(category: str) -> Callable:
    """Return scorer type for a given category. Defaults to exact."""
    if category not in SCORER_DISPATCH:
        raise ValueError(f"Unknown category: {category!r}. Add it to SCORER_DISPATCH.")

    return SCORER_DISPATCH[category]

def run_eval(
    cases: list[EvalCase],
    provider: LLMProvider,
    judge: Union[LLMProvider, None] = None,
    system_prompt: Union[str, None] = None,
) -> list[EvalResult]:
    """Run all eval cases through a provider and score each result.

        Args:
            cases: list of EvalCase objects to evaluate
            provider: LLMProvider used to generate outputs
            judge: optional LLMProvider used by LLM-as-judge scorer
            system_prompt: optional system prompt passed to the provider
    """
    results = []
    for case in cases:
        output = provider.ask(user_input=case.prompt, system_prompt=system_prompt).text
        scorer = get_scorer(case.category)

        if scorer is score_with_llm and judge:
            results.append(scorer(case, output, judge))
        elif scorer is score_with_llm and not judge:
            results.append(score_exact(case, output))
        else:
            results.append(scorer(case, output))

    return results

def print_report(results: list[EvalResult]) -> None:
    """Print a summary table of eval results."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{passed} passed, in total of {total}. ({(passed / total) * 100:.0f}%)")
    print("-" * 60)

    for r in results:
        status = 'PASSED' if r.passed else 'FAILED'
        print(f"{status} - [{r.case.category}] {r.case.prompt}")

        if not r.passed: print(f"Reason: {r.reason[:40]!r}") # truncated for terminal readability
