import sys
from pathlib import Path
from typing import Callable, Union

from eval_types import EvalCase, EvalResult
from exact_match import score_exact

sys.path.append(str(Path(__file__).parent.parent / "week-05"))
from provider import LLMProvider


def run_eval(
    cases: list[EvalCase],
    provider: LLMProvider,
    scorer: Callable = score_exact,
    judge: Union[LLMProvider, None] = None,
    system_prompt: Union[str, None] = None,
) -> list[EvalResult]:
    """Run all eval cases through a provider and score each result.

        Args:
            cases: list of EvalCase objects to evaluate
            provider: LLMProvider used to generate outputs
            scorer: scoring function, defaults to score_exact
            judge: optional LLMProvider used by LLM-as-judge scorer
            system_prompt: optional system prompt passed to the provider
    """
    results = []
    for case in cases:
        output = provider.ask(user_input=case.prompt, system_prompt=system_prompt).text

        if judge:
            results.append(scorer(case, output, judge))
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
