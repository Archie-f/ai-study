import dataclasses
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

from eval_types import EvalCase, EvalResult
from eval_harness import run_eval
from eval_dataset import Category

sys.path.append(str(Path(__file__).parent.parent / 'week-05'))
from provider import LLMProvider

PRICING = {
    "claude":  {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000},
    "open_ai": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "ollama":  {"input": 0.00,             "output": 0.00},
}

AVG_OUTPUT_TOKENS = 200
RESULTS_DIR = Path("results")

def estimate_cost(dataset: list[EvalCase], providers: list[str]) -> float:
    """Rough cost estimate in USD before running the batch.

        Assumes ~200 output tokens per call and uses pricing for each provider.
    """
    total = 0.0

    for case in dataset:
        input_tokens = len(case.prompt.split()) * 1.3

        for provider in providers:
            if provider not in PRICING:
                raise ValueError(f"Unknown provider: {provider!r}. Add it to PRICING.")

            input_price = input_tokens * PRICING[provider]["input"]
            output_price = AVG_OUTPUT_TOKENS * PRICING[provider]["output"]
            total += input_price + output_price

    return total

def run_batch(
    dataset: list[EvalCase],
    providers: dict[str, LLMProvider],
    judge: Optional[LLMProvider] = None,
    persist: bool = True,
    system_prompt: Optional[str] = ""
) -> dict[str, list[EvalResult]]:
    """Run all cases against all providers. Returns results keyed by provider name.

    Args:
        dataset:   List of EvalCase objects to evaluate.
        providers: Dict mapping provider name → LLMProvider instance.
        judge:     Optional LLMProvider to use for LLM-as-judge scoring.
        persist:   If True, write results to results/batch_YYYY-MM-DD_HH-MM.json.
        system_prompt: Optional system prompt passed to all providers. If None, providers use their default.
    """
    all_results: dict[str, list[EvalResult]] = {}
    for provider_name, provider in providers.items():
        results = run_eval(
            cases=dataset,
            provider=provider,
            judge=judge,
            system_prompt=system_prompt,
        )
        all_results[provider_name] = results

    if persist:
        _persist(all_results)

    return all_results


def _persist(all_results: dict[str, list[EvalResult]]) -> None:
    """Persist batch results to a timestamped JSON file in the results directory.

        Creates the results/ directory if it does not exist. Each call writes a new
        file named batch_YYYY-MM-DD_HH-MM.json containing all provider results as
        plain dicts, suitable for later loading and reporting.

        Args:
            all_results: Dict mapping provider name → list of EvalResult objects.
        """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = RESULTS_DIR / f"batch_{timestamp}.json"

    # Convert dataclasses to plain dicts for JSON serialization
    serializable = {
        name: [dataclasses.asdict(r) for r in results]
        for name, results in all_results.items()
    }
    path.write_text(json.dumps(serializable, indent=2))
    print(f"Results saved to {path}")

def get_categorized_result_numbers(all_results: dict[str, list[EvalResult]]) -> dict[str, dict[str, int]]:
    """Categorize passed results by category."""
    category_classified: dict[str, dict[str, int]] = {
        Category.factual:       {"passed": 0, "total": 0},
        Category.summarization: {"passed": 0, "total": 0},
        Category.sentiment:     {"passed": 0, "total": 0},
    }

    for results in all_results.values():
        for eval_result in results:
            cat = eval_result.case.category
            category_classified[cat]["total"] += 1
            if eval_result.passed:
                category_classified[cat]["passed"] += 1

    return category_classified

def print_batch_summary(all_results: dict[str, list[EvalResult]]) -> None:
    """Print per-provider pass rates after a batch run."""
    print("\n-- Batch Summary ----------------------------")
    for provider_name, results in all_results.items():
        total  = len(results)
        passed = sum(1 for r in results if r.passed)
        rate   = passed / total if total > 0 else 0.0
        print(f"  {provider_name:<10}  {passed}/{total}  ({rate:.0%})")

    print("\n-- Categorized Results ----------------------")
    categorized_results = get_categorized_result_numbers(all_results)
    for cat in categorized_results.keys():
        passed = categorized_results[cat]["passed"]
        total = categorized_results[cat]["total"]
        rate = passed / total if total > 0 else 0.0
        print(f"  {cat:<15}  {passed}/{total:<5}  ({rate:.0%})")

    print("-" * 45 + "\n")

